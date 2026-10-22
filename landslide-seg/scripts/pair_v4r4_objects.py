"""三架构对象级配对（v4r4 §7）。

用途：把 LateDetail / Self / Cross 按**同一 GT 对象身份**对齐，回答
"新增命中了哪些对象、失去了哪些对象"，而不是只比一个聚合百分比。

做法：
- 每个瓦片只做一次 GT 连通域标记，``gt_id`` 由 GT 掩膜唯一确定，
  因此三架构共用同一套对象身份；
- 各架构各自从概率图取 0.5 阈值预测，做 8 邻接标记与一对一贪心匹配
  （与 ``segformer_metrics.match_objects`` 完全同一实现，IoU≥0.25）；
- 输出逐对象 CSV + 按尺寸档/事件的计数表 + 三架构共同漏检清单。

不重新推理：概率图来自 ``evaluate_segformer.py --save-predictions``。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from landslide_seg.training.segformer_metrics import (  # noqa: E402
    label_objects,
    match_objects,
    object_table,
)

BINS = [("1_4", 1, 4), ("5_9", 5, 9), ("10_19", 10, 19), ("20_49", 20, 49),
        ("50_199", 50, 199), ("ge200", 200, 10**9)]
IOU_THR = 0.25
THRESHOLD = 0.5


def bin_of(area: int) -> str:
    for name, lo, hi in BINS:
        if lo <= area <= hi:
            return name
    return "ge200"


def load_pred(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    ids = [str(s) for s in z["sample_ids"].tolist()]
    return {"prob": z["probability"], "gt": z["gt"], "valid": z["valid"], "ids": ids, "path": str(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--late", required=True)
    ap.add_argument("--self", dest="self_", required=True)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--late-name", default="LateDetail")
    ap.add_argument("--tag", default="best_val")
    ap.add_argument("--out-dir", default="reports/v4r4_results")
    args = ap.parse_args()

    arms = {"late": load_pred(Path(args.late)), "self": load_pred(Path(args.self_)), "cross": load_pred(Path(args.cross))}
    ref_ids = arms["late"]["ids"]
    for k, v in arms.items():
        if v["ids"] != ref_ids:
            raise SystemExit(f"[fatal] {k} 的样本顺序与 LateDetail 不一致，不能配对")
        for field in ("gt", "valid"):
            if not np.array_equal(v[field], arms["late"][field]):
                raise SystemExit(f"[fatal] {k} 的 {field} 与 LateDetail 不一致")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, sid in enumerate(ref_ids):
        gt = arms["late"]["gt"][i].astype(bool)
        valid = arms["late"]["valid"][i].astype(bool)
        gt_lab, n_gt = label_objects(gt, 2)
        gt_objs = object_table(gt_lab, n_gt, valid)
        gt_eval = [o for o in gt_objs if not o["truncated"]]
        if not gt_objs:
            continue
        matched, matched_all = {}, {}
        for k, v in arms.items():
            pred = (v["prob"][i] >= THRESHOLD) & valid
            p_lab, n_p = label_objects(pred, 2)
            p_objs = object_table(p_lab, n_p, None)
            # v2: match evaluable GT separately; truncated GT must not compete
            # for predictions when calculating evaluable-object recall.
            m_all = match_objects(gt_lab, gt_objs, p_lab, p_objs, IOU_THR)
            m_eval = match_objects(gt_lab, gt_eval, p_lab, p_objs, IOU_THR)
            m = {o["id"]: source[o["id"]] for o in gt_objs
                 for source in (m_all if o["truncated"] else m_eval,)
                 if o["id"] in source}
            matched[k] = m
            matched_all[k] = m_all
        for o in gt_objs:
            gid = o["id"]
            rows.append({
                "sample_id": sid,
                "gt_id": gid,
                "area_px": o["area"],
                "bin": bin_of(o["area"]),
                "truncated": bool(o["truncated"]),
                "matched_late": gid in matched["late"],
                "matched_self": gid in matched["self"],
                "matched_cross": gid in matched["cross"],
                **{f"matched_all_{k}": gid in matched_all[k] for k in arms},
                "iou_late": round(matched["late"].get(gid, {}).get("iou", 0.0), 6),
                "iou_self": round(matched["self"].get(gid, {}).get("iou", 0.0), 6),
                "iou_cross": round(matched["cross"].get(gid, {}).get("iou", 0.0), 6),
            })

    csv_path = out_dir / f"objects_paired_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 汇总 ----
    def tally(sel, *, all_gt=False) -> dict:
        sub = [r for r in rows if sel(r)]
        n = len(sub)
        prefix = "matched_all_" if all_gt else "matched_"
        hit = {a: sum(1 for r in sub if r[prefix + a]) for a in ("late", "self", "cross")}
        return {"n_objects": n, "matched": hit,
                "recall": {a: (hit[a] / n if n else None) for a in hit}}

    summary = {"n_tiles": len(ref_ids), "n_objects": len(rows), "threshold": THRESHOLD,
               "iou_threshold": IOU_THR, "arms": {k: v["path"] for k, v in arms.items()},
               "overall": tally(lambda r: True, all_gt=True),
               "evaluable": tally(lambda r: not r["truncated"]),
               "by_bin": {name: tally(lambda r, b=name: r["bin"] == b and not r["truncated"])
                          for name, _, _ in BINS}}
    summary["metric_version"] = "v2_evaluable_matching"
    summary["evaluations_verified"] = {}
    for name, data in arms.items():
        evaluation = Path(data["path"]).parent / f"eval_{args.tag}.json"
        if not evaluation.is_file():
            raise FileNotFoundError(f"Need official evaluation to validate paired counts: {evaluation}")
        original = json.loads(evaluation.read_text(encoding="utf-8"))["overall"]["objects"]
        assert summary["evaluable"]["n_objects"] == original["n_gt_evaluable"]
        assert summary["evaluable"]["matched"][name] == original["n_matched_evaluable"], name
        assert summary["overall"]["matched"][name] == original["n_matched_all"], name
        for size, values in summary["by_bin"].items():
            assert values["matched"][name] == original["fine_bins"][size]["n_matched_evaluable"], (name, size)
        summary["evaluations_verified"][name] = str(evaluation)

    # ---- 新增命中 / 失去命中 ----
    def transitions(a: str, b: str, sel) -> dict:
        sub = [r for r in rows if sel(r)]
        gained = [r for r in sub if r[f"matched_{b}"] and not r[f"matched_{a}"]]
        lost = [r for r in sub if r[f"matched_{a}"] and not r[f"matched_{b}"]]
        return {"gained": len(gained), "lost": len(lost),
                "gained_by_bin": {name: sum(1 for r in gained if r["bin"] == name) for name, _, _ in BINS},
                "lost_by_bin": {name: sum(1 for r in lost if r["bin"] == name) for name, _, _ in BINS}}

    summary["cross_vs_self"] = transitions("self", "cross", lambda r: not r["truncated"])
    summary["cross_vs_late"] = transitions("late", "cross", lambda r: not r["truncated"])
    summary["self_vs_late"] = transitions("late", "self", lambda r: not r["truncated"])

    # ---- 三架构共同漏检（可对照 v4r3 的 172/63）----
    miss_all = [r for r in rows if not r["truncated"] and not any(r[f"matched_{a}"] for a in ("late", "self", "cross"))]
    summary["common_miss_all_three"] = {
        "n": len(miss_all),
        "by_bin": {name: sum(1 for r in miss_all if r["bin"] == name) for name, _, _ in BINS},
        "le_19px": sum(1 for r in miss_all if r["area_px"] <= 19),
        "le_4px": sum(1 for r in miss_all if r["area_px"] <= 4),
    }

    (out_dir / f"objects_summary_{args.tag}.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    md = [f"# 三架构对象级配对（{args.tag}）\n",
          f"对象总数 {summary['n_objects']}，瓦片 {summary['n_tiles']}；阈值 {THRESHOLD}，匹配 IoU≥{IOU_THR}。\n",
          "## 按尺寸档的对象召回（仅可评估对象）\n",
          "| 尺寸档 | 对象数 | LateDetail | Self | Cross |",
          "|---|---:|---:|---:|---:|"]
    for name, _, _ in BINS:
        b = summary["by_bin"][name]
        def f(a):
            v = b["recall"][a]
            return f"{v:.3f} ({b['matched'][a]}/{b['n_objects']})" if v is not None else "-"
        md.append(f"| {name} | {b['n_objects']} | {f('late')} | {f('self')} | {f('cross')} |")
    for label, key in (("Cross − Self", "cross_vs_self"), ("Cross − LateDetail", "cross_vs_late"),
                       ("Self − LateDetail", "self_vs_late")):
        t = summary[key]
        md.append(f"\n**{label}**：新增命中 {t['gained']}，失去命中 {t['lost']}")
        md.append(f"- 新增按档：{t['gained_by_bin']}")
        md.append(f"- 失去按档：{t['lost_by_bin']}")
    cm = summary["common_miss_all_three"]
    md.append(f"\n**三架构共同漏检**：{cm['n']} 个（≤19px {cm['le_19px']}，≤4px {cm['le_4px']}）；按档 {cm['by_bin']}")
    (out_dir / f"objects_paired_{args.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary["overall"], indent=1, ensure_ascii=False))
    print(json.dumps({k: summary["by_bin"][k]["recall"] for k in summary["by_bin"]}, indent=1, ensure_ascii=False))
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()

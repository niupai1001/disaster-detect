"""EVENT_CV_V5 结果汇总：四折控制组 + 16 个冻结 probe（+ 预留 U-TAE 配对）。

产出三张主表：
1. 跨事件完整系统与 probe（inner best / outer best，按折）
2. 事件组宏平均与最差组（不可拆分事件组）
3. 成本与完成状态
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BINS = ["1_4", "5_9", "10_19", "20_49", "50_199", "ge200"]
FOLDS = ("F1", "F2", "F3", "F4")
PROBES = {"sf_b0_rgb": "Probe-segformer_b0", "sf_b2_rgb": "Probe-segformer_b2",
          "dofa_rgb": "Probe-dofa_rgb", "dofa_10": "Probe-dofa_10"}
CONTROL = "SegFormer-LateDetail-EventCVV5"
GROUPS = {  # 不可拆分事件组
    "italy": ["italy"], "kyrgyz": ["kyrgyzstan1", "kyrgyzstan2"], "nz": ["newzealand"],
    "jp_a": ["hiroshima"], "jp_b": ["hokkaido"], "cn": ["china"], "in_th": ["thrissur"],
    "af": ["chimanimani"], "carib": ["dominicamaria"], "id": ["indonesia"],
    "ph": ["itogon"], "pr": ["usa_puertorico"],
}


def load_run(d: Path) -> dict | None:
    p = d / "train_summary.json"
    if not p.is_file():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    out = {"n_train": s.get("n_train"), "n_val": s.get("n_val"), "best_step": s.get("best_step"),
           "best_val_iou": s.get("best_val_iou"), "seconds": s.get("seconds"),
           "params": (s.get("params") or {}).get("total"), "peak": s.get("peak_alloc_mib")}
    for tag in ("best_val", "last_val", "best_test", "last_test"):
        f = d / "eval" / f"eval_{tag}.json"
        if not f.is_file():
            continue
        e = json.loads(f.read_text(encoding="utf-8"))
        px = e["overall"]["pixel"]
        out[tag] = {"iou": px.get("iou"), "ap": px.get("average_precision"),
                    "precision": px.get("precision"), "recall": px.get("recall"),
                    "boundary_f1": e["overall"]["boundary"].get("f1"),
                    "by_event": {k: v["pixel"] for k, v in e["by_event"].items()}}
        ob = d / "eval" / f"objects_{tag}.json"
        if ob.is_file():
            agg = {b: {"n": 0, "hit": 0} for b in BINS}
            for row in json.loads(ob.read_text(encoding="utf-8")):
                for b in BINS:
                    x = (row.get("fine_bins") or {}).get(b)
                    if x:
                        agg[b]["n"] += x["n_objects_evaluable"]
                        agg[b]["hit"] += x["n_matched_evaluable"]
            out[tag]["scale"] = {b: (v["hit"] / v["n"] if v["n"] else None) for b, v in agg.items()}
    return out


def f4(v, n=4):
    return "-" if v is None else f"{v:.{n}f}"


def group_stats(by_event: dict) -> tuple[float | None, float | None, str]:
    vals = []
    for g, evs in GROUPS.items():
        xs = [by_event[e]["iou"] for e in evs if e in by_event and by_event[e].get("iou") is not None]
        if xs:
            vals.append((g, sum(xs) / len(xs)))
    if not vals:
        return None, None, "-"
    macro = sum(v for _, v in vals) / len(vals)
    worst = min(vals, key=lambda t: t[1])
    return macro, worst[1], worst[0]


def main() -> None:
    runs: dict[str, dict[str, dict]] = {"control": {}, **{k: {} for k in PROBES}}
    for fold in FOLDS:
        d = ROOT / f"outputs/segformer/event_cv_v5/{CONTROL}-{fold}/1234"
        r = load_run(d)
        if r:
            runs["control"][fold] = r
        for short, exp in PROBES.items():
            r = load_run(ROOT / f"outputs/segformer/event_cv_v5_probe/{exp}-{fold}/1234")
            if r:
                runs[short][fold] = r

    out_dir = ROOT / "reports" / "event_cv_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# EVENT_CV_V5 结果\n",
         "开发型事件交叉验证；外部留出不参与任何选择，选模只看内部验证的事件平均 IoU@0.5。\n",
         "## 表 1 · 跨事件完整系统与冻结 probe（best 检查点）\n",
         "| 组 | 折 | 训练块 | inner IoU | **outer IoU** | outer AP | outer Recall | 边界 F1 | 1–4px 召回 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    order = [("control", "完整系统（SegFormer-B0+细节）")] + [(k, k) for k in PROBES]
    for key, label in order:
        for fold in FOLDS:
            v = runs.get(key, {}).get(fold)
            if not v or "best_test" not in v:
                L.append(f"| {label} | {fold} | - | - | - | - | - | - | - |")
                continue
            b, o = v.get("best_val") or {}, v["best_test"]
            sc = (o.get("scale") or {}).get("1_4")
            L.append(f"| {label} | {fold} | {v['n_train']} | {f4((b or {}).get('iou'))} | "
                     f"**{f4(o['iou'])}** | {f4(o['ap'])} | {f4(o['recall'])} | {f4(o['boundary_f1'])} | {f4(sc)} |")

    L += ["\n## 表 2 · 事件组宏平均与最差组（outer，best）\n",
          "| 组 | 折 | 宏平均 IoU | 最差组 | 最差组 IoU | 逐事件 IoU |",
          "|---|---|---:|---|---:|---|"]
    for key, label in order:
        for fold in FOLDS:
            v = runs.get(key, {}).get(fold)
            if not v or "best_test" not in v:
                continue
            be = v["best_test"]["by_event"]
            macro, worst_v, worst_g = group_stats(be)
            ev = " ".join(f"{k}:{f4(vv['iou'],3)}" for k, vv in sorted(be.items()))
            L.append(f"| {label} | {fold} | {f4(macro)} | {worst_g} | {f4(worst_v)} | {ev} |")

    L += ["\n## 表 3 · 成本与完成状态\n",
          "| 组 | 折 | 参数量 | 可训练参数 | 训练块 | 耗时 | 峰值显存 | 状态 |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for key, label in order:
        for fold in FOLDS:
            v = runs.get(key, {}).get(fold)
            if not v:
                L.append(f"| {label} | {fold} | - | - | - | - | - | 未完成 |")
                continue
            L.append(f"| {label} | {fold} | {v.get('params', 0):,} | - | {v['n_train']} | "
                     f"{f4(v.get('seconds'), 0)} s | {f4(v.get('peak'), 0)} MiB | 完成 |")

    (out_dir / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\n[saved] {out_dir}/results_summary.md")


if __name__ == "__main__":
    main()

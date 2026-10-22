"""三架构对照汇总（v4r4 §8）。

读三份 run 产物（LateDetail 复用 v4r1 detailres_n100，Self/Cross 为本轮），
输出统一对照表：DEV 均值（best 与固定 1200 步两种口径）、逐事件 IoU、
合并 IoU/Dice/Precision/Recall/AP、边界 F1、对象召回、参数/显存/耗时。

两种口径都给，是因为 DEV 同时是选择集：``best`` 会系统性偏高，
固定步数读数才可横向比（见 reports/v4_review_2026-10-20.md §2.1）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ARMS = {
    "LateDetail": "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234",
    "Self": "outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_self_residual_n100_u3600_s1234/1234",
    "Cross": "outputs/segformer/s2_10band_v4r4/segformer_b0_detail_early_cross_exchange_n100_u3600_s1234/1234",
}
FIXED_STEPS = [500, 1200, 2400, 3600]


def load_run(d: Path) -> dict:
    rel = d.relative_to(ROOT)
    summary = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
    out = {
        "run_dir": str(rel).replace("\\", "/"),
        "n_train": summary.get("n_train"),
        "n_val": summary.get("n_val"),
        "params": summary.get("params"),
        "best_val_iou": summary.get("best_val_iou"),
        "best_step": summary.get("best_step"),
        "target_updates": summary.get("target_updates"),
        "complete": summary.get("complete"),
        "samples_exposed": summary.get("samples_exposed"),
        "seconds": summary.get("seconds"),
        "peak_alloc_mib": summary.get("peak_alloc_mib"),
        "peak_reserved_mib": summary.get("peak_reserved_mib"),
        "dev_steps": {},
        "train_iou_steps": {},
    }
    for row in summary.get("history", []):
        s = int(row["global_step"])
        if s in FIXED_STEPS:
            dev = row.get("dev") or {}
            out["dev_steps"][s] = dev.get("event_mean_iou")
            out["train_iou_steps"][s] = (row.get("train_clean") or {}).get("pooled", {}).get("iou")
    for tag, split in (("best_val", "val"), ("last_val", "val"), ("best_train", "train"), ("last_train", "train")):
        p = d / "eval" / f"eval_{tag}.json"
        if not p.is_file():
            continue
        e = json.loads(p.read_text(encoding="utf-8"))
        o = e["overall"]
        px = o["pixel"]
        by = {k: v["pixel"]["iou"] for k, v in e["by_event"].items()}
        out[tag] = {
            "split": split,
            "n_samples": e["n_samples"],
            "event_mean_iou": e.get("event_mean_iou"),
            "by_event": by,
            "iou": px.get("iou"), "dice": px.get("dice"),
            "precision": px.get("precision"), "recall": px.get("recall"),
            "ap": px.get("average_precision"),
            "fpr": px.get("false_positive_rate"),
            "boundary_f1": o["boundary"].get("f1"),
            "objects": o.get("objects"),
            "latency": (e.get("efficiency") or {}).get("latency_b1"),
            "eval_peak_mib": (e.get("efficiency") or {}).get("inference_peak_alloc_mib"),
        }
    return out


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/v4r4_results")
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, rel in ARMS.items():
        d = ROOT / rel
        if not (d / "train_summary.json").is_file():
            print(f"[skip] {name}: 缺少 train_summary.json ({rel})")
            continue
        data[name] = load_run(d)
    if len(data) < 2:
        raise SystemExit("可用 run 不足，无法对照")

    (out_dir / "arms_summary.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# v4r4 三架构对照\n"]
    L.append("`LateDetail` 复用 v4r1 `detailres_n100`（配置除 exchange 外逐项相同）；"
             "`Self` 为容量对照，`Cross` 为真交换。三者同一 seed=1234、同一训练池 2,223、同一 3,600 更新预算。\n")

    L.append("## 1. DEV 两种口径\n")
    L.append("| 架构 | best DEV | best@步 | DEV@500 | DEV@1200 | DEV@2400 | DEV@3600 | 训练 IoU@3600 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for n, v in data.items():
        d = v["dev_steps"]
        L.append(f"| {n} | {fmt(v['best_val_iou'])} | {v['best_step']} | {fmt(d.get(500))} | "
                 f"{fmt(d.get(1200))} | {fmt(d.get(2400))} | {fmt(d.get(3600))} | {fmt(v['train_iou_steps'].get(3600))} |")

    L.append("\n## 2. best 检查点：DEV 逐事件\n")
    evs = sorted({e for v in data.values() for e in (v.get("best_val") or {}).get("by_event", {})})
    L.append("| 架构 | 事件均值 | " + " | ".join(evs) + " | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 |")
    L.append("|---|---:|" + "---:|" * len(evs) + "---:|---:|---:|---:|---:|---:|")
    for n, v in data.items():
        b = v.get("best_val") or {}
        if not b:
            L.append(f"| {n} | - | " + " | ".join("-" for _ in evs) + " | - | - | - | - | - | - |")
            continue
        cells = " | ".join(fmt(b["by_event"].get(e)) for e in evs)
        L.append(f"| {n} | {fmt(b['event_mean_iou'])} | {cells} | {fmt(b['iou'])} | {fmt(b['dice'])} | "
                 f"{fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['ap'])} | {fmt(b['boundary_f1'])} |")

    L.append("\n## 3. last 检查点：DEV\n")
    L.append("| 架构 | 事件均值 | " + " | ".join(evs) + " | 合并 IoU | AP |")
    L.append("|---|---:|" + "---:|" * len(evs) + "---:|---:|")
    for n, v in data.items():
        b = v.get("last_val") or {}
        if not b:
            L.append(f"| {n} | - | " + " | ".join("-" for _ in evs) + " | - | - |")
            continue
        cells = " | ".join(fmt(b["by_event"].get(e)) for e in evs)
        L.append(f"| {n} | {fmt(b['event_mean_iou'])} | {cells} | {fmt(b['iou'])} | {fmt(b['ap'])} |")

    L.append("\n## 4. 对象召回（可评估对象）\n")
    L.append("| 架构 | 检查点 | GT 对象 | 可评估 | 命中(可评估) | 对象召回 | 未匹配预测对象/瓦片 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for n, v in data.items():
        for tag in ("best_val",):
            b = v.get(tag)
            if not b:
                continue
            o = b.get("objects") or {}
            L.append(f"| {n} | {tag} | {o.get('n_gt')} | {o.get('n_gt_evaluable')} | "
                     f"{o.get('n_matched_evaluable')} | {fmt(o.get('object_recall_evaluable'),3)} | "
                     f"{fmt(o.get('unmatched_prediction_objects_per_tile'))} |")

    L.append("\n## 5. TRAIN 侧（同检查点）\n")
    L.append("| 架构 | TRAIN 合并 IoU | TRAIN 事件均值 | TRAIN AP |")
    L.append("|---|---:|---:|---:|")
    for n, v in data.items():
        b = v.get("best_train") or {}
        L.append(f"| {n} | {fmt(b.get('iou'))} | {fmt(b.get('event_mean_iou'))} | {fmt(b.get('ap'))} |")

    L.append("\n## 6. 成本\n")
    L.append("| 架构 | 总参数 | 相比 LateDetail | 训练秒 | 训练峰值显存 MiB | 推理 p50 ms | 推理峰值 MiB | 更新/曝光 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    base = (data.get("LateDetail") or {}).get("params", {}).get("total")
    for n, v in data.items():
        p = v.get("params") or {}
        b = v.get("best_val") or {}
        extra = (p.get("total", 0) - base) if base else None
        L.append(f"| {n} | {p.get('total', 0):,} | {extra:+,} | {fmt(v['seconds'],1)} | "
                 f"{fmt(v.get('peak_alloc_mib'),0)} | {fmt((b.get('latency') or {}).get('p50_ms'),2)} | "
                 f"{fmt(b.get('eval_peak_mib'),0)} | {v.get('samples_exposed')} |")

    md = out_dir / "arms_comparison.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {md}")


if __name__ == "__main__":
    main()

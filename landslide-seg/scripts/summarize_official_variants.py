"""官方单期四组对照：基线 / +DEM / +坡度 / pos_weight=5。

同一协议、同一批样本、同一预算、同一 seed，只改一个因素。
输出 VAL 与 TEST 两个 split 的 best/last 指标、逐事件、尺度召回，以及相对基线的差值。
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

RUNS = {
    "baseline": "outputs/segformer/official_single/SegFormer-LateDetail-OfficialSplit-SingleDate/1234",
    "DEM": "outputs/segformer/official_single_geo/SegFormer-LateDetail-OfficialSingle-DEM/1234",
    "Slope": "outputs/segformer/official_single_geo/SegFormer-LateDetail-OfficialSingle-Slope/1234",
    "DEMSlope": "outputs/segformer/official_single_geo/SegFormer-LateDetail-OfficialSingle-DEMSlope/1234",
    "PW5": "outputs/segformer/official_single/SegFormer-LateDetail-OfficialSingle-PW5/1234",
}
BINS = ["1_4", "5_9", "10_19", "20_49", "50_199", "ge200"]


def collect(d: Path) -> dict:
    s = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
    out = {"n_train": s.get("n_train"), "n_val": s.get("n_val"),
           "best_val_iou": s.get("best_val_iou"), "best_step": s.get("best_step"),
           "selection_metric": s.get("selection_metric"),
           "seconds": s.get("seconds"), "params": s.get("params"),
           "peak_alloc_mib": s.get("peak_alloc_mib"),
           "in_chans": (json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
                        .get("extra_channel", "none"))}
    for tag in ("best_val", "last_val", "best_test", "last_test"):
        p = d / "eval" / f"eval_{tag}.json"
        if not p.is_file():
            continue
        e = json.loads(p.read_text(encoding="utf-8"))
        o, px = e["overall"], e["overall"]["pixel"]
        out[tag] = {
            "n": e["n_samples"], "event_mean": e.get("event_mean_iou"),
            "iou": px.get("iou"), "dice": px.get("dice"), "precision": px.get("precision"),
            "recall": px.get("recall"), "ap": px.get("average_precision"),
            "boundary_f1": o["boundary"].get("f1"), "objects": o.get("objects"),
            "by_event": {k: v["pixel"]["iou"] for k, v in e["by_event"].items()},
        }
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
            out[tag]["scale_n"] = {b: v["n"] for b, v in agg.items()}
    return out


def f(v, n=4):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main() -> None:
    data = {}
    for k, rel in RUNS.items():
        d = ROOT / rel
        if (d / "train_summary.json").is_file():
            data[k] = collect(d)
        else:
            print(f"[skip] {k}")
    if "baseline" not in data:
        raise SystemExit("缺基线")
    out_dir = ROOT / "reports" / "official_single"
    (out_dir / "variants_summary.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# 官方单期：基线 / +DEM / +坡度 / pos_weight=5\n",
         "同一官方 split、同一批 3,012 训练块、同 3,600 更新、同 seed；每组只改一个因素。\n",
         "## 1. 训练\n",
         "| 组 | 输入通道 | 参数量 | best@步 | VAL 选模值 | 耗时 | 峰值显存 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in data.items():
        L.append(f"| {k} | {v['in_chans']} | {(v.get('params') or {}).get('total',0):,} | {v['best_step']} | "
                 f"{f(v['best_val_iou'])} | {f(v['seconds'],0)} s | {f(v['peak_alloc_mib'],0)} MiB |")

    for split in ("val", "test"):
        kind = "VAL" if split == "val" else "TEST"
        L.append(f"\n## 2.{1 if split=='val' else 2} {kind}（best 检查点）\n")
        L.append("| 组 | 合并 IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in data.items():
            b = v.get(f"best_{split}")
            if not b:
                continue
            o = b.get("objects") or {}
            L.append(f"| {k} | {f(b['iou'])} | {f(b['dice'])} | {f(b['precision'])} | {f(b['recall'])} | "
                     f"{f(b['ap'])} | {f(b['boundary_f1'])} | {f(o.get('object_recall_evaluable'))} |")
        base = data["baseline"].get(f"best_{split}")
        if base:
            L.append(f"\n**相对基线的差值（{kind}，best）**\n")
            L.append("| 组 | ΔIoU | ΔDice | ΔPrecision | ΔRecall | ΔAP | Δ边界F1 |")
            L.append("|---|---:|---:|---:|---:|---:|---:|")
            for k, v in data.items():
                if k == "baseline":
                    continue
                b = v.get(f"best_{split}")
                if not b:
                    continue
                L.append(f"| {k} | {b['iou']-base['iou']:+.4f} | {b['dice']-base['dice']:+.4f} | "
                         f"{b['precision']-base['precision']:+.4f} | {b['recall']-base['recall']:+.4f} | "
                         f"{b['ap']-base['ap']:+.4f} | {(b['boundary_f1'] or 0)-(base['boundary_f1'] or 0):+.4f} |")

    L.append("\n## 3. TEST 尺度召回（对象级，best）\n")
    L.append("| 组 | " + " | ".join(f"{b} px" for b in BINS) + " |")
    L.append("|---|" + "---:|" * len(BINS))
    for k, v in data.items():
        b = v.get("best_test") or {}
        sc = b.get("scale")
        if not sc:
            continue
        L.append(f"| {k} | " + " | ".join(f"{f(sc[b],3)}" for b in BINS) + " |")

    L.append("\n## 4. last 检查点（第 3600 步）\n")
    L.append("| 组 | VAL IoU | VAL AP | TEST IoU | TEST AP |")
    L.append("|---|---:|---:|---:|---:|")
    for k, v in data.items():
        a, b = v.get("last_val") or {}, v.get("last_test") or {}
        L.append(f"| {k} | {f(a.get('iou'))} | {f(a.get('ap'))} | {f(b.get('iou'))} | {f(b.get('ap'))} |")

    L.append("\n## 5. 说明\n")
    L.append("- 六组同协议、同样本、同预算；DEM 与坡度组的标签与 valid 直接沿用基线缓存，逐块相同。")
    L.append("- 新增通道在首层卷积零初始化，初始输出与基线逐位一致，差异只来自训练。")
    L.append("- 单 seed，不构成统计显著；差值小于 0.01 的应按噪声看待。")

    md = out_dir / "variants_summary.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {md}")


if __name__ == "__main__":
    main()

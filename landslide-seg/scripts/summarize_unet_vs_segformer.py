"""八组对照：SegFormer 与 U-Net 各自跑基线 / +DEM / +坡度 / pos_weight=5。

关心的不是跨架构绝对分数（参数量差 3.8 倍、LR 未为 U-Net 调），而是
**同一组改动在两种架构上是否复现同样的方向**。
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

FAMILIES = {
    "SegFormer": {
        "baseline": "outputs/segformer/official_single/SegFormer-LateDetail-OfficialSplit-SingleDate/1234",
        "DEM": "outputs/segformer/official_single_geo/SegFormer-LateDetail-OfficialSingle-DEM/1234",
        "Slope": "outputs/segformer/official_single_geo/SegFormer-LateDetail-OfficialSingle-Slope/1234",
        "DEMSlope": "outputs/segformer/official_single_geo/SegFormer-LateDetail-OfficialSingle-DEMSlope/1234",
        "PW5": "outputs/segformer/official_single/SegFormer-LateDetail-OfficialSingle-PW5/1234",
    },
    "UNet": {
        "baseline": "outputs/segformer/official_unet/UNet-ResNet18-OfficialSingle-Baseline/1234",
        "DEM": "outputs/segformer/official_unet/UNet-ResNet18-OfficialSingle-DEM/1234",
        "Slope": "outputs/segformer/official_unet/UNet-ResNet18-OfficialSingle-Slope/1234",
        "PW5": "outputs/segformer/official_unet/UNet-ResNet18-OfficialSingle-PW5/1234",
    },
}
BINS = ["1_4", "5_9", "10_19", "20_49", "50_199", "ge200"]


def collect(d: Path) -> dict:
    s = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    out = {"params": (s.get("params") or {}).get("total"),
           "best_step": s.get("best_step"), "best_val_iou": s.get("best_val_iou"),
           "seconds": s.get("seconds"), "peak": s.get("peak_alloc_mib"),
           "type": meta.get("type", "?"), "extra": meta.get("extra_channel", "none")}
    for tag in ("best_val", "last_val", "best_test", "last_test"):
        p = d / "eval" / f"eval_{tag}.json"
        if not p.is_file():
            continue
        e = json.loads(p.read_text(encoding="utf-8"))
        px = e["overall"]["pixel"]
        out[tag] = {"iou": px.get("iou"), "dice": px.get("dice"), "precision": px.get("precision"),
                    "recall": px.get("recall"), "ap": px.get("average_precision"),
                    "boundary_f1": e["overall"]["boundary"].get("f1"),
                    "obj_recall": (e["overall"].get("objects") or {}).get("object_recall_evaluable"),
                    "event_mean": e.get("event_mean_iou")}
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


def f(v, n=4):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main() -> None:
    data: dict[str, dict] = {}
    for fam, arms in FAMILIES.items():
        data[fam] = {}
        for arm, rel in arms.items():
            d = ROOT / rel
            if (d / "train_summary.json").is_file():
                data[fam][arm] = collect(d)
            else:
                print(f"[skip] {fam}/{arm}")
    out_dir = ROOT / "reports" / "official_single"
    (out_dir / "unet_vs_segformer.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    L = ["# 两种架构 × 四组改动（官方单期协议）\n",
         "同数据、同划分、同统计、同损失、同预算（3,600 更新 × batch 64）、同 seed 1234、同选模口径。\n",
         "## 1. 训练概况\n",
         "| 架构 | 组 | 参数量 | 输入通道 | best@步 | VAL 选模值 | 耗时 | 峰值显存 |",
         "|---|---|---:|---|---:|---:|---:|---:|"]
    for fam, arms in data.items():
        for arm, v in arms.items():
            L.append(f"| {fam} | {arm} | {v['params']:,} | {v['extra']} | {v['best_step']} | "
                     f"{f(v['best_val_iou'])} | {f(v['seconds'],0)} s | {f(v['peak'],0)} MiB |")

    for split, kind in (("best_val", "VAL"), ("best_test", "TEST")):
        L.append(f"\n## 2.{1 if split=='best_val' else 2} {kind}（best 检查点）\n")
        L.append("| 架构 | 组 | IoU | Dice | Precision | Recall | AP | 边界 F1 | 对象召回 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for fam, arms in data.items():
            for arm, v in arms.items():
                b = v.get(split)
                if not b:
                    continue
                L.append(f"| {fam} | {arm} | {f(b['iou'])} | {f(b['dice'])} | {f(b['precision'])} | "
                         f"{f(b['recall'])} | {f(b['ap'])} | {f(b['boundary_f1'])} | {f(b['obj_recall'])} |")

    L.append("\n## 3. 族内差值（各组 − 本族基线）\n")
    for split, kind in (("best_val", "VAL"), ("best_test", "TEST")):
        L.append(f"### {kind}\n")
        L.append("| 组 | SegFormer ΔIoU | SegFormer ΔRecall | SegFormer ΔPrecision | UNet ΔIoU | UNet ΔRecall | UNet ΔPrecision |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm in ("DEM", "Slope", "DEMSlope", "PW5"):
            cells = []
            for fam in ("SegFormer", "UNet"):
                arms = data.get(fam, {})
                if arm in arms and "baseline" in arms and arms[arm].get(split) and arms["baseline"].get(split):
                    a, b = arms[arm][split], arms["baseline"][split]
                    cells += [f"{a['iou']-b['iou']:+.4f}", f"{a['recall']-b['recall']:+.4f}",
                              f"{a['precision']-b['precision']:+.4f}"]
                else:
                    cells += ["-", "-", "-"]
            # 顺序：SegFormer 在前
            sf, un = cells[:3], cells[3:]
            L.append(f"| {arm} | " + " | ".join(sf + un) + " |")
        L.append("")

    L.append("\n## 4. 方向一致性检验\n")
    L.append("同一改动在两种架构上 ΔIoU 与 ΔRecall 的符号是否一致（TEST，best）：\n")
    L.append("| 组 | ΔIoU 符号 | ΔRecall 符号 | 结论 |")
    L.append("|---|---|---|---|")
    for arm in ("DEM", "Slope", "DEMSlope", "PW5"):
        sig = {}
        for fam in ("SegFormer", "UNet"):
            arms = data.get(fam, {})
            if arm in arms and "baseline" in arms and arms[arm].get("best_test") and arms["baseline"].get("best_test"):
                a, b = arms[arm]["best_test"], arms["baseline"]["best_test"]
                sig[fam] = (a["iou"] - b["iou"], a["recall"] - b["recall"])
        if len(sig) == 2:
            same_iou = (sig["SegFormer"][0] > 0) == (sig["UNet"][0] > 0)
            same_rec = (sig["SegFormer"][1] > 0) == (sig["UNet"][1] > 0)
            L.append(f"| {arm} | {'一致' if same_iou else '**相反**'} | {'一致' if same_rec else '**相反**'} | "
                     f"{'可复现' if (same_iou and same_rec) else '未复现'} |")
        else:
            L.append(f"| {arm} | - | - | 缺数据 |")

    L.append("\n## 5. TEST 尺度召回（对象级，best）\n")
    L.append("| 架构 | 组 | " + " | ".join(f"{b} px" for b in BINS) + " |")
    L.append("|---|---|" + "---:|" * len(BINS))
    for fam, arms in data.items():
        for arm, v in arms.items():
            sc = (v.get("best_test") or {}).get("scale")
            if not sc:
                continue
            L.append(f"| {fam} | {arm} | " + " | ".join(f"{f(sc[b],3)}" for b in BINS) + " |")

    L.append("\n## 6. 结论边界\n")
    L.append("- U-Net 组（ResNet18-UNet，14.35M）与 SegFormer 组（3.75M）参数量差 3.8 倍，")
    L.append("  且学习率沿用 SegFormer 那一套，没有为 U-Net 单独调——**跨架构的绝对分数不可直接比较**。")
    L.append("- 有意义的是**族内差值**：同一改动在本族内只改一个因素，可判断方向是否复现。")
    L.append("- 单 seed；0.01 量级的差值按噪声看待。")

    md = out_dir / "unet_vs_segformer.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {md}")


if __name__ == "__main__":
    main()

"""从既有评测 JSON 提取精确率 / 召回率 / F1（Dice）并生成汇总文档。

只读既有产物，不重算、不改写历史指标：
  - outputs/segformer/{v1,v2_clean,v3_p20}/ablation_eval_metric_v2.json  （A0/A1/B0/B1，3 seeds）
  - outputs/segformer/coarse_to_fine/coarse_to_fine_analysis.json       （A1/A1_AUX/CF0/CF1/CF1_B/U0，单 seed）
  - outputs/segformer/single_head/eval_v2.json                          （CF0_NOAUX/SINGLE_SUM/SINGLE_AFF，单 seed）

用法：
    python scripts/make_prf1_table.py [--out docs/metrics_precision_recall_f1_2026-10-19.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "outputs" / "segformer"

NAME = {
    "A0": "SegFormer 直接多光谱分割",
    "A1": "SegFormer 原网格细节残差分割",
    "B0": "SegFormer 光谱投影分割",
    "B1": "SegFormer 光谱投影与细节残差分割",
    "A1_AUX": "细节残差分割·双输出监督",
    "CF0": "空间残差分割·双输出监督",
    "CF1": "概率引导空间残差分割",
    "CF1_B": "概率引导空间残差分割·边界加权",
    "CF0_NOAUX": "空间残差分割·最终输出监督",
    "SINGLE_SUM": "单头语义—细节相加融合",
    "SINGLE_AFF": "单头语义—细节局部注意力融合",
    "U0": "ResNet18 U-Net 多光谱分割",
}

VERSIONS = [
    ("s2_10band_v1", "v1"),
    ("s2_10band_v2_clean", "v2_clean"),
    ("s2_10band_v3_p20", "v3"),
]

VARIANT_ORDER = ["A1", "A1_AUX", "CF0", "CF1", "CF1_B", "CF0_NOAUX", "SINGLE_SUM", "SINGLE_AFF", "U0"]


def load_aggregate(version: str) -> dict:
    return json.loads((SEG / version / "ablation_eval_metric_v2.json").read_text(encoding="utf-8"))


def aggregate_metrics(version: str, exp: str, split: str) -> dict:
    block = load_aggregate(version)["aggregate"][exp][split]
    keys = {"P": "pixel.precision", "R": "pixel.recall", "F1": "pixel.dice", "IoU": "pixel.iou"}
    return {k: (block[v]["mean"], block[v]["std"], block[v]["values"]) for k, v in keys.items()}


def variant_metrics() -> dict[str, dict[str, dict]]:
    """返回 {exp: {split: {"final": {...}, "coarse": {...} | None}}}，单 seed。"""
    out: dict[str, dict[str, dict]] = {}

    c2f = json.loads((SEG / "coarse_to_fine" / "coarse_to_fine_analysis.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        for exp, payload in c2f["splits"][split]["per_experiment"].items():
            out.setdefault(exp, {})[split] = {
                "final": payload["final"],
                "coarse": payload.get("coarse"),
            }

    single = json.loads((SEG / "single_head" / "eval_v2.json").read_text(encoding="utf-8"))
    for exp, payload in single["experiments"].items():
        for split in ("val", "test"):
            pooled = payload["split"][split]["pooled"]
            out.setdefault(exp, {})[split] = {
                "final": {
                    "precision": pooled["precision"],
                    "recall": pooled["recall"],
                    "dice": pooled["dice"],
                    "iou": pooled["iou"],
                },
                "coarse": None,
            }
    return out


def fmt(t: tuple[float, float, list[float]], nd: int = 4) -> str:
    return f"{t[0]:.{nd}f}±{t[1]:.{nd}f}"


def build(out_path: Path) -> str:
    variants = variant_metrics()
    lines: list[str] = []
    add = lines.append

    add("# 精确率 / 召回率 / F1：全部已跑实验的完整指标")
    add("")
    add("由 `scripts/make_prf1_table.py` 从既有评测 JSON 生成，只提取与派生，未改动任何历史结果。")
    add("")
    add("## 口径")
    add("")
    add("- 全部为 **pooled** 像素级指标（跨图累计 TP/FP/FN 后计算），最终阈值固定 **0.5**。")
    add("- `P = TP/(TP+FP)`，`R = TP/(TP+FN)`，`F1 = 2TP/(2TP+FP+FN)`（二分类下即 Dice），`IoU = TP/(TP+FP+FN)`。")
    add("- 主架构四组（A0/A1/B0/B1）为 **3 seeds**，报 mean±std；变体组为 **单 seed（1234）**，因此不给 std。")
    add("- `C`（粗输出）= 官方解码器分类器的直接结果，未经细节残差修正；U-Net 无独立 `C`，其 coarse 列与 final 相同。")
    add("- val = hiroshima 448 张（单事件）；test = italy（v3 下为 2,795 张）。两者**不可直接比较**。")
    add("")

    add("## 1. v3 主架构（3 seeds）")
    add("")
    add("| 架构 | val P | val R | val F1 | val IoU | test P | test R | test F1 | test IoU |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for exp in ("A0", "A1", "B0", "B1"):
        v = aggregate_metrics("s2_10band_v3_p20", exp, "val")
        t = aggregate_metrics("s2_10band_v3_p20", exp, "test")
        add(
            f"| {NAME[exp]} | {fmt(v['P'])} | {fmt(v['R'])} | {fmt(v['F1'])} | {fmt(v['IoU'])} | "
            f"{fmt(t['P'])} | {fmt(t['R'])} | {fmt(t['F1'])} | {fmt(t['IoU'])} |"
        )
    add("")

    add("## 2. v3 架构变体（seed 1234，单 seed）")
    add("")
    add("### 2.1 最终输出 F")
    add("")
    add("| 架构 | val P | val R | val F1 | val IoU | test P | test R | test F1 | test IoU |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for exp in VARIANT_ORDER:
        v = variants[exp]["val"]["final"]
        t = variants[exp]["test"]["final"]
        add(
            f"| {NAME[exp]} | {v['precision']:.4f} | {v['recall']:.4f} | {v['dice']:.4f} | {v['iou']:.4f} | "
            f"{t['precision']:.4f} | {t['recall']:.4f} | {t['dice']:.4f} | {t['iou']:.4f} |"
        )
    add("")

    add("### 2.2 粗输出 C（同一 checkpoint，同一阈值 0.5）")
    add("")
    add("| 架构 | val P | val R | val F1 | val IoU | test P | test R | test F1 | test IoU |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for exp in VARIANT_ORDER:
        if variants[exp]["val"].get("coarse") is None:
            continue
        v = variants[exp]["val"]["coarse"]
        t = variants[exp]["test"]["coarse"]
        add(
            f"| {NAME[exp]} | {v['precision']:.4f} | {v['recall']:.4f} | {v['dice']:.4f} | {v['iou']:.4f} | "
            f"{t['precision']:.4f} | {t['recall']:.4f} | {t['dice']:.4f} | {t['iou']:.4f} |"
        )
    add("")

    add("## 3. 三个数据版本的 val（3 seeds）")
    add("")
    add("| 数据版本 | 架构 | P | R | F1 | IoU |")
    add("|---|---|---:|---:|---:|---:|")
    for version, label in VERSIONS:
        for exp in ("A0", "A1", "B0", "B1"):
            a = aggregate_metrics(version, exp, "val")
            add(f"| {label} | {NAME[exp]} | {fmt(a['P'])} | {fmt(a['R'])} | {fmt(a['F1'])} | {fmt(a['IoU'])} |")
    add("")

    add("## 4. 派生量：预测面积 / GT 面积（= R / P）")
    add("")
    add("该比值等于 `(TP+FP)/(TP+FN)`，即实际输出面积相对真值的倍数。>1 表示整体过检，<1 表示整体漏检；")
    add("它把 0.5 这个固定阈值在各组、各域上的实际落点显式化。")
    add("")
    add("| 架构 | val 比值 | test 比值 | test/val |")
    add("|---|---:|---:|---:|")
    for exp in VARIANT_ORDER:
        v = variants[exp]["val"]["final"]
        t = variants[exp]["test"]["final"]
        rv = v["recall"] / v["precision"]
        rt = t["recall"] / t["precision"]
        add(f"| {NAME[exp]} | {rv:.3f} | {rt:.3f} | {rt / rv:.2f}× |")
    add("")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/metrics_precision_recall_f1_2026-10-19.md")
    args = ap.parse_args()
    out_path = ROOT / args.out
    text = build(out_path)
    print(f"written {out_path.relative_to(ROOT)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()

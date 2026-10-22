"""生成本轮汇总报告所需的图：架构对比、训练曲线、尺度召回、数据构成。"""

from __future__ import annotations

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "outputs" / "segformer" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig_arch_compare() -> None:
    rows = [
        ("Direct10 + detail residual", 0.3068, 0.0683),
        ("Direct10 + conditional refiner (P=pred)", 0.3158, 0.0628),
        ("Direct10 + conditional refiner (P=0)", 0.3135, 0.0632),
        ("Direct10 + refiner + boundary-weighted", 0.3143, 0.0664),
        ("Direct10 + detail residual + coarse aux", 0.2976, 0.0639),
        ("Direct10 + refiner (P=0), no coarse aux", 0.2987, 0.0529),
        ("Single-head: sum", 0.2692, 0.0755),
        ("Single-head: AFF", 0.2690, 0.0727),
        ("U-Net ResNet18", 0.2583, 0.0567),
        ("Direct10, no detail", 0.2459, 0.0661),
        ("10->3 projection, no detail", 0.2525, 0.0575),
        ("10->3 projection + detail residual", 0.2930, 0.0579),
    ]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    val = [r[1] for r in rows]
    test = [r[2] for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    axes[0].barh(y, val, color="#3b6ea5")
    axes[0].set_xlim(0.22, 0.34)
    axes[0].set_title("val IoU (hiroshima 448)")
    for i, v in enumerate(val):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    axes[1].barh(y, test, color="#b5651d")
    axes[1].set_xlim(0.04, 0.085)
    axes[1].set_title("test IoU (Italy 2,795, v3 set)")
    for i, v in enumerate(test):
        axes[1].text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "all_v3_val_test.png", dpi=130)
    plt.close(fig)


def fig_curves() -> None:
    root = ROOT / "outputs" / "segformer" / "s2_10band_v3_p20"
    groups = {
        "residual / refiner": {
            "Direct10 + residual": "A1",
            "conditional refiner (P=pred)": "CF1",
            "conditional refiner (P=0)": "CF0",
        },
        "single head": {
            "no coarse aux (P=0 refiner)": "CF0_NOAUX",
            "single-head sum": "SINGLE_SUM",
            "single-head AFF": "SINGLE_AFF",
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (title, expmap) in zip(axes, groups.items()):
        for label, exp in expmap.items():
            p = root / exp / "1234" / "metrics.jsonl"
            if not p.is_file():
                continue
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            ax.plot([r["epoch"] for r in rows], [r["val_iou"] for r in rows], label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("val IoU")
    axes[1].set_ylim(0, 0.34)
    fig.tight_layout()
    fig.savefig(OUT / "v3_training_curves.png", dpi=130)
    plt.close(fig)


def fig_scale() -> None:
    p = ROOT / "outputs" / "segformer" / "s2_10band_v3_p20" / "scale_gap_v3.json"
    d = json.loads(p.read_text(encoding="utf-8"))["experiments"]
    exps = ["A0", "A1", "B0", "B1"]
    labels = ["Direct10", "Direct10+res", "Proj10->3", "Proj10->3+res"]
    bins = ["small", "medium", "large"]
    colors = ["#8c8c8c", "#3b6ea5", "#b5651d", "#2e8b57"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        x = np.arange(len(bins))
        w = 0.18
        for j, (exp, lab) in enumerate(zip(exps, labels)):
            vals = [d[exp][split][b]["object_recall_evaluable"] for b in bins]
            ax.bar(x + (j - 1.5) * w, vals, w, label=lab, color=colors[j])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b} objects" for b in bins])
        ax.set_title(f"{split}: object recall by GT size")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("object recall (IoU>=0.25)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "v3_scale_recall.png", dpi=130)
    plt.close(fig)


def fig_data() -> None:
    files = {
        "italy": 5321, "dominicamaria": 1148, "chimanimani": 1133, "hiroshima": 864,
        "indonesia": 573, "hokkaido": 290, "kyrgyzstan1": 277, "itogon": 235, "china": 159,
    }
    cert = {
        "italy": 2795, "dominicamaria": 374, "chimanimani": 406, "hiroshima": 448,
        "indonesia": 270, "hokkaido": 150, "kyrgyzstan1": 133, "itogon": 119, "china": 77,
    }
    events = sorted(files, key=lambda e: files[e], reverse=True)
    x = np.arange(len(events))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, [files[e] for e in events], 0.4, label="files (10,000)", color="#b0b0b0")
    ax.bar(x + 0.2, [cert[e] for e in events], 0.4, label="certified", color="#3b6ea5")
    for i, e in enumerate(events):
        rate = cert[e] / files[e]
        ax.text(i, files[e] + 80, f"{rate*100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("tiles")
    ax.set_title("v3 data: files vs certified per event (annotation rate on top)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "v3_data_composition.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_arch_compare()
    fig_curves()
    fig_scale()
    fig_data()
    print(sorted(p.name for p in OUT.glob("*.png")))

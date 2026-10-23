"""图：U-TAE 在项目数据划分上的训练曲线（DEV 事件平均 IoU，逐 epoch）。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/research"
BASE = ROOT / "outputs/utae_v4r1/v4r1"
REF = 0.3081

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(mode: str) -> tuple[list[int], list[float], list[float]]:
    p = BASE / mode / "1234" / "metrics.jsonl"
    eps, pooled, emean = [], [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        eps.append(r["epoch"])
        pooled.append(r["inner"]["pooled"]["iou"])
        emean.append(r["inner"]["event_mean_iou"])
    return eps, pooled, emean


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax = axes[0]
    for mode, color, label in (("single", "#4C72B0", "单期（1 帧）"), ("multi", "#C44E52", "完整 15 期")):
        p = BASE / mode / "1234" / "metrics.jsonl"
        if not p.is_file():
            continue
        eps, pooled, emean = load(mode)
        ax.plot(eps, emean, color=color, lw=1.6, label=f"{label}：事件平均 IoU")
        best_i = max(range(len(emean)), key=emean.__getitem__)
        ax.plot([eps[best_i]], [emean[best_i]], "o", color=color, ms=6)
        ax.annotate(f"最佳 epoch {eps[best_i]}\n{emean[best_i]:.4f}", (eps[best_i], emean[best_i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8, color=color)
        ax.plot([eps[-1]], [emean[-1]], "s", color=color, ms=5)
    ax.axhline(REF, color="#555555", ls="--", lw=1)
    ax.text(1, REF + 0.004, f"SegFormer 十波段基线 {REF:.4f}", fontsize=8.5, color="#444444")
    ax.set_xlabel("epoch（每 epoch 139 次更新）")
    ax.set_ylabel("DEV 事件平均 IoU（广岛与特里苏尔等权）")
    ax.set_title("U-TAE 在项目数据划分上的验证曲线：峰值出现在早期，之后回落", fontsize=12)
    ax.set_ylim(0.05, 0.42)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    for offset, mode, label, color in [(-0.18, "single", "单期", "#4C72B0"),
                                       (0.18, "multi", "15期", "#C44E52")]:
        result = json.loads((BASE / mode / "1234/eval_dev_summary.json").read_text(encoding="utf-8"))["best"]
        values = [result["by_event"][ev]["pixel"]["iou"] for ev in ("hiroshima", "thrissur")]
        bars = axes[1].bar([offset, 1 + offset], values, width=.34, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.3f", fontsize=9)
        obj = result["overall"]["objects"]["tiny"]
        bar = axes[2].bar([offset], [obj["object_recall_evaluable"]], width=.34, color=color)
        axes[2].bar_label(bar, labels=[f'{obj["n_matched_evaluable"]}/{obj["n_objects_evaluable"]}'], fontsize=10)
    axes[1].set(xticks=[0, 1], xticklabels=["广岛", "特里苏尔"], ylim=(0, .45), title="最佳模型：逐事件IoU")
    axes[1].legend(frameon=False)
    axes[2].set(xticks=[-.18, .18], xticklabels=["单期", "15期"], ylim=(0, .15), title="最佳模型：1–4像素对象召回")
    fig.tight_layout()
    fig.savefig(OUT / "utae_v4r1_curves.png", dpi=180)
    fig.savefig(OUT / "utae_v4r1_curves.svg")
    print(OUT / "utae_v4r1_curves.png")


if __name__ == "__main__":
    main()

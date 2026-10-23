"""图：光谱三级（S1/S2/S3）处理组与对照相对基线的差值。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/segformer/s2_10band_v4r1_spectral_ladder"
REF = ROOT / "outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n100_u3600_s1234/1234"
OUT = ROOT / "docs/assets/research"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROWS = [("S1 显式输入·指数", "S1Indices"), ("S1 对照·原始波段", "S1Control"),
        ("S2 波长身份", "S2Wavelength"), ("S2 对照·可学习偏置", "S2Control"),
        ("S3 低秩交互 rank4", "S3LowRank"), ("S3 对照·满秩混合", "S3Control")]


def main() -> None:
    ref = json.loads((REF / "train_summary.json").read_text(encoding="utf-8"))["best_val_iou"]
    vals, ap, hi, th = [], [], [], []
    for _, key in ROWS:
        d = BASE / f"SegFormer-B0-LateDetail-OODv4-{key}" / "1234"
        s = json.loads((d / "train_summary.json").read_text(encoding="utf-8"))
        e = json.loads((d / "eval" / "eval_val.json").read_text(encoding="utf-8"))
        vals.append(s["best_val_iou"])
        ap.append(e["overall"]["pixel"]["average_precision"])
        ev = {k: v["pixel"]["iou"] for k, v in e["by_event"].items()}
        hi.append(ev["hiroshima"]); th.append(ev["thrissur"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(ROWS))
    delta = np.array(vals) - ref
    colors = ["#42876a" if d > 0 else "#bb6a63" for d in delta]
    ax.barh(y, delta, color=colors)
    ax.axvline(0, color="#454545", lw=1)
    for i, (d, v) in enumerate(zip(delta, vals)):
        ax.text(d + (0.0008 if d >= 0 else -0.0008), i, f"{v:.4f} ({d:+.4f})",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.set_yticks(y, [r[0] for r in ROWS])
    ax.invert_yaxis()
    ax.set_xlim(-0.040, 0.010)
    ax.set_xlabel(f"验证事件平均 IoU 相对基线（{ref:.4f}）的差值")
    ax.set_title("光谱三级：处理组与对照", fontsize=12)
    ax.grid(axis="x", alpha=0.25, ls=":")

    ax = axes[1]
    w = 0.38
    ax.barh(y - w / 2, np.array(hi) - 0.3114, height=w, color="#4f78a0", label="广岛（基线 0.3114）")
    ax.barh(y + w / 2, np.array(th) - 0.3049, height=w, color="#c9873f", label="特里苏尔（基线 0.3049）")
    ax.axvline(0, color="#454545", lw=1)
    ax.set_yticks(y, [r[0] for r in ROWS])
    ax.invert_yaxis()
    ax.set_xlabel("逐事件 IoU 相对各自基线的差值")
    ax.set_title("两个验证事件的方向是否一致", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(axis="x", alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(OUT / "v4r1_spectral_ladder.png", dpi=180)
    fig.savefig(OUT / "v4r1_spectral_ladder.svg")
    print(OUT / "v4r1_spectral_ladder.png")


if __name__ == "__main__":
    main()

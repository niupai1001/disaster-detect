"""图：四折 U-TAE 三种时间输入的 OUTER 结果（逐折 + 逐事件）。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "research"
FOLDS = ("F1", "F2", "F3", "F4")
MODES = ("single", "history", "multi")
COLORS = {"single": "#4C72B0", "history": "#DD8452", "multi": "#55A868"}
LABEL = {"single": "单期＋DEM", "history": "截至目标日期", "multi": "15 期"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    data = json.loads((ROOT / "outputs" / "utae_cv" / "summary.json").read_text(encoding="utf-8"))["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.05, 1.35]})

    # 左：逐折分组柱
    ax = axes[0]
    w = 0.26
    xs = np.arange(len(FOLDS))
    for k, m in enumerate(MODES):
        vals, colors = [], []
        for f in FOLDS:
            r = data[f][m]
            vals.append(r["outer"]["iou"] if r.get("status") == "done" else np.nan)
            colors.append(COLORS[m])
        ax.bar(xs + (k - 1) * w, vals, w, color=colors, label=LABEL[m],
               edgecolor="white", linewidth=0.8)
        for xi, v in zip(xs + (k - 1) * w, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f}\n(外部 {len(data[f]['single']['by_event'])} 事件)" for f in FOLDS], fontsize=9)
    ax.set_ylabel("OUTER pooled IoU（阈值 0.5）")
    ax.set_title("逐折：三种时间输入的外部留出 IoU", fontsize=11)
    ax.set_ylim(0, 0.27)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    pending = [f for f in FOLDS if data[f]["multi"].get("status") != "done"]
    status_note = "全部四折已完成" if not pending else "空白柱 = multi 尚未完成训练及评测（" + "、".join(pending) + "）"
    ax.text(0.0, -0.17, status_note, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="#666666")

    # 右：逐事件连线（单期 → 截至 / 多期）
    ax = axes[1]
    events = sorted({e for f in FOLDS for e in data[f]["single"]["by_event"]})
    ys = np.arange(len(events))
    for i, e in enumerate(events):
        fold = next(f for f in FOLDS if e in data[f]["single"]["by_event"])
        base = data[fold]["single"]["by_event"][e]["iou"]
        ax.plot([base], [i], "o", color=COLORS["single"], ms=6, zorder=3)
        for m in ("history", "multi"):
            v = (data[fold][m].get("by_event") or {}).get(e, {}).get("iou")
            if v is None:
                continue
            ax.plot([base, v], [i, i], "-", color=COLORS[m], lw=1.2, alpha=0.75)
            ax.plot([v], [i], "o", color=COLORS[m], ms=5, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{e}" for e in events], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("单个外部事件的 pooled IoU")
    ax.set_title("逐事件：连线段表示同一模型换时间输入后的变化", fontsize=11)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    for m in MODES:
        ax.plot([], [], "o-", color=COLORS[m], label=LABEL[m], ms=5, lw=1.2)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    out = ASSETS / "utae_cv_modes.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    print(out)


if __name__ == "__main__":
    main()

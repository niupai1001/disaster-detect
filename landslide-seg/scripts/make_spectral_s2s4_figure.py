"""S1/S3/S4 汇总图：事件平均 IoU（两事件方向）、微小对象召回、门控诊断。

读取 ``summary_spectral.json`` 与 ``gate_diagnostics.json``，输出
``docs/assets/research/v4r2_spectral.png``。只读结果，不训练。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "segformer" / "v4r2_spectral" / "summary_spectral.json"
GATES = ROOT / "outputs" / "segformer" / "v4r2_spectral" / "gate_diagnostics.json"
OUT = ROOT / "docs" / "assets" / "research" / "v4r2_spectral.png"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    arms = [a for a in data["arms"] if a.get("evals", {}).get("best")]
    labels = [a["display"] for a in arms]
    best = [a["evals"]["best"] for a in arms]
    baseline = data["references"]["baseline_v4r1_3600_best500"]["best"]["event_mean_iou"]
    s2 = max(data["references"]["s2_wavelength_map"]["best"]["event_mean_iou"],
             data["references"]["s2_free_map"]["best"]["event_mean_iou"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax = axes[0]
    x = np.arange(len(arms))
    ax.bar(x, [b["event_mean_iou"] for b in best], color="#4c72b0", width=0.6, label="事件平均 IoU")
    ax.scatter(x - 0.15, [b["by_event_iou"]["hiroshima"] for b in best], marker="o", s=26,
               color="#dd8452", label="广岛", zorder=3)
    ax.scatter(x + 0.15, [b["by_event_iou"]["thrissur"] for b in best], marker="s", s=26,
               color="#55a868", label="特里苏尔", zorder=3)
    ax.axhline(baseline, ls="--", lw=1, color="#333", label=f"十波段参照 {baseline:.4f}")
    ax.axhline(s2, ls=":", lw=1, color="#999", label=f"S2 同预算参照 {s2:.4f}")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("IoU（阈值 0.5）")
    ax.set_title("2000 步正式组：事件平均与两事件方向")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    tiny = [(b["tiny_objects"]["matched_evaluable"] or 0, b["tiny_objects"]["evaluable"] or 0)
            for b in best]
    ax.bar(x, [t[0] / t[1] if t[1] else 0 for t in tiny], color="#c44e52", width=0.6)
    for xi, (m, n) in zip(x, tiny):
        ax.text(xi, (m / n if n else 0) + 0.005, f"{m}/{n}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("1–4 像素对象召回")
    ax.set_title("微小对象（DEV 77 个可评价）")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    if GATES.is_file():
        g = json.loads(GATES.read_text(encoding="utf-8"))["results"]
        g = [r for r in g if r.get("gate_mean") is not None]
        gl = [r["arm"].replace("s4_", "").upper() for r in g]
        xx = np.arange(len(g))
        w = 0.26
        ax.bar(xx - w, [r["gate_pos_mean"] for r in g], w, label="GT 滑坡")
        ax.bar(xx, [r["gate_bg_mean"] for r in g], w, label="背景")
        ax.bar(xx + w, [r.get("gate_fp_mean") or 0 for r in g], w, label="FP")
        ax.set_xticks(xx, gl, rotation=20)
        ax.set_ylabel("门控均值")
        ax.set_title("学习门/固定门在目标与背景上的分布")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "gate_diagnostics.json 尚未生成", ha="center", va="center")
        ax.axis("off")

    fig.suptitle("S1 新增通道、S3 波段关系与 S4 单尺度融合：2000 步、单 seed1234、DEV669", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("saved", OUT)


if __name__ == "__main__":
    main()

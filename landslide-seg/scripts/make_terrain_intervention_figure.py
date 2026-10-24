"""只画地形分支无重训干预：同权重下关分支/固定系数/打乱/平均地形的指标变化。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
font = Path("C:/Windows/Fonts/msyh.ttc")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams["axes.unicode_minus"] = False

LABELS = {"normal": "正常（学到的α）", "alpha0": "关闭地形分支", "fixed_alpha_0.01": "固定α=0.01",
          "shuffled_terrain": "打乱地形对应", "mean_terrain": "地形取批均值"}


def main():
    data = json.loads((ROOT / "outputs/research_completion/terrain_interventions.json")
                      .read_text(encoding="utf-8"))
    names = list(data["conditions"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key in zip(axes, ("event_mean_iou", "average_precision")):
        values = ([data["conditions"][n]["event_mean_iou"] for n in names] if key == "event_mean_iou"
                  else [data["conditions"][n]["pixel"]["average_precision"] for n in names])
        bars = ax.bar(range(len(names)), values, color=["#1f77b4"] + ["#a6cee3"] * (len(names) - 1))
        for pos, value in enumerate(values):
            ax.annotate(f"{value:.4f}", (pos, value), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set(xticks=range(len(names)), xticklabels=[LABELS[n] for n in names], ylim=(0, max(values) * 1.12))
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(alpha=.25, axis="y")
    axes[0].set(title="两个验证事件等权平均IoU", ylabel="IoU")
    axes[1].set(title="像素平均精度 AP", ylabel="AP")
    fig.suptitle(f"地形评分迁移分割：同权重非重训干预（学到的α={data['alpha']:.4f}，单seed1234）")
    fig.tight_layout()
    out = ROOT / "docs/assets/research/terrain_interventions.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

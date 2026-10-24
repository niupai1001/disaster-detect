"""只画组成对照（5灾后/3灾后/2灾前+3灾后）的已验收最佳结果；缺失组留空。"""
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


def main():
    data = json.loads((ROOT / "outputs/segformer/v4r3_composition/summary_composition.json")
                      .read_text(encoding="utf-8"))
    rows = data["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for pos, row in enumerate(rows):
        ev = row["evals"]["best"]
        if not row["formal_result"] or ev is None:
            for ax in axes:
                ax.text(pos, .05, "未完成", transform=ax.get_xaxis_transform(),
                        color="#777777", ha="center")
            continue
        tiny = ev["fine_bins"]["1_4"]
        for ax, value in zip(axes, (ev["event_mean_iou"], tiny["object_recall_evaluable"])):
            ax.scatter(pos, value, s=70, color="#1f77b4")
            ax.annotate(f"{value:.4f}", (pos, value), xytext=(0, 10), textcoords="offset points",
                        ha="center")
    axes[0].set(title="两个验证事件等权平均IoU", ylabel="IoU", ylim=(.25, .30))
    axes[1].set(title="1–4像素对象召回（支持数77）", ylabel="对象召回", ylim=(0, .06))
    for ax in axes:
        ax.set(xticks=range(len(rows)), xticklabels=[r["label"] for r in rows],
               xlabel="帧组成（同成员、同2000步）", xlim=(-.5, len(rows) - .5))
        ax.grid(alpha=.25)
    fig.suptitle("SegFormer B0：灾前灾后组成，单seed1234，最佳检查点（回顾性条件）")
    fig.tight_layout()
    out = ROOT / "docs/assets/research/temporal_composition.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

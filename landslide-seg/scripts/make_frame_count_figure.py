"""只画已完成且验收通过的时相数量结果；缺失组留空，不插值。"""
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
    data = json.loads((ROOT / "outputs/segformer/v4r1_temporal_frame_count/summary_frame_count.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for pos, row in enumerate(data["runs"]):
        ev = row["evals"]["best"]
        if not row["formal_result"] or ev is None:
            for ax in axes:
                ax.text(pos, .05, "未完成", transform=ax.get_xaxis_transform(), color="#777777", ha="center")
            continue
        tiny = ev["fine_bins"]["1_4"]
        for ax, value in zip(axes, (ev["event_mean_iou"], tiny["object_recall_evaluable"])):
            ax.scatter(pos, value, s=70, color="#1f77b4")
            ax.annotate(f"{value:.4f}", (pos, value), xytext=(0, 10), textcoords="offset points", ha="center")
    axes[0].set(title="两个验证事件等权平均IoU", ylabel="IoU", ylim=(.22, .35))
    axes[1].set(title="1–4像素对象召回（支持数77）", ylabel="对象召回", ylim=(0, .15))
    for ax in axes:
        ax.set(xticks=range(5), xticklabels=[str(r["n_frames"]) for r in data["runs"]], xlabel="输入观测期数", xlim=(-.5, 4.5))
        ax.grid(alpha=.25)
    fig.suptitle("SegFormer B0＋参考影像细节：单seed1234，最佳检查点")
    fig.tight_layout()
    out = ROOT / "docs/assets/research/temporal_frame_count.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

"""只画 U-TAE 时间对照：固定期数与开关组（30 轮窗口），参考 15 期组另标。"""
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

LABELS = {"frames1": "1期", "frames3": "3期", "frames5": "5期",
          "mean": "15期均值", "content": "15期无日期", "date15": "15期日期(75轮)"}


def main():
    data = json.loads((ROOT / "outputs/research_completion/utae_time_summary.json")
                      .read_text(encoding="utf-8"))
    runs = [r for r in data["runs"] if r.get("label") in LABELS]
    names = [r["label"] for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key in zip(axes, ("event_mean_iou", "average_precision")):
        values = []
        for r in runs:
            ev = (r.get("evals") or {}).get("best")
            values.append(ev[key] if (ev and r.get("formal_result")) else 0.0)
        colors = ["#1f77b4" if r["label"] in ("frames1", "frames3", "frames5", "mean") else "#cccccc"
                  for r in runs]
        ax.bar(range(len(names)), values, color=colors)
        for pos, (value, r) in enumerate(zip(values, runs)):
            ax.annotate(f"{value:.4f}" if value else "训练中", (pos, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set(xticks=range(len(names)), xticklabels=[LABELS[n] for n in names],
               ylim=(0, max(values) * 1.15))
        ax.tick_params(axis="x", labelrotation=18)
        ax.grid(alpha=.25, axis="y")
    axes[0].set(title="两个验证事件等权平均IoU", ylabel="IoU")
    axes[1].set(title="像素平均精度 AP", ylabel="AP")
    fig.suptitle("U-TAE 时间对照：固定期数与开关组（1/3/5 与均值组为 30 轮；灰色为 75 轮参考）")
    fig.tight_layout()
    out = ROOT / "docs/assets/research/utae_time_controls.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

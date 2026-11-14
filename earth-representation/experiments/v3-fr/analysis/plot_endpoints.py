"""Compare S6 validation curves and the legacy contribution from measured endpoints."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(sys.argv[1])
out = root / "summary"
figures = Path(sys.argv[2]) if len(sys.argv)>2 else out/"figures"
names = json.loads((root/"experiment_display_names.json").read_bytes())
with (out / "endpoint_metrics_and_cost.csv").open(encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))
arms = [r["arm"] for r in rows]
assert arms == ["B0", "BP", "N0", "N1", "N2", "C"]
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
colors = ["#184c72", "#476c84", "#248276", "#4a6d3d", "#936826", "#a5422d"]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.9), layout="constrained")
for ax, task in zip(axes[0], ("A", "B1"), strict=True):
    for arm, color in zip(arms, colors, strict=True):
        path = root / "s6_component_trials" / arm / "status.json"
        nodes = json.loads(path.read_bytes())["validation"]
        assert [node["step"] for node in nodes] == [0, 42, 116, 233]
        ax.plot([n["step"] for n in nodes], [n["groups"][f"{task}/all"]["mse"] for n in nodes],
                label=names[arm]["name"], color=color, marker="o", linewidth=2 if arm in ("B0", "C") else 1.15,
                markersize=4, linestyle="--" if arm=="C" else "-")
    title = "B1 完整验证集 · 总输出" if task=="B1" else "A 完整验证集"
    ax.set(title=title, xlabel="优化器更新步", ylabel="原生像元 MSE", xticks=[0, 42, 116, 233])
    ax.grid(axis="y", alpha=.25)
axes[0,1].legend(frameon=False, fontsize=8)

x = np.arange(len(rows))
ax = axes[1,0]
for key, label, width, color in (("B1_main_texture","共享主路径",-.37,"#184c72"),
                                 ("B1_texture","加上历史细节支路",.37,"#bf7244")):
    ax.barh(x+width/2, [float(r[key]) for r in rows], .34, color=color, label=label)
ax.set(title="B1 纹理误差 · 第 233 步", xlabel="高纹理梯度 MSE",
       yticks=x, yticklabels=[names[arm]["name"] for arm in arms], xlim=(0,.16), ylim=(5.55,-1.35))
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="x", alpha=.25)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)

ax = axes[1,1]
screen = json.loads((root/"s6_component_trials/screen.json").read_bytes())["effects"]["C"]["N2"]
for j, (task, color) in enumerate(zip(("A","B1"), ("#184c72","#a5422d"), strict=True)):
    for i,sensor in enumerate(("all","S2","L8")):
        metric = screen[f"paired"][f"{task}/{sensor}"]["metrics"]["mse"]
        low,high = metric["bootstrap_95"]
        value = metric["mean"]
        y = 2-i + (.12 if j==0 else -.12)
        ax.errorbar(value, y, xerr=[[value-low],[high-value]], fmt="o" if j==0 else "s", color=color,
                    capsize=3, label=task if i==0 else None)
ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
ax.set(title="FiLM 浅层 ResNet · D1 子像素减 D0 连续插值", xlabel="配对 MSE 差（右侧表示退化）",
       yticks=[0,1,2], yticklabels=["L8","S2","全部"], ylim=(-.45,2.45))
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", alpha=.25)
fig.suptitle("V3-FR · seed 17 · 383 条验证观测 · 每组 233 次更新", fontsize=17, weight="bold")
fig.savefig(figures/"s6_overview.png", dpi=175)
plt.close(fig)

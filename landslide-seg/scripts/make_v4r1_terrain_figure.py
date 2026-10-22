"""Render saved terrain results using defined labels and best-checkpoint alpha."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs/assets/research"
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
LABELS=['当前参照：\n十波段影像＋细节分支', '12通道共同输入：\n影像＋高程＋坡度', '独立地形分支：\n直接随分割训练', '独立地形分支：\n先做地形自监督', '独立地形分支：\n加载标注训练的评分头', '独立地形分支：\n地形自监督预算加倍', '独立地形分支：\n固定融合系数']
def main():
    rows=json.loads((ROOT/"outputs/segformer/s2_10band_v4r1_geo_raw/summary_terrain.json").read_text(encoding="utf-8"))
    base=rows[0]["best_val"]
    fig,axes=plt.subplots(1,2,figsize=(18,6.5),gridspec_kw={"width_ratios":[1.3,1]})
    ax=axes[0];ys=np.arange(len(rows));vals=[r["best_val"] for r in rows]
    ax.barh(ys,vals,color=["#7c8796"]+["#42876a" if v>base else "#bb6a63" for v in vals[1:]])
    ax.axvline(base,color="#454545",ls="--",lw=1)
    for i,v in enumerate(vals):ax.text(v+.0004,i,f"{v:.4f} ({v-base:+.4f})",va="center",fontsize=9)
    ax.set_yticks(ys,LABELS);ax.invert_yaxis();ax.set_xlim(.285,.319)
    ax.set_xlabel("验证事件平均 IoU（广岛与特里苏尔各占一半）")
    ax.set_title("我们的数据划分（留事件验证）：当前参照与地形对照",fontsize=13)
    ax.grid(axis="x",alpha=.2)
    ax=axes[1];items=[(label,row) for label,row in zip(LABELS,rows) if row.get("alpha") is not None]
    ax.barh(range(len(items)),[row["alpha"] for _,row in items],color="#4f78a0")
    ax.set_yticks(range(len(items)),[label for label,_ in items]);ax.invert_yaxis();ax.set_xlim(0,.30)
    for i,(_,row) in enumerate(items):ax.text(row["alpha"]+.004,i,f"{row['alpha']:.3f}",va="center",fontsize=10)
    ax.set_xlabel("地形融合系数α（可学习组初始0.01；固定组保持0.01）")
    ax.set_title("验证选出的最佳模型中的系数，不是训练末态",fontsize=12)
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.text(.03,.025,"左：分数与相对基线差值。右：实际地形修正为 α × tanh(地形分数)，仅凭系数不能判断地形贡献。",fontsize=11)
    for ext in ["png","svg"]:fig.savefig(OUT/f"v4r1_terrain_arms.{ext}",dpi=170,bbox_inches="tight")
    plt.close(fig);print(OUT/"v4r1_terrain_arms.png")
if __name__=="__main__":main()

"""Render the documented terrain connections and the three training targets (CPU only)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/research"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
BLUE, GREEN, GOLD = "#dfebf7", "#e4f1df", "#fff0d4"

def box(ax, x, y, w, h, text, color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025",fc=color,ec="#536273"))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,lw=1.3,color="#536273"))

def save(fig, name):
    fig.savefig(OUT / (name+".png"),dpi=170,bbox_inches="tight")
    fig.savefig(OUT / (name+".svg"),bbox_inches="tight")
    plt.close(fig)

def main():
    fig,ax=plt.subplots(figsize=(17,7.4));ax.set(xlim=(0,16),ylim=(0,7));ax.axis("off")
    ax.set_title("同一张滑坡掩膜如何由影像、原始网格细节与地形共同生成",fontsize=17,pad=14)
    for args in [(.2,4.75,2.0,.9,"单张十波段影像",BLUE),(3,5.7,2.4,.9,"MiT分层编码器",BLUE),(6,5.7,2.5,.9,"SegFormer原解码器",BLUE),(9.3,5.7,2,.9,"基础分割分数",BLUE),(3,4.0,2.4,.9,"不下采样\n局部细节卷积",BLUE),(6,4.0,2.5,.9,"解码特征上采样\n与细节特征拼接",BLUE),(9.3,4.0,2,.9,"细节修正分数",BLUE),(.2,1.8,2,.9,"高程与坡度",GREEN),(3,1.8,2.4,.9,"ResNet18初始层\n与第一组残差块",GREEN),(6,1.8,2.5,.9,"卷积分数预测头\n再上采样",GREEN),(9.3,1.8,2,.9,"tanh限幅后\n乘地形系数α",GREEN),(12.2,4.0,1.2,1.0,"三项分数\n相加",GOLD),(14,4.0,1.7,1.0,"sigmoid\n滑坡概率",GOLD)]:box(ax,*args)
    for a,b in [((2.2,5.2),(3,6.15)),((5.4,6.15),(6,6.15)),((8.5,6.15),(9.3,6.15)),((2.2,5.0),(3,4.45)),((5.4,4.45),(6,4.45)),((7.25,5.7),(7.25,4.9)),((8.5,4.45),(9.3,4.45)),((11.3,6.15),(12.2,4.9)),((11.3,4.45),(12.2,4.45)),((2.2,2.25),(3,2.25)),((5.4,2.25),(6,2.25)),((8.5,2.25),(9.3,2.25)),((11.3,2.25),(12.2,4.1)),((13.4,4.5),(14,4.5))]:arrow(ax,a,b)
    ax.text(.2,1.02,"蓝色两路保留影像基线结构并共同训练；绿色支路只读地形，最后修改分割分数。",fontsize=11)
    ax.text(.2,.48,"地形修正不是独立易发性概率。加载标签比例预训练头的组不满足初始地形修正为零。",fontsize=11)
    fig.tight_layout();save(fig,"terrain_fusion_arch")
    fig,ax=plt.subplots(figsize=(17,7.5));ax.set(xlim=(0,16),ylim=(0,7.4));ax.axis("off")
    ax.set_title("地形训练：网络输入、预测内容和监督目标来自哪里",fontsize=17,pad=12)
    lanes=[(5.3,"地形自监督\n先学地形本身","高程与坡度\n同位置遮挡约一半","地形编码器\n＋临时重建解码器","预测被遮挡的高程","目标：原归一化高程\n损失：遮挡位置绝对误差"),(3.1,"用滑坡标注训练评分\n复用已有标注","未遮挡的\n高程与坡度","地形编码器\n＋卷积分数头","预测局部滑坡比例","目标：人工掩膜区块均值\n损失：二元交叉熵"),(.9,"最终像素分割\n人工标注监督","十波段影像\n＋地形输入","影像基线与地形分支\n按分数融合","预测最终滑坡分割","目标：原逐像素人工掩膜\n损失：BCE＋Dice")]
    for y,title,inp,net,pred,target in lanes:
        ax.text(.15,y+.43,title,ha="left",va="center",fontsize=11,weight="bold")
        for x,w,text,col in [(3.0,2.2,inp,BLUE),(5.8,2.5,net,GREEN),(8.9,2.2,pred,BLUE),(11.8,4.0,target,GOLD)]:box(ax,x,y,w,.95,text,col)
        for a,b in [((5.2,y+.48),(5.8,y+.48)),((8.3,y+.48),(8.9,y+.48)),((11.1,y+.48),(11.8,y+.48))]:arrow(ax,a,b)
    ax.text(.15,.18,"三行只使用训练事件；验证事件不参与预训练。比例标签来自相同训练标注，不是新增历史易发性调查。",fontsize=10.5)
    fig.tight_layout();save(fig,"terrain_training_tasks")
    print("Rendered terrain_fusion_arch and terrain_training_tasks")
if __name__=="__main__":main()

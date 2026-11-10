"""Render fixed-panel baseline evidence; no data fitting or model updates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
OUT=ROOT/'outputs/encoder_v3/hierarchical/reconstruction_research'
report=json.loads((OUT/'native_baselines_v1/report.json').read_text())
assert report['status']=='PASS' and report['observations']==383 and report['optimizer_updates']==0
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
keys=['model_mse','visible_mean_mse','nearest_mse','idw8_mse','local_affine8_mse']
labels=['原模型232步','可见均值','最近邻','距离加权8邻点','局部平面外推']
colors=['#ce514c','#8d9aac','#507bb7','#168375','#d59b39']
fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
values=[report['means'][k] for k in keys]
axes[0].bar(range(5),values,color=colors)
axes[0].set_xticks(range(5),labels,rotation=25,ha='right');axes[0].set_ylim(0,max(values)*1.2)
for i,v in enumerate(values):axes[0].text(i,v+.015,f'{v:.4f}',ha='center')
axes[0].set_title('完整383 VAL：同一A遮盖与QA');axes[0].set_ylabel('原生标准化MSE / z²（越低越好）')
for j,sensor in enumerate(('S2','L8')):
    axes[1].bar(np.arange(4)+(j-.5)*.36,[report['by_source'][sensor][k] for k in keys[:4]],width=.36,label=f'{sensor} / {report["by_source"][sensor]["observations"]}观测')
axes[1].set_xticks(range(4),labels[:4],rotation=25,ha='right');axes[1].set_title('分来源均值（非波段数加权）');axes[1].legend()
x=np.array([m['idw8_mse'] for m in report['members']]);y=np.array([m['model_mse'] for m in report['members']])
assert np.all(x>0) and np.all(y>0)
axes[2].scatter(x,y,c=['#507bb7' if m['sensor']=='S2' else '#d59b39' for m in report['members']],s=11,alpha=.6)
lo=min(x.min(),y.min())*.8;hi=max(x.max(),y.max())*1.2
axes[2].plot([lo,hi],[lo,hi],'--',color='gray');axes[2].set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),xlabel='距离加权插值 MSE',ylabel='原模型 MSE')
axes[2].set_title(f'{int((y>x).sum())}/383观测：原模型误差更大（线上方）')
for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('重建诊断：无需训练的可见信号插值已优于当前模型\n仅排除EnMAP的一轮预热检查点；不能据此宣称Token丢失全部信息或已充分收敛',fontsize=13)
fig.savefig(OUT/'baseline_summary.png',dpi=150);plt.close(fig)
print('RECONSTRUCTION_SUMMARY_PASS',flush=True)

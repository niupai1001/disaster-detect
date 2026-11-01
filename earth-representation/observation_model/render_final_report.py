"""Populate the single research entry from completed evaluation artifacts, never plans."""
import csv
import argparse
import json
import os
import time
from pathlib import Path

from .data import ROOT,RELEASE
from .paired import CROSS_RELEASE
from .render_training import NAMES,main as training_plot

OUT=Path(os.environ.get("OBSERVATION_OUTPUT_ROOT",ROOT/"observation_model/outputs"))
REFERENCE_OUT=ROOT/"observation_model/outputs"


def load(path):return json.loads(path.read_text())
def number(value):return "缺项" if value is None else f"{value:.4f}"


def main():
    folder=OUT/"final";folder.mkdir(exist_ok=True)
    required=[folder/"selection.json",folder/"evaluation_index.json",folder/"probe_index.json"]
    missing=[str(p) for p in required if not p.exists()]
    status=OUT/"final_evaluation_status.json"
    if not status.exists() or load(status)["status"]!="evaluation_and_probes_complete_pending_report_and_final_audit":missing.append("final evaluation/probe completion")
    if missing:
        (folder/"report_readiness.json").write_text(json.dumps(dict(status="incomplete",missing=missing),indent=2))
        print(json.dumps(dict(status="incomplete",missing=missing)));return
    selected=load(required[0]);index=load(required[1]);probes=load(required[2]);training_plot()
    cost=load(folder/"training_costs_live.json")["runs"]
    def evaluation(name,mode,split):
        row=next(r for r in index if r["name"]==name and r["mode"]==mode and r["split"]==split and r["pair_set"]=="strict" and r["route"]=="direct")
        return load(Path(row["result"]))
    def score(name,mode,split):return evaluation(name,mode,split)["scores"]["singleton" if mode=="masked" else "cross30"]["score"]
    def paired(name,mode):return number(score(name,mode,"val"))+" / "+number(score(name,mode,"test"))
    membership=load(RELEASE/"release.json")["observations"];pairs=load(CROSS_RELEASE/"release.json")
    observation_counts=[sum(n for k,n in membership.items() if k.startswith(s+'/')) for s in ['train','val','test']]
    pair_counts='/'.join(str(pairs['pairs'][s]) for s in ['train','val','test']);acquisition_counts='/'.join(str(pairs['independent_acquisition_pairs'][s]) for s in ['train','val','test'])
    lines=["## 完整研究结果", "", "### 阅读约定与实验设置", "",
           "本轮研究同一卫星的整波段恢复，以及S2和Landsat之间的真实双向观测预测。S2输入保留10/20 m，Landsat输入为30 m；跨源主目标统一到真实Landsat的30 m网格。10 m的Landsat波段输出只是模型估计，其退化结果才能与真实30 m观测比较。", "",
           f"已核对6309个本地共同位置和94635条S2时相；它们不是6309组已准入原生配对。当前冻结单源训练{observation_counts[0]}、验证{observation_counts[1]}、测试{observation_counts[2]}份原生观测。严格跨源训练/验证/测试为{pair_counts}个位置，对应{acquisition_counts}组独立采集，覆盖有限。下载保持暂停。", "",
           "新模型自监督seed=0，各组独立调度和早停。Phase2随机初始化；Phase3两组继承固定同源Transformer第10000步权重；Phase4若执行，两组继承同一通过验证门槛的跨源权重。轻量任务头随机初始化，seed=1234。表中归一化RMSE先将各波段RMSE除以仅由训练集确定的标准差，再按传感器或预测方向等权汇总。例如RMSE=0.01、训练标准差=0.05时，该波段归一化RMSE为0.2。所有方法使用同一真实目标QA；预测覆盖不足仍计入误差。", "",
           "### 主表", "", "表1：无标签验证选择的最佳权重。每格依次为Val / Test，越低越好；两任务使用不同目标网格，不能跨列当作同一误差。", "",
           "| 模型与训练目标 | 本阶段训练更新 | 最佳权重步数 | 同源恢复：归一化RMSE | 双向跨源30 m：归一化RMSE |", "|---|---:|---:|---:|---:|"]
    for kind,label in [("mean","训练均值"),("interpolation","邻近波段插值"),("ridge","Ridge光谱回归"),("mlp","小型条件MLP")]:
        lines.append(f"| {label} | {'10000' if kind=='mlp' else '解析/无迭代'} | 各任务独立选择 | {paired('simple_'+kind,'masked')} | {paired('simple_'+kind,'cross')} |")
    for run,result in selected["runs"].items():
        point=next(p for p in selected["points"] if p["run"]==run and p["role"]=="best")
        lines.append(f"| {NAMES[run]} | {result['step']} | {point['step']} | {paired(run+'_best','masked')} | {paired(run+'_best','cross')} |")
    lines += ["", "表2：末态、共同里程碑与已记录成本。末态/共同里程碑两数均来自Test；步数是本阶段计数，祖先预训练权重另见配置。独立最佳不是严格等预算消融。", "",
              "| 模型与训练目标 | 共同步数 | 同源Test：末态 / 共同 | 跨源Test：末态 / 共同 | 有日志更新（其中重复） | 活跃时长/小时 |", "|---|---:|---:|---:|---:|---:|"]
    for run in selected["runs"]:
        point=next(p for p in selected["points"] if p["run"]==run and p["role"]=="common");c=cost[run]
        values=[number(score(run+'_last',mode,'test'))+' / '+number(score(run+'_common',mode,'test')) for mode in ['masked','cross']]
        lines.append(f"| {NAMES[run]} | {point['step']} | {values[0]} | {values[1]} | {c['logged_optimizer_updates']}（{c['logged_repeated_updates']}） | {c['logged_active_wall_seconds']/3600:.2f} |")
    probe_results={}
    for row in probes:probe_results.setdefault(row["run"],{})[(row["dataset"],row["task"])]=load(Path(row["result"]))
    for name in ["legacy_cnn_dino","dofa","anysat"]:
        probe_results[name]={}
        for dataset,task in [("worldcover","classification"),("worldcover","segmentation"),("landslide","segmentation")]:
            probe_results[name][(dataset,task)]=load(REFERENCE_OUT/"probes"/dataset/name/task/"result.json")
    aliases=dict(legacy_cnn_dino="历史共享单波段CNN-DINO",dofa="官方完整DOFA",anysat="官方完整AnySat")
    lines += ["", "表3：冻结表征＋统一线性任务头。每格依次为最佳头/末态头；编码器由无标签验证固定，未用标签反选预训练权重。WorldCover两列来自同一地图来源，不是两个独立真值任务。", "",
              "| 冻结编码器 | WorldCover分类：Test平均AP | WorldCover分割：Test IoU | 滑坡辅助：研发验证事件平均IoU |", "|---|---:|---:|---:|"]
    for name,results in probe_results.items():
        cells=[]
        for key in [("worldcover","classification"),("worldcover","segmentation"),("landslide","segmentation")]:
            r=results[key];cells.append(number(r.get("best_score"))+" / "+number(r.get("last_score")))
        lines.append("| "+" | ".join([NAMES.get(name,aliases.get(name,name)),*cells])+" |")
    import matplotlib.pyplot as plt
    names=list(probe_results);fig,axes=plt.subplots(1,3,sharey=True,figsize=(17,max(5,len(names)*.55)),constrained_layout=True)
    tasks=[(('worldcover','classification'),'WorldCover分类：Test平均AP'),(('worldcover','segmentation'),'WorldCover分割：Test IoU'),(('landslide','segmentation'),'滑坡辅助：研发验证事件平均IoU')]
    for ax,(key,title) in zip(axes,tasks):
        values=[probe_results[name][key].get('best_score') for name in names]
        ax.barh(range(len(names)),[float('nan') if v is None else v for v in values],color=['#777777' if name=='masked_random' else '#4477AA' for name in names])
        for i,value in enumerate(values):ax.annotate(number(value),(0 if value is None else value,i),xytext=(3,0),textcoords='offset points',va='center',fontsize=9)
        ax.set_title(title);ax.set_xlim(0,max([v for v in values if v is not None]+[.01])*1.2);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(names)),[NAMES.get(name,aliases.get(name,name)) for name in names]);axes[0].invert_yaxis()
    fig.savefig(folder/'figures/frozen_probes.png',dpi=160);plt.close(fig)
    lines += ["", "### 图与解释", "", "![训练曲线](outputs/final/figures/training_curves.png)", "",
              "每行是一类任务，左列是自身目标的训练损失，右列是统一的无标签验证评分。浅色为较早会话，深色为最新恢复会话。损失项不同的曲线不能直接按高低排名。", "",
              "![真实双向跨源预测固定案例](outputs/final/figures/cross_cross30_fixed_cross_test.png)", "",
              "每行一个预测方向，从左到右是可见输入波段、真实目标、模型估计、绝对误差及真实目标有效区。各图保留自己的像元网格；跨源目标是共同30 m。首个案例预先固定，不按预测质量筛选；显示范围不改变数值评价。", "",
              "![冻结表征线性读出](outputs/final/figures/frozen_probes.png)", "",
              "每行一个预先固定的编码器；三列分别是地图分类、地图分割和滑坡辅助任务。柱为该编码器经任务头验证选模后的分数，不按柱高重新选择预训练权重。分类仅计训练集非恒定且正负例均至少3个的类别，完整类别及阳性数见结果文件；地图两列仍来自同一来源。", "",
              "### 展开结论", ""]
    for mode,title in [("masked","同源整波段恢复"),("cross","真实双向跨源预测")]:
        candidates={run:score(run+'_best',mode,'test') for run in selected['runs']};winner=min(candidates,key=candidates.get)
        baseline=min((score('simple_'+kind,mode,'test'),kind) for kind in ['mean','interpolation','ridge','mlp'])
        lines += [f"{title}：预先固定的神经网络候选中，{NAMES[winner]}的Test归一化RMSE最低，为{candidates[winner]:.4f}；合理简单参照的最低值为{baseline[0]:.4f}（{baseline[1]}）。这只是单种子、当前集合上的描述性比较；最低Test分数不反过来选择预训练权重，也不证明训练机制的单因素因果。", ""]
    lines += ["空间约束阶段的实际决策为：`"+selected['physics_gate']['status']+"`。决策依据是验证集上的双向门槛及采集组Bootstrap，不能把工程短试跑当作通过。完整理由见`outputs/phase4/gate.json`。", "",
              "### 限制与下一步", "",
              "独立人工土地覆盖真值仍缺失，通用语义结论保持未验证。WorldCover只说明20 m地图一致性；滑坡只作辅助，其研发验证继续用于任务头选模。所有任务使用中心1200 m窗口，不能直接混用历史1280 m结果。线性头输入维数和参数量因编码器而异，失败也不能单独归因于编码器。", "",
              "历史CNN-DINO曾用36幅广岛影像做无标签验证选模，不能称作未接触广岛的参照；DOFA和AnySat的预训练地域覆盖未知，规模和初始化不同。单种子不说明显著性，严格跨源验证只有2组独立采集。", "",
              "系统重启造成回滚；日志无法解析的片段和未记录的启动成本单列缺项，日志成本是可追溯下界，活跃时长不是独占GPU计算时间。停止于训练上限的组不自动称为已收敛。", "",
              "### 附录与复现入口", "", "- 配置、成员、来源、统计、初始化、停止原因：`outputs/runs/*/resolved_config.json`及`training_result.json`。",
              "- 冻结选择、最佳/末态/共同里程碑、导出哈希：`outputs/final/selection.json`。完整可恢复断点留在原运行目录；推理导出不替代训练断点。",
              "- 逐波段MAE、RMSE、偏差及红边/SWIR分组：`outputs/final/per_band_metrics.csv`、`band_group_metrics.csv`及`evaluation_index.json`。",
              "- 冻结探针、逐事件/对象/边界指标、参数量与成本：`outputs/final/probe_index.json`及其结果文件。",
              "- 同期HLS正负控制：`outputs/phase0/hls_same_acquisition_controls.json`；HLS协调产品与原始L2反射率分开，不冒充独立仪器真值。",
              "- [Training Viewer](http://127.0.0.1:8766/)；协议验收与完整交付审计仍须单独核对。", ""]
    rows=[]
    for entry in index:
        for band in load(Path(entry['result']))['bands']:rows.append({**{k:entry[k] for k in ['name','mode','split','pair_set','route']},**band})
    with (folder/'per_band_metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    groups=[]
    for entry in index:
        bands=load(Path(entry['result']))['bands']
        for label,names in [('红边及窄近红外',{'S2:B05','S2:B06','S2:B07','S2:B8A'}),('SWIR',{'S2:B11','S2:B12','L8:SR_B6','L8:SR_B7'})]:
            members=[b for b in bands if b['band'] in names]
            if members:groups.append(dict(name=entry['name'],mode=entry['mode'],split=entry['split'],pair_set=entry['pair_set'],route=entry['route'],band_group=label,mean_band_rmse=sum(b['rmse'] for b in members)/len(members)))
    with (folder/'band_group_metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    path=ROOT/'observation_model/README.md';text=path.read_text();start='<!-- FINAL_RESULTS_START -->';end='<!-- FINAL_RESULTS_END -->'
    if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
    rendered='\n'.join(lines).replace('outputs/final/',OUT.relative_to(ROOT/'observation_model').as_posix()+'/final/')
    rendered=rendered.replace('outputs/runs/',OUT.relative_to(ROOT/'observation_model').as_posix()+'/runs/')
    if (OUT/'budget.json').exists():
        budget=load(OUT/'budget.json')
        rendered+='\n\n本节成本表统计修正版运行；旧版另有已记录'+str(budget['old_logged_updates'])+'次更新，含'+str(budget['old_logged_replay_updates'])+'次重复计算，仍计入整体成本。分配及未记录成本预留见`'+(OUT/'budget.json').relative_to(ROOT/'observation_model').as_posix()+'`。达到修正版预算上限不等于已确认收敛。\n'
    path.write_text(text.rstrip()+'\n\n'+start+'\n'+rendered+'\n'+end+'\n')
    (folder/'report_readiness.json').write_text(json.dumps(dict(status='report_generated_pending_final_delivery_audit',missing=[]),indent=2));print('Stable README result section generated from actual artifacts')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:
        print('Waiting for completed frozen-checkpoint evaluation and probes; no final report claimed',flush=True)
        while True:
            path=OUT/'final_evaluation_status.json'
            try:ready=path.exists() and load(path)['status']=='evaluation_and_probes_complete_pending_report_and_final_audit'
            except json.JSONDecodeError:ready=False
            if ready:break
            time.sleep(30)
    main()

"""Render actual completed USE evidence; incomplete groups never receive scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets/research/use'
LABELS={'baseline':'十波段直入＋原网格细节','mean':'USE共享编码＋均值汇聚',
 'self_mean':'USE波段自注意力＋均值汇聚','self_query':'USE自注意力＋查询汇聚',
 'physical':'USE注意力＋物理描述','random':'USE注意力＋固定随机描述',
 'capacity':'USE扩宽无注意力编码','mean_mixed':'无注意力USE＋混合波段训练',
 'physical_mixed':'物理USE＋混合波段训练','pretrained_mixed':'物理USE＋外部预训练后微调'}


def main():
    OUT.mkdir(exist_ok=True)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    records=[]
    for name,label in LABELS.items():
        directory=ROOT/f'outputs/use/s2_10band_v4r1_p20/use_{name}/1234'
        path=directory/'train_summary.json'
        if not path.exists():continue
        training=json.loads(path.read_text())
        if not training.get('complete') or training['global_step']!=2000:continue
        row=dict(id=name,label=label,training=training,evaluations={})
        for subset in ('full','rgb','rgb_nir','six','unseen5'):
            p=directory/f'eval/eval_dev_{subset}.json'
            if p.exists():row['evaluations'][subset]=json.loads(p.read_text())
        records.append(row)
    import hashlib
    sources=[ROOT/f'outputs/use/s2_10band_v4r1_p20/use_{r["id"]}/1234/train_summary.json' for r in records]
    for r in records:
        sources.extend(sorted((ROOT/f'outputs/use/s2_10band_v4r1_p20/use_{r["id"]}/1234/eval').glob('eval_dev_*.json')))
    sources.append(Path(__file__))
    if (ROOT/'outputs/use/costs.json').exists():sources.append(ROOT/'outputs/use/costs.json')
    (OUT/'figure_provenance.json').write_text(json.dumps(dict(command='python -X utf8 scripts/summarize_use.py',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=1),encoding='utf-8')
    if not records:return
    figure,axes=plt.subplots(1,2,figsize=(13,7),gridspec_kw={'width_ratios':[1.5,1]})
    y=list(range(len(records)))
    axes[0].barh(y,[r['training']['best_val_iou'] for r in records],color='#3976a8')
    axes[0].set_yticks(y,[r['label'] for r in records],fontproperties=font,fontsize=10)
    axes[0].invert_yaxis();axes[0].set_xlabel('DEV两事件等权IoU',fontproperties=font)
    axes[0].set_xlim(0,max(.35,max(r['training']['best_val_iou'] for r in records)*1.2))
    for i,r in enumerate(records):axes[0].text(r['training']['best_val_iou']+.003,i,f"{r['training']['best_val_iou']:.4f}",va='center',fontsize=9)
    tiny=[r['evaluations'].get('full',{}).get('overall',{}).get('objects',{}).get('use_bins',{}).get('1_4',{}).get('object_recall_evaluable',float('nan')) for r in records]
    axes[1].barh(y,tiny,color='#bd7648');axes[1].invert_yaxis();axes[1].set_yticks(y,['']*len(y))
    axes[1].set_xlabel('1–4像素对象召回（77个）',fontproperties=font);axes[1].set_xlim(0,.2)
    for i,v in enumerate(tiny):axes[1].text(v+.002,i,f'{v:.4f}',va='center',fontsize=9)
    for ax in axes:ax.grid(axis='x',alpha=.2)
    figure.suptitle(f'USE首轮：已完成{len(records)}/10组监督训练；单seed开发证据',fontproperties=font,fontsize=15)
    figure.tight_layout(rect=[0,0,1,.95]);figure.savefig(OUT/'results.png',dpi=150);plt.close(figure)
    figure,axes=plt.subplots(4,3,figsize=(14,12),sharex=True,sharey=True)
    for ax,row in zip(axes.flat,records):
        history=row['training']['history'];ax.plot([r['global_step'] for r in history],[r['dev']['event_mean_iou'] for r in history])
        ax.axhline(records[0]['training']['best_val_iou'],ls='--',color='gray',lw=.8)
        ax.set_title(row['label'],fontproperties=font,fontsize=10);ax.set_ylim(0,.35);ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(records):]:ax.set_visible(False)
    figure.supxlabel('成功更新次数；虚线为本轮十波段直入的最佳IoU',fontproperties=font)
    figure.supylabel('DEV两事件等权IoU',fontproperties=font);figure.tight_layout();figure.savefig(OUT/'training.png',dpi=150);plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(13,5))
    for row in records:
        if row['id'] not in ('mean','mean_mixed','physical','physical_mixed','pretrained_mixed'):continue
        order=[('rgb',3),('rgb_nir',4),('unseen5',5),('six',6),('full',10)]
        views=row['evaluations'];points=[(c,views[s]['event_mean_iou']) for s,c in order if s in views]
        if len(points)>1:axes[0].plot(*zip(*points),marker='o',label=row['label'])
        full=views.get('full')
        if full:axes[1].plot([1,2,3],[full['overall']['objects']['use_bins'][k]['object_recall_evaluable'] for k in ('1_4','5_25','26_plus')],marker='o',label=row['label'])
    axes[0].set_xticks([3,4,5,6,10]);axes[0].set_xlabel('波段数：5为未见组合，身份亦不同',fontproperties=font)
    axes[0].set_ylabel('固定十波段最佳权重的事件平均IoU',fontproperties=font)
    axes[1].set_xticks([1,2,3],['1–4 (n=77)','5–25 (n=418)','26+ (n=1184)'])
    axes[1].set_xlabel('对象面积（像素），全十波段评价',fontproperties=font);axes[1].set_ylabel('一对一匹配对象召回',fontproperties=font)
    for ax in axes:ax.grid(alpha=.2);ax.legend(prop=font,fontsize=8)
    figure.tight_layout();figure.savefig(OUT/'bands_sizes.png',dpi=150);plt.close(figure)
    import sys
    if '--cases' in sys.argv:failure_cases(records)
    write_documents(records)


def failure_cases(records):
    """Freeze cases from GT/event/size and seed, before looking at predictions."""
    import sys
    import hashlib
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import torch
    from landslide_seg.config import load_segformer_config
    from landslide_seg.data.segformer_dataset import read_manifest,load_stats,S2SegDataset,collate
    from landslide_seg.data.use_dataset import USESegDataset
    from landslide_seg.training.segformer_trainer import load_checkpoint
    from landslide_seg.training.segformer_metrics import label_objects,object_table
    torch.set_num_threads(4)
    cfg=load_segformer_config(ROOT/'configs/segformer/use/physical_mixed.yaml')['segformer']
    rows=[r for r in read_manifest(ROOT/cfg['data']['manifest']) if r['split']=='val']
    members={r['sample_id']:r for r in rows};candidates={}
    for r in rows:
        with np.load(r['cache']) as z:
            labels,n=label_objects(z['mask']);objects=object_table(labels,n,z['valid'])
        for obj in objects:
            if obj['truncated']:continue
            area=obj['area'];size='1_4' if area<=4 else '5_25' if area<=25 else '26_plus'
            candidate=dict(sample_id=r['sample_id'],event=r['event_id'],size=size,**obj)
            identity=f"1234:{r['sample_id']}:{obj['id']}";candidate['hash']=hashlib.sha256(identity.encode()).hexdigest()
            candidates.setdefault((r['event_id'],size),[]).append(candidate)
    cases=[]
    for event,size in [('hiroshima','1_4'),('hiroshima','5_25'),('hiroshima','26_plus'),('thrissur','5_25'),('thrissur','26_plus')]:
        cases.extend(sorted(candidates[(event,size)],key=lambda c:c['hash'])[:2 if size=='1_4' else 1])
    picked=[members[c['sample_id']] for c in cases];stats=load_stats(ROOT/cfg['data']['stats'])
    names=[n for n in ('baseline','physical_mixed','pretrained_mixed') if any(r['id']==n for r in records)]
    predictions={};sources={}
    for name in names:
        path=ROOT/f'outputs/use/s2_10band_v4r1_p20/use_{name}/1234/best.pt'
        sources[name]=hashlib.sha256(path.read_bytes()).hexdigest();model,_=load_checkpoint(path,device='cpu');model.eval()
        dataset=(S2SegDataset if name=='baseline' else USESegDataset)(picked,stats,augment=False,preload=True)
        with torch.inference_mode():predictions[name]=torch.sigmoid(model(collate([dataset[i] for i in range(len(dataset))])['image'])['logits'][:,0]).numpy()
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    figure,axes=plt.subplots(len(cases),2+len(names),figsize=(3*(2+len(names)),3*len(cases)),squeeze=False)
    for i,c in enumerate(cases):
        r=picked[i]
        with np.load(r['cache']) as z:
            rgb=np.clip(z['x10'][[2,1,0]].transpose(1,2,0)/3000,0,1)**.7;gt=z['mask'];valid=z['valid']
        y0,x0,y1,x1=c['bbox'];cy=(y0+y1)//2;cx=(x0+x1)//2
        ya=max(0,cy-24);xa=max(0,cx-24);yb=min(128,ya+48);xb=min(128,xa+48)
        panels=[(rgb,'RGB固定显示缩放'),(gt,'原标注（红框为所选对象）')]+[(predictions[n][i],LABELS[n]+' 概率') for n in names]
        for j,(panel,title) in enumerate(panels):
            ax=axes[i,j];image=panel[ya:yb,xa:xb];ax.imshow(image,cmap='gray' if j==1 else 'viridis',vmin=0,vmax=1,interpolation='nearest')
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((x0-xa-.5,y0-ya-.5),x1-x0,y1-y0,fill=False,edgecolor='red',lw=1))
            if j>1:ax.contour(valid[ya:yb,xa:xb]&(image>=.5),levels=[.5],colors='white',linewidths=.5)
            ax.set_title(title,fontproperties=font,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_ylabel(f"{c['event']}\n{c['area']}像素\n{c['sample_id']}",fontproperties=font,fontsize=7)
    figure.tight_layout();figure.savefig(OUT/'cases.png',dpi=130);plt.close(figure)
    (OUT/'case_provenance.json').write_text(json.dumps(dict(selection='GT面积、事件与seed1234身份哈希；不依据模型预测选择',cases=cases,
        checkpoint_sha256=sources,crop_size=48,display='RGB固定0至0.3反射率，gamma .7；放大不增加原生分辨率'),indent=1,ensure_ascii=False),encoding='utf-8')


def write_documents(records):
    """Maintain the single stage ledger and its fixed data-document explanation."""
    ledger=ROOT/'docs/assets/research/stage_ledger.json';data=json.loads(ledger.read_text(encoding='utf-8'))
    stage=next(s for s in data['stages'] if s['id']=='universal-spectral-encoder')
    external_path=ROOT/'outputs/use/external/acquisition_status.json'
    external=json.loads(external_path.read_text()) if external_path.exists() else {}
    count={s:sum(r['sensor']==s for r in external.get('samples',[])) for s in ('s2','landsat')}
    complete=len(records)==10 and all('full' in r['evaluations'] for r in records)
    stage['delta']='已接通独立观测、共享逐波段空间编码、2×2波段关系、查询汇聚、32→10内部桥接及原网格细节分支；十维内部接口不限制输入波段数。'
    stage['status']='首轮训练与评价完成，待最终报告审查' if complete else '九组监督已验收；外部扩量和预训练尚未完成'
    stage['result']=f"{len(records)}/10组2000更新监督实验完成，独立评价669块；本轮直入参照IoU={records[0]['training']['best_val_iou']:.4f}。"+('; '.join(f"{r['label']}={r['training']['best_val_iou']:.4f}" for r in records[1:]))+f"。外部已接受S2 {count['s2']}、Landsat {count['landsat']}位置；下载状态{external.get('status','unknown')}。单seed，不宣称显著性或跨传感器分割有效。"
    stage['next']='审查完整产物、失败案例和成本后交接。' if complete else '沿completion_status.json继续完整数据验收→10000更新预训练→第10组100步smoke与重新初始化2000步正式微调→固定五组合评价；不重复前九组。'
    stage['handoff'].update(external='scripts/prepare_use_external.py --per-sensor 5000',
        continuation='scripts/run_use.py --finish-stage',continuation_state='outputs/use/completion_status.json',
        pending=['最终报告审查与交接'] if complete else ['外部10000位置及完整验收','10000更新重建预训练','第10组微调与独立评价','最终报告审查与交接'])
    for step,row in zip(stage['execution_plan']['steps'],records):
        step['status']=f"2000更新完成；best事件平均IoU={row['training']['best_val_iou']:.4f}@{row['training']['best_step']}；独立669块评价已完成。"
    stage['evidence']=list(dict.fromkeys(stage['evidence']+['outputs/use/supervised_acceptance.json',
        'outputs/use/supervised_evaluations.json','outputs/use/initialization_audit.json','outputs/use/costs.json',
        'outputs/use/segmentation_interface_acceptance.json','outputs/use/pretraining_engineering_acceptance.json',
        'outputs/use/external/pilot_acceptance.json','outputs/use/external/acquisition_status.json']))
    stage['evidence']=[p for p in stage['evidence'] if (ROOT/p).exists()]
    lookup={r['id']:r for r in records}
    def comparison(name,left,right,meaning):
        a=lookup[left];b=lookup[right]
        return dict(name=name,comparison=a['label']+' 与 '+b['label'],evaluation_context='同一669块开发验证，seed1234，各2000更新；按完整十波段选best',
            observation=f"事件平均IoU：{a['training']['best_val_iou']:.4f} 与 {b['training']['best_val_iou']:.4f}。",
            possible_answer=meaning,evidence=[f'outputs/use/s2_10band_v4r1_p20/use_{n}/1234/eval/eval_dev_full.json' for n in (left,right)])
    stage['comparisons']=[comparison('集合编码的代价','mean','baseline','共享编码可运行可变集合，但首轮分割结果下降，工程等价不代表保留十波段能力。'),
        comparison('波段注意力','self_mean','mean','注意力组略高于均值组，仍远低于直入参照；单seed不能证明稳定收益。'),
        comparison('查询汇聚','self_query','self_mean','查询相对均值的差异较小，不能当作确定改进。'),
        comparison('物理描述与身份容量','physical','random','物理描述未超过同结构随机身份描述；不支持本轮物理信息带来收益的结论。'),
        comparison('参数容量控制','capacity','physical','扩宽无注意力编码控制USE参数量，差2个参数；结果不能排除不同优化难度。'),
        comparison('混合波段训练','physical_mixed','physical','完整十波段IoU下降；还须结合固定同一权重的分组合曲线，不能按组合重新选模。')]
    if 'pretrained_mixed' in lookup:stage['comparisons'].append(comparison('外部预训练整体效果','pretrained_mixed','physical_mixed','只改变USE外部初始化；不能分离Landsat或传感器多样性的作用，没有跨传感器滑坡标签验收。'))
    ledger.write_text(json.dumps(data,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    lines=['## USE首轮：统一监督与外部预训练验收','',
        '### 阅读约定与实验设置','',
        '使用**我们的数据划分（留事件验证）**：九事件2223块训练，广岛448块与特里苏尔221块开发验证，反复用于研发选模，不能当作盲测。单期、无地形，seed1234。十波段统计由559块训练核心拟合；USE输入质量掩膜由源SCL和有限值检查建立，标签可用性只控制监督。', '',
        '各组物理批量8、FP32、无梯度累积，冻结BatchNorm运行统计且训练仿射参数；2000次成功更新、日程3600次、预热180次。AdamW编码器/其他组件学习率6e-5/6e-4、权重衰减0.01，损失0.5 BCE＋0.5 Dice。官方MiT/解码器初始化及随机细节分支相同；100步冒烟权重不续入正式实验。', '',
        '完整模型在127×129奇数尺寸的3/4/6/10/13波段前后向、集合置换及混批补齐通过atol=1e-5、rtol=1e-4验收。初次补充验收漏关TF32曾失败；使用与正式训练一致的严格FP32后通过，不修改模型与旧权重，失败日志保留。13波段仅为合成接口证据。', '',
        f"当前完成{len(records)}/10组监督及独立评价。外部样本S2 {count['s2']}、Landsat {count['landsat']}，状态`{external.get('status','unknown')}`；不足各5000并通过完整审查时不能启动正式预训练。实际阶段状态以[台账](assets/research/stage_ledger.json)与[队列](../outputs/use/completion_status.json)为准。",'',
        '### 主表（合并）','',
        'IoU主列为两事件等权；Dice、Precision、Recall、像素AP、边界F1按669块合并计数。每组均选完整十波段事件平均IoU最高的权重，不能与末态混称。历史0.3081训练设置不同，仅作背景。','',
        '| 模型连接与训练方式（均为SegFormer B0＋原网格细节） | best更新 | 事件平均IoU | 广岛IoU | 特里苏尔IoU | Dice | Precision | Recall | 像素AP | 边界F1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def num(value):return f'{value:.4f}' if isinstance(value,(int,float)) else '—'
    for r in records:
        e=r['evaluations'].get('full')
        if not e:continue
        p=e['overall']['pixel'];events=e['by_event']
        values=[r['label'],str(r['training']['best_step']),num(e['event_mean_iou']),num(events['hiroshima']['pixel']['iou']),num(events['thrissur']['pixel']['iou'])]+[num(p[k]) for k in ('dice','precision','recall','average_precision')]+[num(e['overall']['boundary']['f1'])]
        lines.append('| '+' | '.join(values)+' |')
    lines+=['','逐事件补充口径：对象为8连通真值，排除触及边界或未知标签邻域的截断对象，一对一贪心匹配IoU≥0.25；每个召回值后括号为该面积段可评价对象数。Thrissur没有1–4像素可评价对象，该格无定义。','',
        '| 模型／开发验证事件 | IoU | Dice | Precision | Recall | 像素AP | 边界F1 | 1–4像素对象召回（数量） | 5–25像素对象召回（数量） | 26+像素对象召回（数量） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        for event,e in r['evaluations'].get('full',{}).get('by_event',{}).items():
            p=e['pixel'];values=[r['label']+'／'+dict(hiroshima='广岛',thrissur='特里苏尔').get(event,event)]+[num(p[k]) for k in ('iou','dice','precision','recall','average_precision')]+[num(e['boundary']['f1'])]
            for key in ('1_4','5_25','26_plus'):
                b=e['objects']['use_bins'][key];n=b['n_objects_evaluable'];values.append((num(b['object_recall_evaluable']) if n else '无定义')+f' ({n})')
            lines.append('| '+' | '.join(values)+' |')
    lines+=['','所有输入组合复用同一个最佳完整十波段权重。3=RGB，4=RGB+B08，6=RGB+B08/B11/B12；未见5=B02/B03/B04/B8A/B11，身份和数量同时不同，因此曲线不是纯波段数量效应。','',
        '| 模型与训练方式 | 3波段IoU | 4波段IoU | 未见5波段IoU | 6波段IoU | 10波段IoU |','|---|---:|---:|---:|---:|---:|']
    for r in records:lines.append('| '+' | '.join([r['label']]+[num(r['evaluations'].get(k,{}).get('event_mean_iou')) for k in ('rgb','rgb_nir','unseen5','six','full')])+' |')
    lines+=['','### 图与解释','',
        '![完整十波段性能与最小对象召回](assets/research/use/results.png)','',
        '每行是一种模型或训练方式；左侧比较完整十波段的两事件等权IoU，右侧比较77个最小可评价对象的召回。工程上能接收不同波段，不等于保留分割性能。','',
        '![逐更新选模曲线](assets/research/use/training.png)','',
        '每个子图对应一组正式训练，共用坐标；横轴是成功更新次数，纵轴是两事件等权IoU。虚线仅表示本轮直入参照的最佳值，不是每一步的参照轨迹。','',
        '![固定权重的分波段与目标尺度评价](assets/research/use/bands_sizes.png)','',
        '左图是相同权重接受不同组合，右图是在完整十波段下按对象面积比较召回；只展示均值与物理USE的完整/混合训练及可用的预训练组以保持可读性，全部数值见上表和原始评价。','',
        '### 展开结论','',
        '共享逐波段编码、掩膜汇聚及关系路径通过可变输入和集合置换验收；它们将十条原始反射率变成十维内部特征再进入主干和细节分支，改变了原参照的输入分布。所有已完成的随机初始化USE组均低于本轮直入参照，且最小对象召回下降超过0.02，须登记退化警报。分布干预和梯度连通只证明模块参与计算，不能说明退化原因。','',
        '物理描述组没有超过同结构固定随机身份描述组，无法支持本轮波长与原生分辨率带来收益的结论。带宽未知字段保持缺失；本地S2使用公开名义中心波长，并非逐产品实测响应。局部路径的描述项可能仅形成空间常量，是否影响小目标仍未验证，不据此临时改结构或损失。','',
        '混合波段训练没有完整保留物理USE的十波段能力。比较必须固定同一最佳十波段权重；RGB或未见组合可运行也不能推导真实Landsat滑坡分割有效。外部预训练的整体收益需与同结构混合训练比较，且缺少S2单传感器预训练对照，不能归因于Landsat。','',
        '### 限制与下一步','',stage['next'], '',
        '单seed、同一开发集合反复选模；工程退化阈值不是显著性阈值。外部排除范围是项目影像图块包络加10公里，不是对所有灾害真实影响边界的独立地理核实。外部采样均匀轮转5°网格不保证全球地表分布代表性，重建误差降低不等于灾害语义有效。','',
        '### 附录：逐seed、成本与产物索引','',
        '本轮只有seed1234，每组16000次样本曝光；实际配置、命令、各步权重、逐更新/逐epoch曲线和曝光记录保留在各run目录。下表训练秒数含训练期间验证；推理吞吐为同一RTX5090上真实十波段批量8、FP32、预热10次后30次完整前向，不含文件读取和CPU到GPU搬运；不能换算未测下载耗时。','',
        '| 模型与训练方式 | 参数数 | 训练及验证秒数 | 测量吞吐（图块/秒） | 推理峰值显存MiB |','|---|---:|---:|---:|---:|']
    cost_path=ROOT/'outputs/use/costs.json';costs=json.loads(cost_path.read_text()) if cost_path.exists() else {'records':[]}
    costs={r['model']:r for r in costs['records']}
    for r in records:
        c=costs.get(r['id'],{});lines.append('| '+' | '.join([r['label'],str(c.get('parameters','—')),num(r['training']['seconds']),num(c.get('images_per_second')),num(c.get('peak_allocated_mib'))])+' |')
    lines+=['','证据：[监督独立评价](../outputs/use/supervised_evaluations.json)、[预算及退化警报](../outputs/use/supervised_acceptance.json)、[成本测量方法及原始值](../outputs/use/costs.json)、[初始化回溯审查](../outputs/use/initialization_audit.json)、[外部试取验收](../outputs/use/external/pilot_acceptance.json)、[图表来源哈希](assets/research/use/figure_provenance.json)。早期USE run_meta缺少官方权重事前SHA，回溯当前目录与基线事前记录相符，但不能伪称每组都保留了事前指纹；后续记录已修复。','']
    if (OUT/'cases.png').exists():lines+=['![按真值固定选择的对象案例](assets/research/use/cases.png)','',
        '每行是一处按事件、面积及seed哈希选出的对象邻域；从左到右为RGB、原标注及各模型前景概率，红框定位同一真值对象，白线是0.5预测阈值。每窗48×48原始像素，放大不增加10米原生空间分辨率；案例选择不依据预测，不能替代完整669块定量评价。详情见[案例清单与权重SHA](assets/research/use/case_provenance.json)。','']
    path=ROOT/'docs/DATA_EVENT_PROTOCOL.md';body=path.read_text(encoding='utf-8');start='<!-- USE_RESULTS_START -->';end='<!-- USE_RESULTS_END -->'
    content=start+'\n\n'+'\n'.join(lines)+'\n'+end
    if start in body:
        before,rest=body.split(start);_,after=rest.split(end);body=before+content+after
    else:body=body.rstrip()+'\n\n'+content+'\n'
    path.write_text(body,encoding='utf-8')


if __name__=='__main__':main()

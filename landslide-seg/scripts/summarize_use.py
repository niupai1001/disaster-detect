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
 'physical_mixed':'物理USE＋混合波段训练','pretrained_mixed':'物理USE＋联合无标签预训练后微调'}


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
    for name in ('history.json','sampling.jsonl','resolved_config.json'):
        path=ROOT/'outputs/use/pretraining'/name
        if path.exists():sources.append(path)
    for name in ('outputs/use_diagnostic_review/comparison.json','outputs/use_recovery_probe/short/g1/summary.json',
                 'outputs/use_recovery_probe/short/g2/summary.json','outputs/use/download_performance/comparison.json',
                 'outputs/use/download_performance/parallel_before.json','outputs/use/download_performance/parallel_after.json'):
        if (ROOT/name).exists():sources.append(ROOT/name)
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
    nrows=(len(records)+2)//3
    figure,axes=plt.subplots(nrows,3,figsize=(14,3.1*nrows),sharex=True,sharey=True,squeeze=False)
    for ax,row in zip(axes.flat,records):
        history=row['training']['history'];ax.plot([r['global_step'] for r in history],[r['dev']['event_mean_iou'] for r in history])
        ax.axhline(records[0]['training']['best_val_iou'],ls='--',color='gray',lw=.8)
        ax.set_title(row['label'],fontproperties=font,fontsize=10,pad=24)
        ax.set_ylim(0,max(.35,max(r['training']['best_val_iou'] for r in records)*1.1));ax.grid(alpha=.2)
        ax.tick_params(labelbottom=True,labelsize=8)
        epoch_axis=ax.secondary_xaxis('top',functions=(lambda v:v*8/2223,lambda v:v*2223/8))
        epoch_axis.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(records):]:ax.set_visible(False)
    figure.supxlabel('下轴：成功更新；上轴：样本曝光/2223的等效epoch；虚线为直入最佳IoU',fontproperties=font)
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
    pretrain=ROOT/'outputs/use/pretraining'
    if (pretrain/'history.json').exists():
        history=json.loads((pretrain/'history.json').read_text());figure,axes=plt.subplots(1,2,figsize=(12,4))
        trace=[json.loads(line) for line in (pretrain/'sampling.jsonl').read_text().splitlines()]
        axes[0].plot([r['step'] for r in trace],[r.get('loss',float('nan')) for r in trace],alpha=.4,lw=.5)
        sources=[('s2','已下载S2'),('landsat','Landsat')]
        if history and 'local_s2' in history[0]['validation']:sources.append(('local_s2','原有无标注S2'))
        for sensor,label in sources:
            axes[1].plot([r['step'] for r in history],[r['validation'][sensor] for r in history],marker='o',label=label)
        axes[1].plot([r['step'] for r in history],[r['score'] for r in history],label='两传感器等权',lw=2)
        for ax in axes:ax.set_xlabel('成功预训练更新',fontproperties=font);ax.grid(alpha=.2)
        axes[0].set_ylabel('每更新的遮挡波段标准化L1',fontproperties=font)
        axes[1].set_ylabel('地理留出重建标准化L1',fontproperties=font);axes[1].legend(prop=font)
        figure.tight_layout();figure.savefig(OUT/'pretraining.png',dpi=150);plt.close(figure)
    import sys
    if '--cases' in sys.argv:failure_cases(records)
    followup_figure()
    download_figure()
    write_documents(records)


def download_figure():
    path=ROOT/'outputs/use/download_performance/comparison.json'
    if not path.exists():return
    data=json.loads(path.read_text());font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    figure,ax=plt.subplots(figsize=(8,3))
    times=[data['before_seconds'],data['after_seconds']]
    ax.barh([1,0],times,color=['#7c8795','#3976a8'])
    ax.set_yticks([1,0],['原4并发，固定批次等待','8并发预取，按原顺序提交'],fontproperties=font)
    for y,value in zip([1,0],times):ax.text(value+.3,y,f'{value:.2f}s',va='center')
    ax.set_xlim(0,max(times)*1.2);ax.set_xlabel('同一16个已接受窗口的读取总秒数',fontproperties=font)
    ax.set_title('单次受控读取对照；不含目录查询和候选筛除',fontproperties=font)
    figure.tight_layout();figure.savefig(OUT/'download_performance.png',dpi=150);plt.close(figure)


def followup_figure():
    """Show the already completed engineering branch without mixing it into DEV scores."""
    comparison=ROOT/'outputs/use_diagnostic_review/comparison.json'
    short=[ROOT/f'outputs/use_recovery_probe/short/{name}/summary.json' for name in ('g1','g2')]
    if not comparison.exists() or not all(p.exists() for p in short):return
    data=json.loads(comparison.read_text(encoding='utf-8'))
    rows=data['activation_rows'];groups=[json.loads(p.read_text(encoding='utf-8')) for p in short]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    figure,axes=plt.subplots(1,2,figsize=(14,5))
    names={'input':'原始输入','bridge':'十维桥接','use.features':'USE特征','mit.stage1':'MiT第一级','mit.stage2':'MiT第二级','detail.stem':'原网格细节','decoder':'解码特征'}
    axes[0].barh(range(len(rows)),[r['cka_physical_vs_mixed'] for r in rows])
    axes[0].set_yticks(range(len(rows)),[names[r['layer']] for r in rows],fontproperties=font)
    axes[0].invert_yaxis();axes[0].set_xlim(0,1.05);axes[0].set_xlabel('同一32块TRAIN快照的线性CKA',fontproperties=font)
    axes[0].set_title('完整波段与混合波段训练：同层表示相似度',fontproperties=font)
    for i,r in enumerate(rows):axes[0].text(r['cka_physical_vs_mixed']+.01,i,f"{r['cka_physical_vs_mixed']:.2f}",va='center')
    for i,g in enumerate(groups):
        for offset,key,color in [(-.16,'fit_end','#3976a8'),(.16,'check_end','#bd7648')]:
            value=g[key]['pooled']['iou'];axes[1].bar(i+offset,value,width=.3,color=color,
                label=('96块拟合','32块检查（已参与标定）')[offset>0] if i==0 else None)
            axes[1].text(i+offset,value+.005,f'{value:.4f}',ha='center',fontsize=9)
    axes[1].set_xticks([0,1],['同一桥接标定后微调','同一标定＋两级特征辅助'],fontproperties=font,fontsize=10)
    axes[1].set_ylim(0,.3);axes[1].set_ylabel('TRAIN内部像素合并IoU',fontproperties=font)
    axes[1].set_title('300更新工程探针：不是669块开发验证',fontproperties=font)
    axes[1].legend(prop=font);figure.tight_layout();figure.savefig(OUT/'followup_engineering.png',dpi=150);plt.close(figure)


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
    from datetime import datetime,timezone
    data['as_of']=datetime.now(timezone.utc).date().isoformat()
    stage=next(s for s in data['stages'] if s['id']=='universal-spectral-encoder')
    external_path=ROOT/'outputs/use/external/acquisition_status.json'
    external=json.loads(external_path.read_text()) if external_path.exists() else {}
    count={s:sum(r['sensor']==s for r in external.get('samples',[])) for s in ('s2','landsat')}
    complete=len(records)==10 and all('full' in r['evaluations'] for r in records)
    stage['delta']='已接通独立观测、共享逐波段空间编码、2×2波段关系、查询汇聚、32→10内部桥接及原网格细节分支；十维内部接口不限制输入波段数。'
    stage['status']='首轮训练与评价完成，待最终报告审查' if complete else '九组监督已验收；外部扩量和预训练尚未完成'
    stage['result']=f"{len(records)}/10组2000更新监督实验完成，独立评价669块；本轮直入参照IoU={records[0]['training']['best_val_iou']:.4f}。"+('; '.join(f"{r['label']}={r['training']['best_val_iou']:.4f}" for r in records[1:]))+f"。外部已接受S2 {count['s2']}、Landsat {count['landsat']}位置；下载状态{external.get('status','unknown')}。单seed，不宣称显著性或跨传感器分割有效。"
    stage['next']='审查完整产物、失败案例和成本后交接。' if complete else '沿completion_status.json继续完整数据验收→10000更新预训练→第10组100步smoke与重新初始化2000步正式微调→固定五组合评价；不重复前九组。'
    stage['execution_plan']['basis']=f'用户已授权完整USE首轮及三处范围移交；当前{len(records)}/10组监督训练与独立评价完成，外部扩量及后续进展以实际队列为准，不将计划记为结果。'
    stored_bytes=sum((ROOT/'outputs/use/external'/r['file']).stat().st_size for r in external.get('samples',[]))
    stage['cost']=dict(supervised_train_validation_seconds=sum(r['training']['seconds'] for r in records),
        accepted_external_npz_bytes=stored_bytes,network_transfer_bytes=None,
        note='训练秒数是各单卡运行训练与验证的墙时之和；存储字节不是网络传输字节，实际传输量未记录，不以本地大小冒充。')
    pretrain_summary=ROOT/'outputs/use/pretraining/summary.json'
    if pretrain_summary.exists():stage['external_pretraining']=json.loads(pretrain_summary.read_text())
    frozen_path=ROOT/'outputs/use/external/acquisition_paused.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8')) if frozen_path.exists() else None
    if frozen and not complete:
        stage['status']='九组监督已验收；下载按用户要求暂停，使用冻结样本预训练并继续第十组'
        stage['next']='沿completion_status.json以冻结外部样本及原有无标注TRAIN事件影像完成审查、10000更新联合预训练及原2223块TRAIN上的第十组2000更新微调，669块DEV只验证；不恢复下载。'
        stage['execution_plan']['basis']='用户最新授权：暂停下载，已接受S2 5000/Landsat1700与原有无标注S2共同预训练；原数据仅九个TRAIN事件且annotated=False，排除留出事件。10000预训练/2000微调更新及batch8不变。'
    local_path=ROOT/'outputs/use/external/local_acceptance.json'
    local=json.loads(local_path.read_text()) if local_path.exists() else None
    if local:
        stage['local_unlabeled']={k:v for k,v in local.items() if k not in ('train_ids','validation_ids')}
        stage['local_unlabeled'].update(train_observations=len(local['train_ids']),validation_observations=len(local['validation_ids']))
    pretrained_done=pretrain_summary.exists() and stage['external_pretraining'].get('complete') is True
    if pretrained_done and not complete:
        queue_path=ROOT/'outputs/use/completion_status.json'
        queue=json.loads(queue_path.read_text()) if queue_path.exists() else {}
        stage['status']='联合无标签预训练10000更新及仅USE迁移验收完成；第十组微调/评价待完成'
        stage['next']='已完成预训练不重跑，沿completion_status.json继续复核、100步冒烟、全新初始化2000更新微调及五组合评价；下载保持暂停。'
        stage['execution_plan']['steps'][-1]['status']='联合预训练10000更新完成，best等权重建误差'+str(stage['external_pretraining']['best_score'])+'；仅USE迁移验收通过，后续队列状态：'+queue.get('status','unknown')+'。'
    stage['handoff'].update(external='用户暂停下载；仅使用current_manifest.json冻结样本' if frozen else 'scripts/prepare_use_external.py --per-sensor 5000',
        continuation='scripts/run_use.py --finish-stage'+(' --current-data --local-unlabeled' if frozen else ''),continuation_state='outputs/use/completion_status.json',
        pending=['最终报告审查与交接'] if complete else ['冻结样本完整验收' if frozen else '外部10000位置及完整验收','10000更新重建预训练','第10组微调与独立评价','最终报告审查与交接'])
    if pretrained_done and not complete:stage['handoff']['pending']=['第10组100步冒烟/全新初始化2000更新微调','固定五组合独立评价','最终报告审查与交接']
    if frozen:
        stage['execution_plan']['basis']=f'用户授权的完整首轮已完成{len(records)}/10组；下载按最新要求暂停，实际联合预训练来源为原有3062个无标注S2位置、已下载S2 5000及Landsat1700。原监督协议与预算不变。'
    if complete and stage.get('final_review',{}).get('complete'):
        stage['status']='USE完整首轮完成并交接；工程通过，研究退化警报保留'
        stage['next']='本轮停止训练与自动接续，下载保持暂停。保留十波段直入作为当前参照；后续仅提出输入适配和小目标诊断，不自动启动扩量、蒸馏或新增实验。'
        stage['handoff']['pending']=[]
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
    if 'pretrained_mixed' in lookup:stage['comparisons'].append(comparison('联合无标签预训练整体效果','pretrained_mixed','physical_mixed','只改变USE初始化，预训练加入原有无标注S2及公开S2/Landsat；不能分离各来源或传感器多样性的作用，没有跨传感器滑坡标签验收。'))
    ledger.write_text(json.dumps(data,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    lines=['## USE首轮：统一监督与外部预训练验收','',
        '### 阅读约定与实验设置','',
        '使用**我们的数据划分（留事件验证）**：九事件2223块训练，广岛448块与特里苏尔221块开发验证，反复用于研发选模，不能当作盲测。单期、无地形，seed1234。十波段统计由559块训练核心拟合；USE输入质量掩膜由源SCL和有限值检查建立，标签可用性只控制监督。', '',
        '各组物理批量8、FP32、无梯度累积，冻结BatchNorm运行统计且训练仿射参数；2000次成功更新、日程3600次、预热180次。AdamW编码器/其他组件学习率6e-5/6e-4、权重衰减0.01，损失0.5 BCE＋0.5 Dice。官方MiT/解码器初始化及随机细节分支相同；100步冒烟权重不续入正式实验。', '',
        '完整模型在127×129奇数尺寸的3/4/6/10/13波段前后向、集合置换及混批补齐通过atol=1e-5、rtol=1e-4验收。初次补充验收漏关TF32曾失败；使用与正式训练一致的严格FP32后通过，不修改模型与旧权重，失败日志保留。13波段仅为合成接口证据。', '',
        f"当前完成{len(records)}/10组监督及独立评价。外部样本S2 {count['s2']}、Landsat {count['landsat']}，状态`{external.get('status','unknown')}`；"+('用户已授权以冻结样本及原有无标注S2联合预训练，下载暂停；全部样本须通过物理量、质量、空间排除与SHA审查，保留两传感器1:1曝光和10000更新。每批8个样本为2个已下载S2、2个原有无标注S2及4个Landsat。原有无标注影像仅取九TRAIN事件中annotated=False的位置，每位置选一期，绝不读取MASK。'+(f"已验收原有无标注位置{local['locations']}个，其中重建训练{len(local['train_ids'])}、空间留出{len(local['validation_ids'])}；空间留出只衡量已知TRAIN事件内的重建，不是未见事件滑坡验证。" if local else '原有无标注清单仍在准备审查。') if frozen else '不足各5000并通过完整审查时不能启动正式预训练。')+"实际阶段状态以[台账](assets/research/stage_ledger.json)与[队列](../outputs/use/completion_status.json)为准。",'',
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
        '每个子图对应一组正式训练，共用坐标；下轴是成功更新次数，上轴是累计样本曝光除以2223得到的等效epoch，纵轴是每100次更新评价的两事件等权IoU。虚线仅表示本轮直入参照的最佳值，不是每一步的参照轨迹。前九组只保留100更新窗口的平均训练损失，无法补造逐次损失；第十组另保留per_update.jsonl逐次记录。','',
        '![固定权重的分波段与目标尺度评价](assets/research/use/bands_sizes.png)','',
        '左图是相同权重接受不同组合，右图是在完整十波段下按对象面积比较召回；只展示均值与物理USE的完整/混合训练及可用的预训练组以保持可读性，全部数值见上表和原始评价。','',
        '### 展开结论','',
        '共享逐波段编码、掩膜汇聚及关系路径通过可变输入和集合置换验收；它们将十条原始反射率变成十维内部特征再进入主干和细节分支，改变了原参照的输入分布。所有已完成的随机初始化USE组均低于本轮直入参照，且最小对象召回下降超过0.02，须登记退化警报。分布干预和梯度连通只证明模块参与计算，不能说明退化原因。','',
        '物理描述组没有超过同结构固定随机身份描述组，无法支持本轮波长与原生分辨率带来收益的结论。带宽未知字段保持缺失；本地S2使用公开名义中心波长，并非逐产品实测响应。局部路径的描述项可能仅形成空间常量，是否影响小目标仍未验证，不据此临时改结构或损失。','',
        '混合波段训练的完整十波段IoU为0.1895，比同结构完整训练0.1955低0.0059；这是本seed的数值下降，低于预设0.01报警门槛，不能据此断言稳定退化。比较固定同一最佳十波段权重；RGB或未见组合可运行也不能推导真实Landsat滑坡分割有效。联合预训练与同结构混合训练比较，缺少单来源或S2单传感器预训练对照，不能归因于Landsat。','',
        '### 限制与下一步','',stage['next'], '',
        '单seed、同一开发集合反复选模；工程退化阈值不是显著性阈值。外部排除范围是项目影像图块包络加10公里，不是对所有灾害真实影响边界的独立地理核实。外部采样均匀轮转5°网格不保证全球地表分布代表性，重建误差降低不等于灾害语义有效。','',
        '### 附录：逐seed、成本与产物索引','',
        '本轮只有seed1234，每组16000次样本曝光；实际配置、命令、各步权重、逐更新/逐epoch曲线和曝光记录保留在各run目录。下表训练秒数含训练期间验证；推理吞吐为同一RTX5090上真实十波段批量8、FP32、预热10次后30次完整前向，不含文件读取和CPU到GPU搬运；不能换算未测下载耗时。','',
        '| 模型与训练方式 | 参数数 | 训练及验证秒数 | 测量吞吐（图块/秒） | 推理峰值显存MiB |','|---|---:|---:|---:|---:|']
    if 'pretrained_mixed' in lookup:
        pre=lookup['pretrained_mixed'];ref=lookup['physical_mixed'];base=lookup['baseline']
        value=pre['evaluations']['full']['event_mean_iou'];old=ref['evaluations']['full']['event_mean_iou'];baseline=base['evaluations']['full']['event_mean_iou']
        conclusion=f'联合预训练通过遮挡波段重建学习输入表示；在同一结构和混合采样下，完整十波段事件平均IoU由{old:.4f}提高至{value:.4f}（+{value-old:.4f}），但相对十波段直入{baseline:.4f}仍低{baseline-value:.4f}。最小对象召回仍为4/77，与随机USE混合训练相同，低于直入11/77；5–25像素对象由90/418到93/418，26+像素由551/1184到570/1184，均未达到直入188/418及752/1184。两个验证事件IoU均提高，固定同一权重的3/4/5/6/10波段IoU也均高于随机USE混合训练；这些是本seed的整体观察，不说明某个数据来源单独有效。'
        pos=lines.index('### 限制与下一步');lines[pos:pos]=[conclusion,'']
    cost_path=ROOT/'outputs/use/costs.json';costs=json.loads(cost_path.read_text()) if cost_path.exists() else {'records':[]}
    costs={r['model']:r for r in costs['records']}
    for r in records:
        c=costs.get(r['id'],{});lines.append('| '+' | '.join([r['label'],str(c.get('parameters','—')),num(r['training']['seconds']),num(c.get('images_per_second')),num(c.get('peak_allocated_mib'))])+' |')
    lines+=['','证据：[监督独立评价](../outputs/use/supervised_evaluations.json)、[预算及退化警报](../outputs/use/supervised_acceptance.json)、[成本测量方法及原始值](../outputs/use/costs.json)、[初始化回溯审查](../outputs/use/initialization_audit.json)、[外部试取验收](../outputs/use/external/pilot_acceptance.json)、[图表来源哈希](assets/research/use/figure_provenance.json)。早期USE run_meta缺少官方权重事前SHA，回溯当前目录与基线事前记录相符，但不能伪称每组都保留了事前指纹；后续记录已修复。','']
    lines += [f"已完成监督训练及验证单卡墙时合计{stage['cost']['supervised_train_validation_seconds']/3600:.3f}小时（冒烟与工程测试另计，不是GPU实际利用小时）；已接受外部NPZ存储{stored_bytes/2**30:.3f}GiB。网络实际传输字节未记录，不能用本地文件大小替代。",'']
    if pretrained_done:
        seconds=stage['external_pretraining']['seconds']
        lines += [f'联合预训练训练及验证单卡墙时为{seconds/60:.2f}分钟，10000次更新、80000次曝光；十组监督共20000更新、160000次曝光，另有十组100步冒烟共1000更新。第2000步保存在last.pt中。混合组的band_views.jsonl由已保存的样本身份与累计曝光按确定性规则重建，并非逐批原始观测日志；三组均为十波段8015、RGB2644、RGB+B08 2669、六波段2672次曝光。','']
    if (ROOT/'outputs/use/final_acceptance.json').exists():
        lines += ['最终核查命令为`python -X utf8 scripts/audit_use.py --final-acceptance`；核对十组预算、样本曝光、固定采样规则、全部步权重及严格重载、独立评价与权重身份、两事件成员数及对象分母、预训练来源和迁移。验收通过表示工程与产物完整，所有USE组相对直入的退化警报仍保留。证据：[最终核查](../outputs/use/final_acceptance.json)、[配置/权重/清单/文档SHA256](../outputs/use/artifacts.sha256)、[仅USE迁移](../outputs/use/transfer_acceptance.json)。','']
    if (OUT/'cases.png').exists():lines+=['![按真值固定选择的对象案例](assets/research/use/cases.png)','',
        '每行是一处按事件、面积及seed哈希选出的对象邻域；从左到右为RGB、原标注及各模型前景概率，红框定位同一真值对象，白线是0.5预测阈值。每窗48×48原始像素，放大不增加10米原生空间分辨率；案例选择不依据预测，不能替代完整669块定量评价。详情见[案例清单与权重SHA](assets/research/use/case_provenance.json)。','']
    if (OUT/'pretraining.png').exists():
        lines+=['![遮挡波段预训练曲线](assets/research/use/pretraining.png)','',
            '左侧每个点是一成功更新的训练L1；右侧每500更新评价一次，分别展示已下载S2、原有无标注S2、Landsat空间留出的重建误差。红线权重为前两者各1/4、Landsat为1/2，与批量配比一致。目标是训练统计标准化的反射率，数值不是原始反射率绝对误差，不能直接比较物理难度；第500步首次评价0.5849至10000步0.3526下降39.7%，并非相对未训练模型的变化。重建降低本身不证明滑坡分割收益。','']
    if 'pretrained_mixed' in lookup:
        delta=lookup['pretrained_mixed']['training']['best_val_iou']-lookup['physical_mixed']['training']['best_val_iou']
        lines += [f'联合无标签预训练整体比较：同一物理USE结构与混合波段分布下，预训练后微调相对随机USE的完整十波段事件平均IoU差为{delta:+.4f}。预训练合并原有无标注S2及公开S2/Landsat；该比较只有一个seed，不能单独归因于任何来源或传感器多样性；重建任务使用的无标签监督不是滑坡语义监督。','']
    path=ROOT/'docs/DATA_EVENT_PROTOCOL.md';body=path.read_text(encoding='utf-8');start='<!-- USE_RESULTS_START -->';end='<!-- USE_RESULTS_END -->'
    content=start+'\n\n'+'\n'.join(lines)+'\n'+end
    if start in body:
        before,rest=body.split(start);_,after=rest.split(end);body=before+content+after
    else:body=body.rstrip()+'\n\n'+content+'\n'
    path.write_text(body,encoding='utf-8')


if __name__=='__main__':main()

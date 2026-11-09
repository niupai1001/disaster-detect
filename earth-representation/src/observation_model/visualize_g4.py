"""Read-only trained G4 comparisons from the exact frozen execution sources."""
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .expanded_dataset import G1OpticalDataset
from .expansion_data import ROOT,OUT,read,save,digest

BASE=ROOT/'observation_model/outputs/encoder_v2'
G4=BASE/'G4_fit';DEST=G4/'prediction_visuals'


def model_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    started=time.perf_counter();DEST.mkdir(exist_ok=True)
    if (DEST/'index.json').exists():raise FileExistsError('Preserve published prediction cases')
    for name,sha in read(G4/'source_manifest.json').items():assert digest(G4/'source'/name)==sha
    # Frozen package resolves all relative imports against G4's actual sources.
    package=types.ModuleType('_g4_prediction_source');package.__path__=[str(G4/'source')]
    sys.modules[package.__name__]=package
    architecture=importlib.import_module(package.__name__+'.encoder_v2')
    tasks=importlib.import_module(package.__name__+'.encoder_v2_tasks')
    controls=importlib.import_module(package.__name__+'.encoder_v2_controls')
    device='cuda:1';torch.set_num_threads(4);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
    checkpoints={name:torch.load(G4/(name+'.pt'),map_location='cpu',weights_only=False) for name in ['best','last']}
    assert checkpoints['best']['state']['update']==150 and checkpoints['last']['state']['update']==200
    models={name:architecture.ObservationEncoderV2().to(device).eval() for name in checkpoints}
    for name,model in models.items():model.load_state_dict(checkpoints[name]['model'])
    hashes={name:model_hash(model) for name,model in models.items()}
    checkpoint_hashes={name:digest(G4/(name+'.pt')) for name in models}
    fits=torch.load(G4/'ridge_TRAIN_only.pt',map_location=device,weights_only=False)
    stats=read(OUT/'statistics.json')['bands'];members=read(G4/'members.json')
    datasets={role:G1OpticalDataset(OUT,BASE/'G1',role) for role in ['TRAIN','VAL']}
    indices={role:{r['id']:i for i,r in enumerate(ds.rows)} for role,ds in datasets.items()}
    entries={'A':[],'B':[]};index=[]
    with torch.inference_mode():
        for role in ['TRAIN','VAL']:
            for sensor in ['S2','L8']:
                row=next(r for r in members if r['role']==role and r['sensor']==sensor)
                assert digest(Path(row['cache']))==row['cache_sha256']
                item=datasets[role][indices[role][row['id']]];image=architecture.image_from_g1(item,device)
                for kind in ['A','B']:
                    band_key=('B08' if sensor=='S2' else 'SR_B5') if kind=='A' else ('B11' if sensor=='S2' else 'SR_B6')
                    original=next(b for b in image.bands if b.band_key==band_key)
                    task=tasks.spatial_task(image,tasks.seed_for(row['id'],'A',0)) if kind=='A' else tasks.band_task(image,band_key)
                    target=next(t for t in task.targets if t.band_key==band_key)
                    predictions={}
                    for name,model in models.items():
                        prediction,coverage=model.predict(task.visible,target.grid)
                        assert torch.equal(target.valid&coverage,target.valid)
                        predictions[name]=prediction
                    if kind=='A':
                        visible=next(b for b in task.visible.bands if b.band_key==band_key)
                        baseline=controls.spatial_prediction(visible,stats[row['instrument']+':'+band_key]['mean'])
                        baseline_name='四近邻空间插值'
                    else:
                        visible=task.visible.bands[0]
                        baseline=controls.ridge_prediction(task.visible,original,fits[sensor+':'+band_key])
                        baseline_name='TRAIN-only Ridge'
                    q=target.valid;std=stats[row['instrument']+':'+band_key]['std'];metrics={}
                    for name,prediction in dict(baseline=baseline,**predictions).items():
                        error=prediction[q]-target.values[q]
                        metrics[name]=dict(mae=float(error.abs().mean()),rmse=float(error.square().mean().sqrt()),
                            nmse=float((error/std).square().mean()),prediction_variance=float(prediction[q].var(unbiased=False)))
                    arrays=dict(visible=visible.values.cpu().numpy(),visible_valid=visible.keep().cpu().numpy(),
                        target=target.values.cpu().numpy(),valid=q.cpu().numpy(),baseline=baseline.cpu().numpy(),
                        best=predictions['best'].cpu().numpy(),last=predictions['last'].cpu().numpy())
                    filename=f'{role}_{sensor}_{kind}_{band_key}.npz';path=DEST/filename
                    np.savez_compressed(path,**arrays)
                    meta=dict(member=row['id'],role=role,sensor=sensor,instrument=row['instrument'],task=kind,
                        target_band=band_key,visible_display_band=visible.band_key,gsd_m=original.support,
                        full_native_target_shape=list(original.values.shape),valid_pixels=int(q.sum()),
                        target_transform=original.transform,crs=original.crs,baseline_name=baseline_name,metrics=metrics,
                        input_hash=tasks.input_hash(task),arrays=filename,arrays_sha256=digest(path),
                        source_cache_sha256=row['cache_sha256'],selection='First frozen member per role/sensor; predefined NIR(A)/SWIR1(B), no quality selection')
                    index.append(meta);entries[kind].append((meta,arrays))
    assert hashes=={name:model_hash(model) for name,model in models.items()}
    assert checkpoint_hashes=={name:digest(G4/(name+'.pt')) for name in models}
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    for kind,rows in entries.items():
        fig,axes=plt.subplots(len(rows),6,figsize=(18,3.4*len(rows)),layout='constrained',squeeze=False)
        for i,(meta,a) in enumerate(rows):
            q=a['valid'];truth=a['target'];lo,hi=np.quantile(truth[q],[.01,.99]);hi=max(hi,lo+1e-6)
            axes[i,0].imshow(np.ma.masked_where(~a['visible_valid'],a['visible']),cmap='viridis')
            axes[i,0].set_title('可见 '+meta['visible_display_band']+('（目标整条不输入）' if kind=='B' else '（白色为不可见）'),fontsize=10)
            for col,name,title in [(1,'target','真实目标 '+meta['target_band']),(2,'baseline',meta['baseline_name']),
                (3,'best','最佳150步'),(4,'last','末态200步')]:
                axes[i,col].imshow(np.ma.masked_where(~q,a[name]),cmap='viridis',vmin=lo,vmax=hi)
                score='' if name=='target' else f'\nMAE={meta["metrics"][name]["mae"]:.4f}'
                axes[i,col].set_title(title+score,fontsize=10)
            error=np.abs(a['best']-truth);maximum=max(float(np.quantile(error[q],.99)),1e-6)
            m=axes[i,5].imshow(np.ma.masked_where(~q,error),cmap='magma',vmin=0,vmax=maximum)
            axes[i,5].set_title('最佳模型绝对误差',fontsize=10);fig.colorbar(m,ax=axes[i,5],shrink=.75)
            axes[i,0].set_ylabel(f'{"训练" if meta["role"]=="TRAIN" else "地域留出"} {meta["sensor"]}\n{meta["gsd_m"][0]:g} m；{meta["valid_pixels"]}有效目标像元')
            for ax in axes[i]:ax.tick_params(labelsize=7)
        fig.suptitle(f'G4训练后固定案例 · {kind}：'+('空间遮盖恢复' if kind=='A' else '整波段恢复')+
            '\n每行目标/参照/两个模型共用真实目标色标；白色不评价；仅显示裁剪色标，不改变数值',fontsize=13)
        fig.savefig(DEST/(kind+'_comparison.png'),dpi=150);plt.close(fig)
    shutil.copyfile(Path(__file__),DEST/'visualize_g4_source.py')
    save(DEST/'index.json',dict(cases=index,source_snapshot='../source',best_update=150,last_update=200,
        checkpoint_sha256=checkpoint_hashes,model_hash_before=hashes,model_hash_after=hashes,
        inference_only=True,optimizer_updates=0,device=device,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        wall_seconds=time.perf_counter()-started,script_sha256=digest(Path(__file__))))
    run=BASE/'tvlp/encoder_v2_G4_fit200'
    events=[json.loads(x) for x in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    for kind in ['A','B']:
        filename=kind+'_comparison.png';destination=run/'artifacts'/filename;shutil.copyfile(DEST/filename,destination)
        event=dict(type='artifact',timestamp=datetime.now(timezone.utc).isoformat(),session_id=events[-1]['session_id'],
            path='artifacts/'+filename,artifact_type='image',tag='trained-fixed-'+kind,phase='val',global_step=200,
            role='comparison',metadata=dict(generated_after_training=True,best_update=150,last_update=200,
                selection='fixed first member per role/sensor, not chosen by quality'))
        with (run/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    p=G4/'README.md';body=p.read_text(encoding='utf-8')
    body+='\n## 训练后真实预测案例\n\n以下按固定成员顺序选择训练/地域留出的S2与Landsat各一份，并预先指定A近红外、B短波红外1，不按模型效果挑例。模型仅推理，原权重与训练结果未改。\n\n![空间遮盖恢复](prediction_visuals/A_comparison.png)\n\n![整波段恢复](prediction_visuals/B_comparison.png)\n\n每图四行依次为训练S2、训练Landsat、地域留出S2、地域留出Landsat；六列为可见输入、真实目标、同任务简单方法、最佳150步、末态200步、最佳模型绝对误差。每行目标及预测共用色标；A只显示被遮盖的合法目标区域，白色不参与评价。图像是逐例结果，不能代替全波段固定评分。逐例原始数组、输入/输出与权重哈希、推理成本见prediction_visuals/index.json。\n'
    p.write_text(body,encoding='utf-8')
    print('G4_TRAINED_PREDICTION_VISUALS_COMPLETE',time.perf_counter()-started)


if __name__=='__main__':main()

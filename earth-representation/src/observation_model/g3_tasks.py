"""G3 task/mask/Viewer acceptance. Exactly zero optimizer updates."""
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .encoder_v2 import ObservationEncoderV2,OpticalImage
from .encoder_v2_tasks import (spatial_task,band_task,seed_for,curriculum,balanced_loss,run_task,
    aggregate_metrics,input_hash,tensor_hash)
from .encoder_v2_runtime import BASE,G3,LOG_ROOT,V2Run,selected_members,load_members,checkpoint,restore
from .check_encoder_v2 import state_hash
from .expansion_data import ROOT,OUT,read,save,digest


def task_preview(entries,destination,kind):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    rows=[]
    for meta,image,task in entries:
        resolutions=[10,20] if meta['sensor']=='S2' and kind=='A' else [task.targets[0].grid.transform[0]]
        for g in resolutions:
            target=next(t for t in task.targets if t.grid.transform[0]==g)
            original=next(b for b in image.bands if b.band_key==target.band_key)
            source=next((b for b in task.visible.bands if b.band_key==target.band_key),None)
            rows.append((meta,image,task,target,original,source))
    columns=7 if kind=='B' else 6
    fig,axes=plt.subplots(len(rows),columns,figsize=(3*columns,3.5*len(rows)),squeeze=False,layout='constrained')
    for index,(meta,image,task,target,original,source) in enumerate(rows):
        raw=original.values.detach().cpu().numpy();qa=original.valid.cpu().numpy();mask=target.valid.cpu().numpy()
        lo,hi=np.quantile(raw[qa],[.01,.99]);limits=dict(vmin=lo,vmax=max(hi,lo+1e-6),cmap='viridis')
        axes[index,0].imshow(np.ma.masked_where(~qa,raw),**limits);axes[index,0].set_title('原始 '+target.band_key)
        axes[index,1].imshow(qa,vmin=0,vmax=1,cmap='gray');axes[index,1].set_title('真实QA有效区')
        if source is not None:
            visible=source.keep().cpu().numpy();values=source.values.detach().cpu().numpy()
            axes[index,2].imshow(np.ma.masked_where(~visible,values),**limits);axes[index,2].set_title('模型可见像元')
        else:
            axes[index,2].imshow(np.zeros_like(raw),cmap='gray',vmin=0,vmax=1);axes[index,2].set_title('目标整波段不输入')
        target_col=3
        if kind=='B':
            other=task.visible.bands[0];a=other.values.cpu().numpy();q=other.valid.cpu().numpy()
            axes[index,3].imshow(np.ma.masked_where(~q,a),cmap='viridis');axes[index,3].set_title('同位置可见 '+other.band_key)
            target_col=4
        axes[index,target_col].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[index,target_col].set_title('真实监督区 '+str(int(mask.sum()))+'像元')
        grid=task.meta['encoder_grid'];left,bottom,right,top=grid['bounds'];ny,nx=grid['shape']
        ax=axes[index,target_col+1]
        bounds=original.bounds
        ax.imshow(raw,extent=(bounds[0],bounds[2],bounds[1],bounds[3]),**limits)
        for y in np.linspace(bottom,top,ny+1):ax.axhline(y,color='white',lw=.25)
        for x in np.linspace(left,right,nx+1):ax.axvline(x,color='white',lw=.25)
        ax.set_title(f'编码Token footprint {ny}×{nx}')
        ax=axes[index,target_col+2];ax.imshow(raw,**limits)
        for x in np.linspace(-.5,raw.shape[1]-.5,min(15,raw.shape[1])+1):ax.axvline(x,color='white',lw=.3)
        for y in np.linspace(-.5,raw.shape[0]-.5,min(15,raw.shape[0])+1):ax.axhline(y,color='white',lw=.3)
        ax.set_title(f'目标原生网格 {raw.shape[1]}×{raw.shape[0]}\n网线稀疏显示，不改变像元')
        area=(original.bounds[2]-original.bounds[0])*(original.bounds[3]-original.bounds[1])
        fraction=task.meta.get('mask_area_fraction',1.)
        axes[index,0].set_ylabel(f'{meta["role"]} {meta["sensor"]} {original.support[0]:g}m\n地面{area:.0f}m²；遮盖{fraction:.1%}')
        for ax in axes[index]:ax.tick_params(labelsize=6)
    fig.suptitle(f'G3零更新任务预览：{kind}；每行一个真实原生目标，监督数量为本次前向曝光',fontsize=13)
    destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=130);plt.close(fig)


def unit_checks():
    rows=[dict(sensor='S2',band='red',loss=torch.tensor(1.)),dict(sensor='S2',band='red',loss=torch.tensor(9.)),
        dict(sensor='S2',band='nir',loss=torch.tensor(4.)),dict(sensor='L8',band='red',loss=torch.tensor(.25))]
    assert float(balanced_loss(rows))==2.375
    short=(torch.ones(4)/2).square().mean();long=(torch.ones(400)/2).square().mean();assert short==long
    rejected=False
    try:balanced_loss([])
    except ValueError:rejected=True
    assert rejected
    save(G3/'loss_tests.json',dict(status='PASS',manual_expected=2.375,observed=float(balanced_loss(rows)),
        pixel_count_does_not_change_weight=True,empty_supervision_rejected=True,TRAIN_statistics_only=True))
    progress=[0,.35,.55,.75,1];expected=[(1,0),(1,0),(.625,.5),(.25,1),(.25,1)]
    actual=[curriculum(u) for u in progress];assert np.allclose(actual,expected)
    for u in [.35,.75]:assert np.max(np.abs(np.array(curriculum(u-1e-7))-np.array(curriculum(u+1e-7))))<1e-5
    save(G3/'scheduler_tests.json',dict(status='PASS',progress=progress,weights=actual,continuous=True,optimizer_updates=0))


def main():
    G3.mkdir(parents=True,exist_ok=True)
    if (G3/'acceptance.json').exists():raise FileExistsError('Preserve G3 accepted artifacts')
    assert read(BASE/'G2_arch/approval_state.json')['status']=='PASS'
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
    members=selected_members();save(G3/'members.json',members);unit_checks()
    device='cuda:1';torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
    model=ObservationEncoderV2().to(device).eval()
    model.load_state_dict(torch.load(BASE/'G2_arch/initialization_untrained.pt',map_location='cpu',weights_only=True)['model'])
    before=state_hash(model);statistics=read(OUT/'statistics.json')['bands'];data=load_members(members,device)
    source_names=['encoder_v2.py','encoder_v2_tasks.py','encoder_v2_runtime.py','g3_tasks.py','model.py','expanded_dataset.py']
    sources={name:digest(ROOT/'observation_model'/name) for name in source_names}
    for name in source_names:
        target=G3/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'observation_model'/name,target)
    save(G3/'source_manifest.json',sources)
    contract=dict(experiment_id='encoder_v2_G3_zero_update',approval_gate='G3',source_sha=sources,
        data_manifest_hash=read(OUT/'release.json')['manifest_sha256'],statistics_hash=digest(OUT/'statistics.json'),
        permitted_optimizer_updates=0,inputs='G1 accepted native S2 10/20m and Landsat8 30m',
        training_task=['A_spatial_MAE','B_whole_band'],mask_space='60m world-coordinate blocks, native grid anchor; before all value operations',
        A_mask_area_fraction_candidate=.60,A_native_pixel_rule='remove every intersecting pixel; supervise only real valid pixels entirely inside full-patch domain',
        B_mask_target_scope='entire_native_band_grid',composite_rule='B target removed from both A and B inputs; A only supervises remaining bands',
        token_count_formula='400*(A/1200^2)*(10/g_min); alpha1, Nmax8192',
        supervision='TRAIN-only std normalized MSE; mean sensor, band, observation, real valid pixel',
        viewer_attached_before_first_update=False,viewer_attached_before_dry_run=True,
        automatic_continuation=False,external_download_allowed=False,members=len(members),
        training_status='G3 has no optimizer steps; next explicit automation instruction authorizes separate bounded G4')
    save(G3/'task_contract.json',contract)
    run=V2Run(contract['experiment_id'],'G3任务与课程验收｜零更新',contract)
    started=time.perf_counter();run.heartbeat(0);previews={};events=[];plan=[];losses={};all_metrics={}
    try:
        for role in ['TRAIN','VAL']:
            for sensor in ['S2','L8']:
                meta,image=next((m,im) for m,im in data if m['role']==role and m['sensor']==sensor)
                first=spatial_task(image,seed_for(meta['id'],'A',0));second=spatial_task(image,seed_for(meta['id'],'A',1))
                repeat=spatial_task(image,seed_for(meta['id'],'A',0));assert input_hash(first)==input_hash(repeat)
                assert first.meta['selected_patches']!=second.meta['selected_patches'] and len(first.meta['selected_patches'])>1
                for draw,task in enumerate([first,second]):
                    with torch.no_grad():rep=model.encode(task.visible)
                    task.meta['encoder_grid']=rep.grids[0]
                    previews.setdefault(f'A_{role}_{draw}',[]).append((meta,image,task))
                    plan.append(dict(member=meta['id'],role=role,kind='A',draw=draw,mask=task.meta,input_hash=input_hash(task)))
                key='B11' if sensor=='S2' else 'SR_B5';btask=band_task(image,key)
                with torch.no_grad():rep=model.encode(btask.visible)
                btask.meta['encoder_grid']=rep.grids[0];previews.setdefault(f'B_{role}',[]).append((meta,image,btask))
                assert all(b.band_key!=key for b in btask.visible.bands)
                plan.append(dict(member=meta['id'],role=role,kind='B',target_keys=[b.band_key for b in image.bands],
                    preview_target=key,input_hash=input_hash(btask)))
                for task in [first,btask]:
                    with torch.no_grad():rows,exposure,cases=run_task(model,task,meta,statistics,True)
                    all_metrics.setdefault((role,task.kind),[]).extend(rows)
                    events.append(dict(member=meta['id'],role=role,kind=task.kind,exposure=exposure,
                        outputs=[dict(band=c['band'],output_hash=c['output_hash']) for c in cases]))
                    if role=='TRAIN' and sensor=='S2':
                        # One actual A/B gradient comparison; no optimizer step.
                        grad_rows,_,_=run_task(model,task,meta,statistics)
                        loss=balanced_loss(grad_rows)
                        grads=torch.autograd.grad(loss,model.parameters(),allow_unused=True)
                        norm=float(torch.stack([g.square().sum() for g in grads if g is not None]).sum().sqrt())
                        assert np.isfinite(norm) and norm>0;losses[task.kind]=norm
                        del grads,grad_rows,loss
                    run.heartbeat(0)
        for (role,kind),rows in all_metrics.items():
            metrics=aggregate_metrics(rows);run.log('diagnostic_'+kind if role=='TRAIN' else 'val_'+kind,metrics,0)
        run.log('gradients',dict(A_norm=losses['A'],B_norm=losses['B']),0)
        run.log('capacity',dict(source_observations=4,encoder_views=len(events),target_pixels=sum(e['exposure']['target_pixels'] for e in events),
            visible_pixels=sum(e['exposure']['visible_source_pixels'] for e in events),native_pixels=sum(e['exposure']['native_input_pixels'] for e in events),
            tokens=sum(e['exposure']['tokens'] for e in events),peak_gpu_MiB=torch.cuda.max_memory_allocated()/1024**2),0)
        save(G3/'fixed_plans.json',plan);save(G3/'actual_exposure.json',events)
        for name,entries in previews.items():
            path=G3/'preview'/(name+'.png');task_preview(entries,path,name[0])
            target=run.dir/'artifacts'/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            run.artifact('artifacts/'+path.name)
        card='<html><meta charset="utf-8"><h1>G3：任务验收，0次优化更新</h1><h2>当前任务</h2><p>60米随机多位置空间遮盖；整波段移除。真实TRAIN/VAL角色隔离。</p><h2>质量指标</h2><p>diagnostic_A/B与val_A/B是随机初始化实际前向，不是训练收敛曲线。</p><h2>容量与监督</h2><pre>'+json.dumps(events,ensure_ascii=False,indent=2)+'</pre><h2>执行状态</h2><p>G3预算0；主线程心跳；恢复会话测试；物理损失关闭。</p></html>'
        target=run.dir/'artifacts/task_card.html';target.write_text(card,encoding='utf-8');run.artifact('artifacts/task_card.html','html','task-card')
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.04)
        state=dict(update=0,sampler_cursor=0,exposure=dict(observations=0,targets=0,pixels=0),
            scheduler=dict(progress=0.,weights=curriculum(0)),early_stop=dict(best=None,waiting=0))
        checkpoint(G3/'resume_zero_update.pt',model,optimizer,state,run,device)
        expected=(random.random(),float(np.random.random()),float(torch.rand(())))
        sequence=run.sequence;old_session=run.session_id;run.finish('cancelled','Intentional zero-update session restore test')
        payload=restore(G3/'resume_zero_update.pt',model,optimizer,device)
        assert expected==(random.random(),float(np.random.random()),float(torch.rand(())))
        assert payload['state']==state and payload['viewer_sequence']==sequence and state_hash(model)==before
        run=V2Run(contract['experiment_id'],'G3任务与课程验收｜零更新',contract,resume=True,sequence=sequence)
        assert run.session_id!=old_session
        run.heartbeat(0);run.log('state',dict(optimizer_updates=0,remaining_updates=0,resume_count=1),0)
        save(G3/'resume_tests.json',dict(status='PASS',optimizer_updates=0,initial_optimizer_state_empty=True,
            nonempty_Adam_state_restore='Requires G4 actual updates; not claimed here',model_RNG_sampler_scheduler_exposure_sequence=True,
            old_session=old_session,new_session=run.session_id,viewer_sequence=run.sequence))
        assert state_hash(model)==before
        run.finish('completed','G3 engineering complete; zero optimizer updates; awaiting protocol validation')
        save(G3/'dry_run.json',dict(status='PASS',device=device,optimizer_updates=0,model_hash_before=before,model_hash_after=state_hash(model),
            wall_seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),gradient_norms=losses))
        (G3/'task_contract.md').write_text('# G3真实任务卡\n\nA：60米世界坐标多位置随机遮盖，候选60%；整块像元相交即移除，边界及QA无效像元不监督。\n\nB：整个真实目标波段从输入删除，目标值与QA仅用于损失。混合A/B时目标B也从A输入删除。\n\n损失：逐有效像元→逐观测→逐波段→逐传感器等权；标准差只来自冻结TRAIN统计。\n\n八份工程成员（四TRAIN、四VAL）和完整原生范围均已冻结；G3前向实际使用其中每角色/传感器首份共四份，其余供后续有界诊断。随机初始化前向不能解释为泛化成绩。\n\nG3优化更新严格0，物理损失关闭；候选课程仅做端点/连续性测试，未宣布是最终训练策略。Viewer已在前向前attach；断点测试的Adam状态为空，非空状态需G4实训再验证。\n',encoding='utf-8')
        print('G3_DRY_RUN_COMPLETE_PENDING_VIEWER_VALIDATION',flush=True)
    except Exception as exc:
        try:run.finish('failed',str(exc))
        except Exception:pass
        save(G3/'failure.json',dict(status='FAIL_STOP_G3',error=repr(exc),optimizer_updates=0));raise


if __name__=='__main__':main()

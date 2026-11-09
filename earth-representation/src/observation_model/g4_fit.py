"""Explicitly authorized <=200-update diagnostic, not a formal pretraining queue."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import random
import shutil
import time

import numpy as np
import torch

from .encoder_v2 import ObservationEncoderV2
from .encoder_v2_tasks import (spatial_task,training_spatial_task,band_task,seed_for,curriculum,run_task,balanced_loss,aggregate_metrics)
from .encoder_v2_runtime import BASE,G3,G4,V2Run,load_members,checkpoint,restore
from .encoder_v2_controls import fit_ridge,ridge_prediction,spatial_prediction
from .check_encoder_v2 import state_hash
from .expansion_data import ROOT,OUT,read,save,digest


def append(path,row):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def metric_row(meta,target,prediction,statistics):
    q=target.valid;std=statistics[meta['instrument']+':'+target.band_key]['std']
    error=prediction[q]-target.values[q]
    return dict(sensor=meta['sensor'],band=target.band_key,loss=(error/std).square().mean())


@torch.no_grad()
def controls(data,statistics,fits):
    output={}
    for role in ['TRAIN','VAL','TRAIN_NEW_MASK']:
        rows=defaultdict(list)
        for meta,image in data:
            if meta['role']!=('TRAIN' if role=='TRAIN_NEW_MASK' else role):continue
            task=spatial_task(image,seed_for(meta['id'],'A',1 if role=='TRAIN_NEW_MASK' else 0))
            for target in task.targets:
                mean=statistics[meta['instrument']+':'+target.band_key]['mean']
                rows['A_mean'].append(metric_row(meta,target,torch.full_like(target.values,mean),statistics))
                band=next(b for b in task.visible.bands if b.band_key==target.band_key)
                rows['A_spatial'].append(metric_row(meta,target,spatial_prediction(band,mean),statistics))
            if role=='TRAIN_NEW_MASK':continue
            for band in image.bands:
                target=band_task(image,band.band_key).targets[0]
                mean=statistics[meta['instrument']+':'+band.band_key]['mean']
                rows['B_mean'].append(metric_row(meta,target,torch.full_like(band.values,mean),statistics))
                rows['B_ridge'].append(metric_row(meta,target,ridge_prediction(image,band,fits[meta['sensor']+':'+band.band_key]),statistics))
        output[role]={key:float(balanced_loss(value)) for key,value in rows.items()}
    return output


@torch.no_grad()
def evaluate(model,data,statistics,run,step):
    model.eval();result={}
    for role in ['TRAIN','VAL','TRAIN_NEW_MASK']:
        rows=defaultdict(list)
        for meta,image in data:
            if meta['role']!=('TRAIN' if role=='TRAIN_NEW_MASK' else role):continue
            run.heartbeat(step)
            task=spatial_task(image,seed_for(meta['id'],'A',1 if role=='TRAIN_NEW_MASK' else 0))
            r,exposure,_=run_task(model,task,meta,statistics);rows['A']+=r
            append(G4/'evaluation_exposure.jsonl',dict(step=step,role=role,member=meta['id'],task='A',exposure=exposure))
            if role=='TRAIN_NEW_MASK':continue
            for band in image.bands:
                r,exposure,_=run_task(model,band_task(image,band.band_key),meta,statistics);rows['B']+=r
                append(G4/'evaluation_exposure.jsonl',dict(step=step,role=role,member=meta['id'],task='B',target=band.band_key,exposure=exposure))
        result[role]={}
        for kind,values in rows.items():
            metrics=aggregate_metrics(values);result[role][kind]=metrics
            phase='train_new_mask_'+kind if role=='TRAIN_NEW_MASK' else ('fixed_train_' if role=='TRAIN' else 'val_')+kind
            run.log(phase,metrics,step)
    append(G4/'evaluation_history.jsonl',dict(step=step,metrics=result))
    return result


def task_weights(tasks):
    weights={}
    for kind in ['A','B']:
        counts=Counter((meta['sensor'],t.band_key) for meta,task in tasks if task.kind==kind for t in task.targets)
        sensors={s for s,b in counts};bands={s:len({b for ss,b in counts if ss==s}) for s in sensors}
        for (sensor,band),count in counts.items():weights[kind,sensor,band]=1/(len(sensors)*bands[sensor]*count)
    return weights


def gradient_vector(model):
    return torch.cat([(p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()) for p in model.parameters()])


def main():
    G4.mkdir(parents=True,exist_ok=True)
    if (G4/'status.json').exists():raise FileExistsError('Do not duplicate or revive an existing G4 run')
    assert read(G3/'acceptance.json')['status']=='PASS'
    authorization=read(G4/'authorization.json')
    assert authorization['permitted_optimizer_updates']==200 and authorization['authorized']
    assert read(G3/'viewer_preflight.json')['error_count']==0
    members=read(G3/'members.json');device='cuda:1'
    config=dict(seed=0,max_updates=200,effective_source_batch=4,microbatch=1,precision='FP32',
        optimizer='AdamW',lr=5e-4,weight_decay=.04,clip=3.,warmup_updates=10,
        lr_schedule='linear warmup10 then fixed5e-4; no low-LR tail within this bounded diagnostic',
        task_schedule='G3 candidate smooth A/B weights at u=step/200; A replay0.25 late',
        training_masks='A resampled per member/logical update; B round robin; composite A excludes B target. Historical run used fixed A; see frozen source.',
        validation='Four existing geographic-isolation VAL members; fixed A draw0, all B target bands',
        train_new_masks='Same TRAIN members, frozen independent A draw1; measures mask generalization separately',
        train_observations=4,val_observations=4,selection='absolute minimum mean fixedVAL(A,B)',
        early_stop='Only finite-value/gradient/geometry/leakage failures within200; no premature Ridge stop',
        resume_test_update=50,checkpoint_interval=25,peak_gpu_cap_GiB=8,automatic_continuation=False,
        data_manifest_sha256=read(OUT/'release.json')['manifest_sha256'],G3_acceptance_sha256=digest(G3/'acceptance.json'),
        semantic_labels_used=False,formal_pretraining=False,viewer_attached_before_first_update=True)
    config['initialization_sha256']=digest(BASE/'G2_arch/initialization_untrained.pt')
    save(G4/'config.json',config)
    names=['g4_fit.py','encoder_v2_controls.py','encoder_v2.py','encoder_v2_tasks.py','encoder_v2_runtime.py','model.py','expanded_dataset.py']
    sources={}
    for name in names:
        p=ROOT/'observation_model'/name;t=G4/'source'/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t);sources[name]=digest(p)
    save(G4/'source_manifest.json',sources);save(G4/'members.json',members)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4);torch.cuda.set_device(device)
    model=ObservationEncoderV2().to(device)
    model.load_state_dict(torch.load(BASE/'G2_arch/initialization_untrained.pt',map_location='cpu',weights_only=True)['model'])
    assert state_hash(model)==read(G3/'dry_run.json')['model_hash_after']
    statistics=read(OUT/'statistics.json')['bands'];data=load_members(members,device);train=[r for r in data if r[0]['role']=='TRAIN']
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.04)
    run=V2Run('encoder_v2_G4_fit200','G4有界学习诊断｜4训练+4留出｜最多200更新',config)
    state=dict(update=0,sampler_cursor=0,exposure=dict(observations=0,views=0,targets=0,pixels=0),
        scheduler=dict(progress=0.,weights=curriculum(0)),early_stop=dict(best=None,waiting=0),resume_count=0)
    started=time.perf_counter();reason='MAX_200_DIAGNOSTIC_UPDATES';best=float('inf')
    try:
        save(G4/'status.json',dict(status='RUNNING_BASELINES',optimizer_updates=0,max_updates=200,viewer_url='http://127.0.0.1:8766/'))
        run.heartbeat(0);fits=fit_ridge(data);torch.save(fits,G4/'ridge_TRAIN_only.pt')
        reference=controls(data,statistics,fits);save(G4/'baselines.json',reference)
        run.log('baseline',reference['VAL'],0)
        initial=evaluate(model,data,statistics,run,0)
        save(G4/'initial_evaluation.json',initial)
        best=(initial['VAL']['A']['score']+initial['VAL']['B']['score'])/2
        state['early_stop']['best']=best
        checkpoint(G4/'best.pt',model,optimizer,state,run,device)
        checkpoint(G4/'last.pt',model,optimizer,state,run,device)
        for step in range(1,201):
            tick=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            lambda_A,lambda_B=curriculum(step/200);tasks=[]
            for index,(meta,image) in enumerate(train):
                key=image.bands[(step-1+index)%len(image.bands)].band_key
                tasks.append((meta,training_spatial_task(image,meta['id'],step,excluded=(key,) if lambda_B>0 else ())))
                if lambda_B>0:tasks.append((meta,band_task(image,key)))
            weights=task_weights(tasks);logged=defaultdict(list);norm_vectors={};prev=gradient_vector(model)
            batch_views=batch_targets=batch_pixels=batch_visible=batch_native=batch_tokens=0
            for meta,task in tasks:
                run.heartbeat(step-1)
                rows,exposure,_=run_task(model,task,meta,statistics)
                coefficient=lambda_A if task.kind=='A' else lambda_B
                contribution=sum(r['loss']*weights[task.kind,r['sensor'],r['band']] for r in rows)*coefficient
                contribution.backward()
                after=gradient_vector(model);increment=after-prev;prev=after
                norm_vectors[task.kind]=norm_vectors.get(task.kind,torch.zeros_like(increment))+increment
                for r in rows:r['loss']=r['loss'].detach()
                logged[task.kind]+=rows
                batch_views+=1;batch_targets+=len(rows);batch_pixels+=exposure['target_pixels']
                batch_visible+=exposure['visible_source_pixels'];batch_native+=exposure['native_input_pixels'];batch_tokens+=exposure['tokens']
                append(G4/'exposure.jsonl',dict(planned_update=step,member=meta['id'],task=task.kind,
                    target_bands=[r['band'] for r in rows],exposure=exposure,mask=task.meta))
                del contribution,rows
            lr=config['lr']*min(step/config['warmup_updates'],1.)
            for group in optimizer.param_groups:group['lr']=lr
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True)
            optimizer.step();state['update']=step;state['sampler_cursor']+=len(train)
            state['scheduler']=dict(progress=step/200,weights=[lambda_A,lambda_B],lr=lr)
            for key,value in dict(observations=len(train),views=batch_views,targets=batch_targets,pixels=batch_pixels).items():state['exposure'][key]+=value
            torch.cuda.synchronize();duration=time.perf_counter()-tick;peak=torch.cuda.max_memory_allocated()
            if peak>8*1024**3:raise RuntimeError('GPU peak exceeded preregistered8GiB')
            for kind,rows in logged.items():run.log('train_'+kind,aggregate_metrics(rows),step)
            run.log('gradients',{kind+'_norm':float(value.norm()) for kind,value in norm_vectors.items()},step)
            run.log('state',dict(optimizer_updates=step,learning_rate=lr,lambda_A=lambda_A,lambda_B=lambda_B,
                remaining_updates=200-step,resume_count=state['resume_count'],gradient_norm=float(gradient)),step)
            run.log('capacity',dict(source_observations=state['exposure']['observations'],encoder_views=state['exposure']['views'],
                target_pixels=state['exposure']['pixels'],visible_pixels=batch_visible,native_pixels=batch_native,tokens=batch_tokens,
                peak_gpu_MiB=peak/1024**2,forward_backward_seconds=duration,source_observations_per_second=len(train)/duration),step)
            run.heartbeat(step)
            save(G4/'status.json',dict(status='RUNNING',optimizer_updates=step,max_updates=200,exposure=state['exposure'],
                last_update_utc=datetime.now(timezone.utc).isoformat(),lr=lr,lambda_A=lambda_A,lambda_B=lambda_B,
                viewer_url='http://127.0.0.1:8766/',run_id=run.run_id))
            print(json.dumps(dict(step=step,A=float(balanced_loss(logged['A'])),B=float(balanced_loss(logged['B'])) if logged['B'] else None,
                lr=lr,seconds=duration)),flush=True)
            if step%25==0:
                evaluation=evaluate(model,data,statistics,run,step)
                score=(evaluation['VAL']['A']['score']+evaluation['VAL']['B']['score'])/2
                if score<best:
                    best=score;state['early_stop']=dict(best=best,waiting=0);checkpoint(G4/'best.pt',model,optimizer,state,run,device)
                else:state['early_stop']['waiting']+=1
                checkpoint(G4/f'step_{step:03d}.pt',model,optimizer,state,run,device);checkpoint(G4/'last.pt',model,optimizer,state,run,device)
            if step==50:
                before=state_hash(model);old_session=run.session_id
                run.finish('cancelled','Planned full-state disk restore at update50; total budget remains200')
                model=ObservationEncoderV2().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.04)
                payload=restore(G4/'last.pt',model,optimizer,device);state=payload['state']
                current=optimizer.state_dict()['state'];saved=payload['optimizer']['state']
                assert current and all(torch.equal(current[k][n].cpu(),saved[k][n].cpu()) for k in saved for n in saved[k] if torch.is_tensor(saved[k][n]))
                assert before==state_hash(model) and state['update']==50
                state['resume_count']+=1
                run=V2Run('encoder_v2_G4_fit200','G4有界学习诊断｜4训练+4留出｜最多200更新',config,resume=True,sequence=payload['viewer_sequence'])
                save(G4/'resume_acceptance.json',dict(status='PASS',restored_update=50,nonempty_Adam_exact=True,
                    model_hash_before=before,model_hash_after=state_hash(model),RNG_sampler_exposure_scheduler=True,
                    old_session=old_session,new_session=run.session_id))
        run.finish('completed','Bounded G4 diagnostic finished; no G5 or cross-source continuation')
        save(G4/'status.json',dict(status='COMPLETED_PENDING_RESEARCH_REVIEW',optimizer_updates=200,reason=reason,
            exposure=state['exposure'],wall_seconds=time.perf_counter()-started,best_selection_score=best,
            convergence='Not established by a200-update diagnostic',automatic_continuation=False))
    except Exception as exc:
        try:run.finish('failed',repr(exc))
        except Exception:pass
        save(G4/'status.json',dict(status='FAILED_STOP',optimizer_updates=state['update'],error=repr(exc),
            exposure=state['exposure'],wall_seconds=time.perf_counter()-started,automatic_continuation=False))
        raise


if __name__=='__main__':main()

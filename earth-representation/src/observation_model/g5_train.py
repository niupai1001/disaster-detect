"""One gated G5 run; total successful updates <=20000, no historical queue."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from .encoder_v2 import EncoderConfig,ObservationEncoderV2
from .encoder_v2_tasks import (training_spatial_task,training_spectral_keys,spectral_task,validation_spatial_task,
    band_task,held_out_combinations,run_task,aggregate_metrics,balanced_loss)
from .g5_protocol import G5,prepare,ObservationStream,curriculum_g5,statistics
from .encoder_v2_runtime import BASE,V2Run,checkpoint,restore
from .g4_fit import task_weights,append
from .check_encoder_v2 import state_hash
from .expansion_data import ROOT,OUT,read,save,digest


def planned_batch(stream,update):
    plans=[];tasks=[]
    for index,anchor in stream.draw(update):
        row=stream.train.rows[index]
        dummy=SimpleNamespace(bands=[SimpleNamespace(band_key=b['band_key']) for b in row['bands']])
        hidden=training_spectral_keys(dummy,row['id'],update,anchor)
        plans.append(dict(index=index,member=row['id'],hidden=hidden,anchor=anchor))
        for kind,keys in [('A',[b['band_key'] for b in row['bands'] if b['band_key'] not in hidden]),('B',hidden)]:
            tasks.append((row,SimpleNamespace(kind=kind,targets=[SimpleNamespace(band_key=k) for k in keys])))
    return plans,task_weights(tasks)


def batch_backward(model,stream,update,config,log=None,benchmark=False):
    plans,weights=planned_batch(stream,update);rows_by_kind=defaultdict(list);a,b=curriculum_g5(update)
    model.train();model.zero_grad(set_to_none=True);exposure=Counter();loss_total=0.
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for plan in plans:
        row,image=stream.image(plan['index'])
        tasks=[training_spatial_task(image,row['id'],update,excluded=plan['hidden']),spectral_task(image,plan['hidden'])]
        for task in tasks:
            if log:log.heartbeat(update-1)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                rows,resource,_=run_task(model,task,row,statistics(),target_sample_pixels=config['train_target_samples_per_band'],logical_update=update)
                loss=sum(r['loss']*weights[task.kind,r['sensor'],r['band']] for r in rows)*(a if task.kind=='A' else b)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite G5 objective')
            loss.backward();loss_total+=float(loss.detach())
            for r in rows:r['loss']=r['loss'].detach()
            rows_by_kind[task.kind]+=rows
            exposure['views']+=1;exposure['target_band_observations']+=len(rows);exposure['supervised_pixels']+=resource['target_pixels']
            exposure['declared_valid_target_pixels']+=resource['full_valid_target_pixels'];exposure['native_local_vectors']+=resource['local_feature_vectors']
            exposure['encoder_feature_elements']+=resource['encoder_feature_elements']
            exposure['geo_tokens']+=resource['tokens']
            exposure['capped_geo_views']+=int(resource['token_budget_capped'])
            if not benchmark:
                append(G5/'exposure.jsonl',dict(update=update,member=row['id'],instrument=row['instrument'],task=task.kind,
                    target_bands=[r['band'] for r in rows],mask_seed=task.meta.get('seed'),resource=resource))
            del rows,loss
        exposure['source_observations']+=1
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    if peak>config['peak_memory_cap_bytes']:raise RuntimeError('G5 preregistered peak-memory cap exceeded')
    return rows_by_kind,dict(exposure),dict(seconds=seconds,peak_bytes=peak,weighted_loss=loss_total),plans


def benchmark():
    config=prepare();torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.set_device(config['device'])
    stream=ObservationStream(config['device']);model=ObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(config['device'])
    before=state_hash(model);rng=stream.rng.bit_generator.state
    rows,exposure,timing,plans=batch_backward(model,stream,1,config,benchmark=True)
    assert exposure['source_observations']==32 and exposure['views']==64 and before==state_hash(model)
    model.zero_grad(set_to_none=True);stream.rng.bit_generator.state=rng
    # Largest admitted window, not a crop or a convenient small smoke member.
    largest=max(stream.indices,key=lambda i:sum(np.prod(b['shape']) for b in stream.train.rows[i]['bands']))
    row,image=stream.image(largest);tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for task in [training_spatial_task(image,row['id'],1,excluded=['B11']),spectral_task(image,['B11'])]:
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
            r,resource,_=run_task(model,task,row,statistics(),target_sample_pixels=2048,logical_update=1);loss=balanced_loss(r)
        loss.backward();model.zero_grad(set_to_none=True)
    torch.cuda.synchronize();maximum=dict(member=row['id'],native_pixels=sum(int(np.prod(b['shape'])) for b in row['bands']),
        seconds=time.perf_counter()-tick,peak_bytes=torch.cuda.max_memory_allocated())
    assert maximum['peak_bytes']<=config['peak_memory_cap_bytes'] and before==state_hash(model)
    save(G5/'throughput_preflight.json',dict(status='PASS',effective_batch=32,full_native_inputs=True,
        full_batch=timing,exposure=exposure,selected_members=[p['member'] for p in plans],largest_window=maximum,
        model_hash_before=before,model_hash_after=state_hash(model),optimizer_updates=0,
        approximate_training_hours_at_cap=timing['seconds']*20000/3600,
        caveat='One actual update-shaped forward/backward; excludes validation/checkpoint I/O and distribution variability'))
    torch.save(dict(model=model.state_dict(),config=config,initialization='seed0_random_native_bank',optimizer_updates=0),G5/'initialization_untrained.pt')
    print('G5_THROUGHPUT_PASS',timing,maximum,flush=True)


@torch.no_grad()
def evaluation(model,stream,config,run,step,challenge=False):
    model.eval();groups=defaultdict(list);started=time.perf_counter();count=0
    cohorts=[('primary',stream.primary)]+([('unseen_response',stream.response)] if challenge else [])
    stats=statistics()
    for cohort,indices in cohorts:
        for index in indices:
            row,image=stream.image(index,'VAL');run.heartbeat(step)
            a=validation_spatial_task(image,row['id'])
            jobs=[('A',a)]+[('B',band_task(image,b.band_key)) for b in image.bands]
            if challenge and cohort=='primary':jobs += [('combo_'+name,spectral_task(image,keys)) for name,keys in held_out_combinations(image)]
            for kind,task in jobs:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                    rows,resource,_=run_task(model,task,row,stats,target_sample_pixels=config['validation_target_samples_per_band'],logical_update=0)
                groups[cohort,kind]+=rows;count+=1
                append(G5/'evaluation_exposure.jsonl',dict(update=step,cohort=cohort,task=kind,member=row['id'],resource=resource))
                if count%20==0:
                    progress=read(G5/'status.json')
                    progress.update(status='VALIDATING',optimizer_updates=step,maximum_updates=20000,
                        pid=os.getpid(),validation_views=count,cohort=cohort,
                        last_progress_utc=datetime.now(timezone.utc).isoformat())
                    save(G5/'status.json',progress)
    result={}
    for (cohort,kind),rows in groups.items():
        metrics=aggregate_metrics(rows);result[cohort+'/'+kind]=metrics
        phase=('val_'+kind) if cohort=='primary' and kind in ['A','B'] else ('unseen_response_'+kind if cohort!='primary' else kind)
        run.log(phase,metrics,step)
    score=(.25*result['primary/A']['score']+result['primary/B']['score'])/1.25
    append(G5/'evaluation_history.jsonl',dict(update=step,score=score,metrics=result,encoder_views=count,seconds=time.perf_counter()-started,
        challenge_not_used_for_selection=True))
    if challenge:
        from .g5_cases import publish_cases
        publish_cases(model,stream,config,run,step)
    return score,result


def launch(resume=False):
    config=read(G5/'config.json');assert read(G5/'approval_state.json')['technical_gate']=='PASS'
    for relative,expected in read(G5/'source_manifest.json')['sha256'].items():
        assert digest(ROOT/relative)==expected, f'Frozen G5 source/config changed: {relative}'
    assert config['maximum_updates']==20000 and digest(ROOT/'observation_model/execution_policy.json')==config['old_execution_policy_sha256']
    if (G5/'status.json').exists() and not resume:raise FileExistsError('Existing G5 run; resume only exact checkpoint')
    lock=G5/'worker.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    device=config['device'];random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4);torch.cuda.set_device(device)
    stream=ObservationStream(device);model=ObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(device)
    model.load_state_dict(torch.load(G5/'initialization_untrained.pt',map_location='cpu',weights_only=True)['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    state=dict(update=0,exposure={},sampler_cursor=0,sampler_rng=stream.rng.bit_generator.state,
        lr_base=config['lr'],best_score=None,significant_best=None,lr_wait=0,stop_wait=0,
        scheduler=dict(progress=0),early_stop=dict(best=None,waiting=0),resume_count=0,wall_seconds=0.,rollback_updates=0)
    sequence=0
    if resume:
        payload=restore(G5/'last.pt',model,optimizer,device);state=payload['state'];stream.rng.bit_generator.state=state['sampler_rng'];sequence=payload['viewer_sequence'];state['resume_count']+=1
        # Count successful but uncheckpointed updates against the same user budget.
        prior=read(G5/'status.json')
        state['rollback_updates']=max(state.get('rollback_updates',0),prior.get('rollback_updates',0))+max(0,prior.get('optimizer_updates',0)-state['update'])
    run=V2Run(config['run_id'],'共享物理Encoder｜原生局部特征｜6402训练｜20k上限',config,resume=resume,sequence=sequence)
    for filename,kind in [('task_card.html','html'),('ridge_audit.png','image'),('architecture.png','image')]:
        source=G5/filename
        if source.exists():
            target=run.dir/'artifacts'/filename;target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
            run.artifact('artifacts/'+filename,kind=kind,tag='G5-preregistered-contract',step=state['update'])
    started=time.perf_counter();reason='MAX_UPDATES_UNCONFIRMED_CONVERGENCE'
    try:
        save(G5/'status.json',dict(status='RUNNING_INITIAL_VALIDATION' if not resume else 'RESUMED',optimizer_updates=state['update'],maximum_updates=20000,pid=os.getpid(),rollback_updates=state['rollback_updates']))
        if not resume:
            score,_=evaluation(model,stream,config,run,0,challenge=True)
            state['best_score']=state['significant_best']=score;state['early_stop']['best']=score
            checkpoint(G5/'best.pt',model,optimizer,state,run,device);checkpoint(G5/'last.pt',model,optimizer,state,run,device)
        for update in range(state['update']+1,20001-state['rollback_updates']):
            rows,exposure,timing,_=batch_backward(model,stream,update,config,run)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True)
            lr=max(config['minimum_lr'],state['lr_base']*min(1,update/config['warmup_updates']))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();state['update']=update;state['sampler_cursor']+=32;state['sampler_rng']=stream.rng.bit_generator.state
            state['scheduler']=dict(progress=update/20000,lr=lr,task_weights=curriculum_g5(update))
            for key,value in exposure.items():state['exposure'][key]=state['exposure'].get(key,0)+value
            a,b=curriculum_g5(update)
            for kind,r in rows.items():run.log('train_'+kind,aggregate_metrics(r),update)
            run.log('state',dict(optimizer_updates=update,remaining_updates=20000-update,learning_rate=lr,
                lambda_A=a,lambda_B=b,gradient_norm=float(grad),resume_count=state['resume_count']),update)
            run.log('capacity',dict(source_observations=state['exposure']['source_observations'],encoder_views=state['exposure']['views'],
                target_pixels=state['exposure']['supervised_pixels'],tokens=exposure['geo_tokens'],
                local_feature_vectors=exposure['native_local_vectors'],encoder_feature_elements=exposure['encoder_feature_elements'],
                peak_gpu_MiB=timing['peak_bytes']/1024**2,forward_backward_seconds=timing['seconds'],source_observations_per_second=32/timing['seconds']),update)
            run.heartbeat(update)
            save(G5/'status.json',dict(status='RUNNING',optimizer_updates=update,maximum_updates=20000,pid=os.getpid(),
                rollback_updates=state['rollback_updates'],actual_successful_updates=update+state['rollback_updates'],
                last_progress_utc=datetime.now(timezone.utc).isoformat(),learning_rate=lr,task_weights=[a,b],
                exposure=state['exposure'],last_batch_seconds=timing['seconds'],viewer_url='http://127.0.0.1:8766/'))
            print(json.dumps(dict(update=update,A=float(balanced_loss(rows['A'])),B=float(balanced_loss(rows['B'])),lr=lr,seconds=timing['seconds'])),flush=True)
            if update==10 or update%config['checkpoint_interval']==0:
                state['wall_seconds']+=time.perf_counter()-started;started=time.perf_counter()
                checkpoint(G5/'last.pt',model,optimizer,state,run,device)
            if update%config['validation_interval']==0:
                score,_=evaluation(model,stream,config,run,update,challenge=update%config['challenge_interval']==0)
                if score<state['best_score']:
                    state['best_score']=score;state['early_stop']['best']=score;checkpoint(G5/'best.pt',model,optimizer,state,run,device)
                if score<state['significant_best']*(1-.001):
                    state['significant_best']=score;state['lr_wait']=state['stop_wait']=0
                else:state['lr_wait']+=1;state['stop_wait']+=1
                if update>=2000 and state['lr_wait']>=4:
                    state['lr_base']=max(config['minimum_lr'],state['lr_base']*.5);state['lr_wait']=0
                checkpoint(G5/'last.pt',model,optimizer,state,run,device)
                if update>=2000 and state['stop_wait']>=12 and state['lr_base']<=config['lr']/16:
                    reason='EARLY_STOP_FIXED_PRIMARY_PLATEAU';break
        state['wall_seconds']+=time.perf_counter()-started
        checkpoint(G5/'last.pt',model,optimizer,state,run,device)
        run.finish('completed','G5 budget/early-stop reached; post-training frozen probe and review pending')
        save(G5/'status.json',dict(status='COMPLETED_PENDING_FROZEN_PROBE',optimizer_updates=state['update'],reason=reason,
            rollback_updates=state['rollback_updates'],actual_successful_updates=state['update']+state['rollback_updates'],
            best_score=state['best_score'],exposure=state['exposure'],wall_seconds=state['wall_seconds'],automatic_extension=False))
    except Exception as exc:
        try:run.finish('failed',repr(exc))
        except Exception:pass
        save(G5/'status.json',dict(status='FAILED_STOP',optimizer_updates=state['update'],error=repr(exc),pid=os.getpid(),
            rollback_updates=state['rollback_updates'],actual_successful_updates=state['update']+state['rollback_updates'],
            last_checkpoint=str(G5/'last.pt'),exposure=state['exposure'],automatic_extension=False));raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.benchmark:benchmark()
    elif args.run or args.resume:launch(args.resume)
    else:print(prepare())


if __name__=='__main__':main()

"""Real epoch/batch training for the shared-memory MAE, with checkpoint-bound TVLP."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
import time
import uuid
import numpy as np
import torch
from encoder_v3.latent_response.checkpoint import digest, snapshot, restore, atomic_replace, parameter_sha, validation_transaction
from encoder_v3.latent_response.run_control import append_event, reconcile_logs, lr_multiplier
from encoder_v3.latent_response.telemetry import primary_process, training_runs_root
from encoder_v3.latent_response.tvlp_writer import RunWriter
from .data import ROOT, load_data, identity, sample_from_dataset, EpochPlan, validate_config, build_model


class StopRequested(Exception):pass


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    atomic_replace(temporary,path)


def save(path, model, optimizer, scheduler, state, binding):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema='shared_mae_checkpoint_v1',identity=binding,**snapshot(model,optimizer,state,scheduler))
    fd, temporary=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        torch.save(payload,temporary);atomic_replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.remove(temporary)


def resume(path, model, optimizer, scheduler, state, binding):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!='shared_mae_checkpoint_v1' or payload.get('identity')!=binding:
        raise ValueError('Shared-MAE checkpoint identity mismatch; legacy LR weights are not a resume')
    live=snapshot(model,optimizer,state,scheduler)
    try:restore(payload,model,optimizer,state,scheduler)
    except Exception:
        restore(live,model,optimizer,state,scheduler);raise


def writer_for(config,binding,state,run_dir,resuming=False):
    if not primary_process():return nullcontext(None)
    root=training_runs_root().resolve();run_dir=Path(run_dir).resolve()
    if state.get('viewer'):
        reference=state['viewer'];path=Path(reference['directory']).resolve()
        if not resuming or not path.is_relative_to(root):raise ValueError('Actual checkpoint load required to resume TVLP')
        meta=json.loads((path/'run.json').read_text())
        if meta['run_id']!=reference['run_id'] or meta['config']['execution_identity']!=digest(binding) or meta['config']['training_directory']!=str(run_dir):
            raise ValueError('Viewer identity mismatch')
        return RunWriter(path,resume=True)
    metrics={name:dict(label=label,unit=unit,direction=direction,role=role) for name,label,unit,direction,role in (
        ('train_loss','训练批次平均重建损失','z²','min','loss'),('val_loss','完整验证集平均重建损失','z²','min','loss'),
        ('epoch_train_loss','完整Epoch平均训练损失','z²','min','loss'),('learning_rate','学习率','','none','other'),
        ('gradient_norm','裁剪前梯度范数','L2','none','other'),('update_seconds','更新耗时','s','min','other'),
        ('observations_seen','累计观测数','个','none','other'),('batch_size','本批观测数','个','none','other'),
        ('val_s2_loss','S2验证损失','z²','min','loss'),('val_landsat_loss','Landsat验证损失','z²','min','loss'))}
    viewer=config['viewer'];variant=dict(viewer['variant'])
    variant['id']+=f"-L{config['patch_metres']:g}"
    variant['name']+=f" · L={config['patch_metres']:g}m"
    directory=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex
    writer=RunWriter(root/directory,experiment=viewer['experiment'],variant=variant,metrics=metrics,
        charts=['train_loss','val_loss'],primary_metric='val_loss',numbers=['learning_rate','gradient_norm','val_s2_loss','val_landsat_loss'],
        config=dict(execution_identity=digest(binding),training_directory=str(run_dir),
            epochs=config['epochs'],batch_size=config['batch_size'],micro_batch_observations=config['micro_batch_observations'],
            patch_metres=config['patch_metres'],objective='Per valid channel mean, then equal available A/B1 tasks, then equal observations',
            validation='Full frozen VAL with fixed masks; no independent EnMAP VAL',
            architecture='Shared physical KV, learned metric patch embedding, visible-only global Transformer, masked-token global decoder'))
    state['viewer']=dict(directory=str(writer.path),run_id=writer.meta['run_id'])
    return writer


def micro_batches(rows,config):
    group=[];pixels=0
    for row in rows:
        size=sum(b['shape'][0]*b['shape'][1] for b in row['bands'])
        if group and (len(group)>=config['micro_batch_observations'] or pixels+size>config['micro_batch_pixels']):
            yield group;group=[];pixels=0
        group.append(row);pixels+=size
    if group:yield group


def batch_loss(model,dataset,rows,resolver,config,*,epoch=0,backward=False,stop_requested=None,lookup=None):
    lookup=lookup or {r['id']:i for i,r in enumerate(dataset.rows)}
    records=[]
    for group in micro_batches(rows,config):
        if stop_requested and stop_requested():raise StopRequested()
        samples=[sample_from_dataset(dataset,lookup[r['id']],resolver,config['device']) for r in group]
        kinds=[model.available_tasks(s) for s in samples]
        values=[0.]*len(samples);details=[{} for _ in samples]
        for kind in ('A','B1'):
            selected=[i for i,task_list in enumerate(kinds) if kind in task_list]
            if not selected:continue
            precision=config.get('precision','float32')
            if precision not in ('float32','bfloat16'):raise ValueError('Unsupported precision')
            if precision=='bfloat16' and not str(config['device']).startswith('cuda'):
                raise ValueError('BF16 release requires validated CUDA execution')
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bfloat16'):
                losses,census=model([samples[i] for i in selected],kind=kind,seed=config['seed'],epoch=epoch)
            weights=losses.new_tensor([1./len(kinds[i])/len(rows) for i in selected])
            objective=(losses*weights).sum()
            if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite objective')
            if backward:objective.backward()
            for i,value,info in zip(selected,losses.detach().cpu().tolist(),census):
                values[i]+=value/len(kinds[i]);details[i][kind]={k:v for k,v in info.items() if k not in ('predictions','target_masks')}
        for row,value,task_info in zip(group,values,details):records.append(dict(id=row['id'],sensor=row['sensor'],loss=value,tasks=task_info))
    return sum(r['loss'] for r in records)/len(records),records


def update(model,dataset,rows,resolver,optimizer,scheduler,state,config,stop_requested=None):
    if any(r['role']!='TRAIN' for r in rows):raise ValueError('Held-out row in training batch')
    optimizer.zero_grad(set_to_none=True)
    loss,records=batch_loss(model,dataset,rows,resolver,config,epoch=state['epoch_index'],backward=True,stop_requested=stop_requested)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
    if stop_requested and stop_requested():raise StopRequested()
    sequence=state.get('spent_updates',state['cursor'])+1
    journal=Path(config['run_dir'])/'optimizer_attempts.jsonl' if config.get('run_dir') else None
    if journal:append_event(journal,dict(event='intent',sequence=sequence,cursor=state['cursor']+1))
    state['spent_updates']=sequence;lr=optimizer.param_groups[0]['lr'];optimizer.step()
    if journal:append_event(journal,dict(event='applied',sequence=sequence,cursor=state['cursor']+1))
    scheduler.step();state['cursor']+=1
    return dict(cursor=state['cursor'],loss=loss,lr=lr,gradient_norm=float(norm),batch_size=len(rows),members=records)


def validate(model,dataset,resolver,optimizer,scheduler,state,config,rows=None):
    panel=dataset.rows if rows is None else rows
    if any(r['role']!='VAL' for r in panel):raise ValueError('Validation requires frozen VAL members')
    fixed=dict(config,seed=config['validation_seed'])
    with validation_transaction(model,optimizer,state,scheduler),torch.no_grad():
        model.eval();metric,records=batch_loss(model,dataset,panel,resolver,fixed,epoch=0)
    source={}
    for record in records:
        name='Landsat' if record['sensor'] in ('L8','L9') else record['sensor']
        source.setdefault(name,[]).append(record['loss'])
    result=dict(metric=metric,observations=len(records),per_source={k:sum(v)/len(v) for k,v in source.items()})
    state['validation']=result
    return result


def main():
    if int(os.environ.get('WORLD_SIZE','1'))>1 or (torch.distributed.is_initialized() and torch.distributed.get_world_size()>1):
        raise RuntimeError('Distributed synchronization is not implemented')
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--readiness',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text());validate_config(config)
    approval=json.loads(args.authorization.read_text());gate=json.loads(args.readiness.read_text())
    torch.set_num_threads(2);torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datasets,resolver=load_data();binding=identity(config,datasets,resolver)
    if approval.get('source')!='user_in_current_chat' or approval.get('identity')!=binding or approval.get('suspended'):
        raise PermissionError('Authorization must bind this exact new architecture/configuration/data')
    if gate.get('identity')!=binding or gate.get('training_release_ready') is not True:
        raise ValueError('Matching shared-MAE engineering and real-update stability gate required')
    if args.run_dir.exists() and not args.resume:raise ValueError('New run requires a new directory')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    write(args.run_dir/'identity.json',binding);write(args.run_dir/'config.json',config)
    model=build_model(config).to(config['device']);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    plan=EpochPlan(datasets['TRAIN'].rows,config['seed'],config['batch_size'])
    total=config['epochs']*len(plan);warmup=min(total-1,round(config['warmup_epochs']*len(plan)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_multiplier(n,total,warmup))
    state=dict(cursor=0,spent_updates=0,epoch_index=0,status='RUNNING',best=None,epoch_loss_sum=0.,epoch_observations=0)
    if args.resume:
        resume(args.resume,model,optimizer,scheduler,state,binding);reconcile_logs(args.run_dir,state);state['status']='RUNNING'
    stop=threading.Event();signal.signal(signal.SIGINT,lambda *_:stop.set());signal.signal(signal.SIGTERM,lambda *_:stop.set())
    requested=lambda:stop.is_set() or (args.run_dir/'STOP_REQUESTED').exists()
    runtime=dict(config,run_dir=str(args.run_dir))
    try:
        with writer_for(config,binding,state,args.run_dir,bool(args.resume)) as writer:
            if state['cursor']==0 and 'validation' not in state:
                initial=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,config)
                state['initial_validation']=initial;state['best']=initial['metric'];state['validation_regressions']=0
                if writer:writer.log(step=0,epoch=0.,values=dict(val_loss=initial['metric']))
                save(args.run_dir/'best.pt',model,optimizer,scheduler,state,binding)
                save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding)
            while state['cursor']<total:
                state['epoch_index']=state['cursor']//len(plan);rows=plan.batch(state['cursor']);start=time.perf_counter()
                record=update(model,datasets['TRAIN'],rows,resolver,optimizer,scheduler,state,runtime,requested)
                record['seconds']=time.perf_counter()-start;epoch=plan.progress(state['cursor'])
                state.update(epoch=epoch,observations_seen=plan.observations_seen(state['cursor']))
                state['epoch_loss_sum']+=record['loss']*len(rows);state['epoch_observations']+=len(rows)
                values=dict(train_loss=record['loss'],learning_rate=record['lr'],gradient_norm=record['gradient_norm'],
                            update_seconds=record['seconds'],batch_size=len(rows),observations_seen=state['observations_seen'])
                if state['cursor']%len(plan)==0:
                    values['epoch_train_loss']=state['epoch_loss_sum']/state['epoch_observations'];state['last_epoch_loss']=values['epoch_train_loss']
                    state['epoch_loss_sum']=0.;state['epoch_observations']=0
                append_event(args.run_dir/'updates.jsonl',dict(record,epoch=epoch))
                if writer:writer.log(step=state['cursor'],epoch=epoch,values=values)
                due=plan.validation_due(state['cursor'],config['validations_per_epoch'])
                if due:
                    result=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,config)
                    worse=state['best'] is not None and result['metric']>state['best']*1.1
                    state['validation_regressions']=state.get('validation_regressions',0)+1 if worse else 0
                    if writer:writer.log(step=state['cursor'],epoch=epoch,values=dict(val_loss=result['metric'],
                        **{('val_s2_loss' if k=='S2' else 'val_landsat_loss'):v for k,v in result['per_source'].items() if k in ('S2','Landsat')}))
                    if state['best'] is None or result['metric']<state['best']:
                        state['best']=result['metric'];save(args.run_dir/'best.pt',model,optimizer,scheduler,state,binding)
                if due or state['cursor']%config['checkpoint_every_batches']==0:save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding)
                write(args.run_dir/'status.json',state)
                print('epoch',epoch,'batch',state['cursor'],'loss',record['loss'],flush=True)
                if state.get('validation_regressions',0)>=2:
                    state['stop_reason']='Two consecutive full validations exceed best loss by 10%'
                    raise StopRequested()
            state['status']='COMPLETE';save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding)
            write(args.run_dir/'status.json',state)
    except StopRequested:
        state['status']='STOPPED';save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding);write(args.run_dir/'status.json',state)
    except Exception as exc:
        write(args.run_dir/'status.json',dict(state,status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()

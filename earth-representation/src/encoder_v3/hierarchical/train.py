"""Observation-equal V3-H training, explicit task weights, EMA and bound recovery."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import signal
import time
import uuid
import math
import numpy as np
import torch
from encoder_v3.shared_mae.data import load_data,sample_from_dataset,EpochPlan,identity as base_identity,ROOT
from encoder_v3.shared_mae.train import write,micro_batches,StopRequested
from encoder_v3.latent_response.checkpoint import snapshot,restore,atomic_replace,digest,validation_transaction
from encoder_v3.latent_response.run_control import append_event,lr_multiplier,reconcile_logs
from encoder_v3.latent_response.tvlp_writer import RunWriter
from .model import HierarchicalMAE


def build_model(config):
    validate_config(config)
    return HierarchicalMAE(**{k:config[k] for k in ('patch_metres','mask_ratio','encoder_depth','lambda_jepa','ema_momentum')})


def validate_config(config):
    if config.get('architecture')!='hierarchical_v3_h' or config.get('objective')!='observation_equal':
        raise ValueError('Explicit V3-H observation-equal configuration required')
    if config.get('scale_policy')!='finest_loaded_native_gsd_xy_times_2_and_4_nested_at_L':
        raise ValueError('Unknown dynamic scale policy')
    for key in ('epochs','batch_size','micro_batch_observations','micro_batch_pixels','encoder_depth','validations_per_epoch','checkpoint_every_batches'):
        if type(config.get(key)) is not int or config[key]<1:raise ValueError('Positive integer '+key+' required')
    if not 0<=config['warmup_epochs']<config['epochs']:raise ValueError('Invalid warmup')
    if set(config['task_weights'])!={'A','B1'} or any(not math.isfinite(v) or v<=0 for v in config['task_weights'].values()):
        raise ValueError('Positive finite A/B1 weights required')
    if config['precision'] not in ('float32','bfloat16'):raise ValueError('Invalid precision')
    if config.get('deterministic') is not True:raise ValueError('Deterministic V3-H execution required')


def identity(config,datasets,resolver):
    binding=base_identity(config,datasets,resolver)
    from encoder_v3.data import sha256
    files={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')}
    entry=ROOT/'scripts/run_v3_h.py';files[str(entry.relative_to(ROOT))]=sha256(entry)
    binding['code_files'].update(files)
    binding['code']=digest(binding['code_files'])
    binding['architecture']='hierarchical_v3_h'
    return binding


def save(path,model,optimizer,scheduler,state,binding):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema='v3_h_checkpoint_v1',identity=binding,**snapshot(model,optimizer,state,scheduler))
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        torch.save(payload,tmp);atomic_replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.remove(tmp)


def resume(path,model,optimizer,scheduler,state,binding):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!='v3_h_checkpoint_v1' or payload.get('identity')!=binding:
        raise ValueError('V3-H checkpoint identity mismatch')
    live=snapshot(model,optimizer,state,scheduler)
    try:restore(payload,model,optimizer,state,scheduler)
    except Exception:
        restore(live,model,optimizer,state,scheduler);raise


def batch_loss(model,dataset,rows,resolver,config,*,epoch=0,backward=False,stop_requested=None):
    lookup={r['id']:i for i,r in enumerate(dataset.rows)}
    records=[]
    for group in micro_batches(rows,config):
        if stop_requested and stop_requested():raise StopRequested()
        samples=[sample_from_dataset(dataset,lookup[r['id']],resolver,config['device']) for r in group]
        for sample,row in zip(samples,group):
            kinds=model.available_tasks(sample)
            total_weight=sum(config['task_weights'][k] for k in kinds)
            value=0.;task_info={}
            for kind in kinds:
                weight=config['task_weights'][kind]/total_weight
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bfloat16'):
                    losses,details=model([sample],kind=kind,seed=config['seed'],epoch=epoch)
                    objective=losses.sum()*weight/len(rows)
                if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite objective')
                if backward:objective.backward()
                value+=float(losses.detach().sum())*weight
                task_info[kind]={k:v for k,v in details[0].items() if k not in ('predictions','target_masks','online_features')}
            records.append(dict(id=row['id'],sensor=row['sensor'],loss=value,tasks=task_info))
    return sum(r['loss'] for r in records)/len(records),records


def update(model,dataset,rows,resolver,optimizer,scheduler,state,config,stop_requested=None):
    if any(r['role']!='TRAIN' for r in rows):raise ValueError('Non-TRAIN row in optimizer batch')
    optimizer.zero_grad(set_to_none=True)
    value,records=batch_loss(model,dataset,rows,resolver,config,epoch=state['cursor']//config['batches_per_epoch'],backward=True,stop_requested=stop_requested)
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['clip_norm'],error_if_nonfinite=True)
    if stop_requested and stop_requested():raise StopRequested()
    lr=optimizer.param_groups[0]['lr']
    sequence=state.get('spent_updates',state['cursor'])+1
    journal=Path(config['run_dir'])/'optimizer_attempts.jsonl' if config.get('run_dir') else None
    if journal:append_event(journal,dict(event='intent',sequence=sequence,cursor=state['cursor']+1))
    state['spent_updates']=sequence
    optimizer.step()
    if journal:append_event(journal,dict(event='applied',sequence=sequence,cursor=state['cursor']+1))
    if model.lambda_jepa>0:model.update_ema()
    scheduler.step();state['cursor']+=1
    return dict(loss=value,members=records,gradient_norm=float(norm),lr=lr,batch_size=len(rows),cursor=state['cursor'])


def validate(model,dataset,resolver,optimizer,scheduler,state,config,rows=None):
    panel=dataset.rows if rows is None else rows
    if any(r['role']!='VAL' for r in panel):raise ValueError('Validation requires VAL role')
    with validation_transaction(model,optimizer,state,scheduler),torch.no_grad():
        model.eval()
        value,records=batch_loss(model,dataset,panel,resolver,dict(config,seed=config['validation_seed']),epoch=0)
    return dict(metric=value,observations=len(records),members=records)


def writer_for(config,binding,state,run_dir,resuming):
    if state.get('viewer'):
        if not resuming:raise ValueError('Real checkpoint load required to resume Viewer')
        meta=json.loads((Path(state['viewer']['directory'])/'run.json').read_text())
        if meta['run_id']!=state['viewer']['run_id'] or meta['config']['execution_identity']!=digest(binding):
            raise ValueError('Viewer identity mismatch')
        return RunWriter(state['viewer']['directory'],resume=True)
    names={'train_loss':'训练批次综合损失','val_loss':'完整验证综合损失','mae_loss':'原生重建MSE',
           'jepa_loss':'潜在预测损失','gradient_norm':'裁剪前梯度范数','learning_rate':'学习率','update_seconds':'批次耗时'}
    metrics={k:dict(label=v,unit='z²' if k=='mae_loss' else '',direction='min' if 'loss' in k else 'none',role='loss' if 'loss' in k else 'other') for k,v in names.items()}
    directory=Path(config['viewer_root'])/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex)
    writer=RunWriter(directory,experiment=dict(id='encoder-v3-h',name='Encoder V3-H · 层次化表征'),
        variant=dict(id='mae' if config['lambda_jepa']==0 else 'hybrid',name='层次MAE' if config['lambda_jepa']==0 else '层次MAE+JEPA',kind='modification',description='动态原生几何、并行三尺度、双向稀疏交互'),
        metrics=metrics,charts=['train_loss','val_loss'],primary_metric='val_loss',numbers=['mae_loss','jepa_loss','gradient_norm','learning_rate'],
        config=dict(execution_identity=digest(binding),training_directory=str(run_dir),architecture='hierarchical_v3_h',task_weights=config['task_weights'],lambda_jepa=config['lambda_jepa']))
    state['viewer']=dict(directory=str(writer.path),run_id=writer.meta['run_id'])
    return writer


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','authorization','readiness','run-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if config['architecture']!='hierarchical_v3_h' or config['objective']!='observation_equal':raise ValueError('V3-H config required')
    if config['epochs']<1 or config['batch_size']<1 or config['precision'] not in ('float32','bfloat16'):raise ValueError('Invalid execution config')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Distributed synchronization not implemented')
    torch.set_num_threads(2);torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datasets,resolver=load_data();binding=identity(config,datasets,resolver)
    approval=json.loads(args.authorization.read_text());gate=json.loads(args.readiness.read_text())
    if approval.get('source')!='user_in_current_chat' or approval.get('identity')!=binding or approval.get('suspended'):raise PermissionError('Authorization identity mismatch')
    if gate.get('identity')!=binding or not gate.get('training_release_ready'):raise ValueError('V3-H engineering gate missing')
    if args.run_dir.exists() and not args.resume:raise ValueError('New training requires a new run directory')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    write(args.run_dir/'identity.json',binding);write(args.run_dir/'config.json',config)
    model=build_model(config).to(config['device'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
    plan=EpochPlan(datasets['TRAIN'].rows,config['seed'],config['batch_size'])
    runtime=dict(config,batches_per_epoch=len(plan),run_dir=str(args.run_dir));total=len(plan)*config['epochs']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_multiplier(n,total,round(config['warmup_epochs']*len(plan))))
    state=dict(cursor=0,spent_updates=0,status='RUNNING',best=None)
    if args.resume:resume(args.resume,model,optimizer,scheduler,state,binding);reconcile_logs(args.run_dir,state)
    state['status']='RUNNING'
    stop=threading.Event();signal.signal(signal.SIGINT,lambda *_:stop.set());signal.signal(signal.SIGTERM,lambda *_:stop.set())
    requested=lambda:stop.is_set() or (args.run_dir/'STOP_REQUESTED').exists()
    try:
        with writer_for(config,binding,state,args.run_dir,bool(args.resume)) as writer:
            state['phase']='INITIAL_VALIDATION' if 'initial_validation' not in state else 'TRAINING'
            write(args.run_dir/'status.json',state)
            if 'initial_validation' not in state:
                result=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,runtime)
                state['initial_validation']=dict(metric=result['metric'],observations=result['observations'])
                state['best']=result['metric'];state['validation_regressions']=0
                write(args.run_dir/'validation_0.json',result)
                writer.log(step=0,epoch=0.,values=dict(val_loss=result['metric']))
                save(args.run_dir/'best.pt',model,optimizer,scheduler,state,binding)
                save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding)
            state['phase']='TRAINING'
            while state['cursor']<total:
                if requested():raise StopRequested()
                start=time.perf_counter();rows=plan.batch(state['cursor'])
                record=update(model,datasets['TRAIN'],rows,resolver,optimizer,scheduler,state,runtime,requested)
                record['seconds']=time.perf_counter()-start;record['epoch']=plan.progress(state['cursor'])
                append_event(args.run_dir/'updates.jsonl',record)
                def term_mean(key):
                    means=[]
                    for member in record['members']:
                        total=sum(config['task_weights'][k] for k in member['tasks'])
                        means.append(sum(config['task_weights'][k]*v[key] for k,v in member['tasks'].items())/total)
                    return sum(means)/len(means)
                writer.log(step=state['cursor'],epoch=record['epoch'],values=dict(train_loss=record['loss'],
                    mae_loss=term_mean('mae'),jepa_loss=term_mean('jepa'),
                    gradient_norm=record['gradient_norm'],learning_rate=record['lr'],update_seconds=record['seconds']))
                due=plan.validation_due(state['cursor'],config['validations_per_epoch'])
                if due:
                    result=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,runtime)
                    state['validation_regressions']=state.get('validation_regressions',0)+1 if result['metric']>state['best']*1.1 else 0
                    state['validation']=dict(metric=result['metric'],observations=result['observations'])
                    write(args.run_dir/f"validation_{state['cursor']}.json",result)
                    writer.log(step=state['cursor'],epoch=record['epoch'],values=dict(val_loss=result['metric']))
                    if result['metric']<state['best']:
                        state['best']=result['metric'];save(args.run_dir/'best.pt',model,optimizer,scheduler,state,binding)
                if due or state['cursor']%config['checkpoint_every_batches']==0:save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding)
                state['epoch']=record['epoch'];state['observations_seen']=plan.observations_seen(state['cursor'])
                write(args.run_dir/'status.json',state)
                print('epoch',record['epoch'],'batch',state['cursor'],'loss',record['loss'],flush=True)
                if state.get('validation_regressions',0)>=2:raise StopRequested()
            state['status']='COMPLETE'
            save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding);write(args.run_dir/'status.json',state)
    except StopRequested:
        state['status']='STOPPED';save(args.run_dir/'last.pt',model,optimizer,scheduler,state,binding);write(args.run_dir/'status.json',state)
    except Exception as exc:
        write(args.run_dir/'status.json',dict(state,status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()

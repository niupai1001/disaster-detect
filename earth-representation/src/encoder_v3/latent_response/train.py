"""Budget-bound LR training with exact task weighting and resumable validation state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import signal
import threading
import numpy as np
import torch
from .checkpoint import digest,parameter_sha,save_checkpoint,load_checkpoint,validation_transaction
from .preflight import ROOT,build_inputs,cases,write
from .data import make_view
from .execution import view_loss,hierarchical_loss,StopRequested
from .run_control import lr_multiplier,require_execution_identity,append_event,reconcile_logs,BudgetExhausted
from .telemetry import training_writer,epoch_length
from .epoch_batches import EpochBatches


def require_training(authorization=None, config=None):
    if not isinstance(authorization,dict) or not isinstance(config,dict):
        raise PermissionError('Explicit budget-bound user training authorization required')
    if authorization.get('source')!='user_in_current_chat' or authorization.get('config_sha256')!=digest(config):
        raise PermissionError('Authorization does not bind this training configuration')
    if 'epochs' in config:
        if config.get('phase') not in ('E2','E3') or authorization.get('phase')!=config['phase']:
            raise PermissionError('Phase mismatch')
        for key in ('epochs','batch_size','validations_per_epoch','checkpoint_every_batches'):
            if type(config.get(key)) is not int or config[key]<1:raise ValueError('Positive integer '+key+' required')
        if any(authorization.get(k)!=config[k] for k in ('epochs','batch_size')):
            raise PermissionError('Authorization must bind epochs and batch size')
        if 'steps' in config or config.get('e2_members') or config.get('objective')!='observation_equal':
            raise ValueError('Epoch training requires the full pool and observation_equal loss, without a step budget')
        if not 0<=config.get('warmup_epochs',0)<config['epochs']:
            raise ValueError('Require 0 <= warmup_epochs < epochs')
        if type(config.get('validation_seed',1729)) is not int:raise ValueError('Integer validation_seed required')
        return
    if config.get('phase') not in ('E2','E3') or type(config.get('steps')) is not int or config['steps']<1:
        raise PermissionError('Positive, explicit E2/E3 update budget required')
    if authorization.get('phase')!=config['phase'] or authorization.get('steps')!=config['steps']:
        raise PermissionError('Phase or budget mismatch')


def select_rows(rows,seed,cursor):
    families={}
    for row in rows:
        if row['role']=='TRAIN' and len(row['bands'])>=2:
            family='S2' if row['sensor']=='S2' else 'Landsat' if row['sensor'] in ('L8','L9') else row['sensor']
            families.setdefault(family,[]).append(row)
    chosen=[]
    for family,members in sorted(families.items()):
        epoch=cursor//len(members)
        members=sorted(members,key=lambda r:hashlib.sha256(f"{seed}:{epoch}:{family}:{r['id']}".encode()).digest())
        chosen.append((family,members[cursor%len(members)]))
    if not chosen:raise ValueError('No admitted TRAIN observations')
    return chosen


def update(model,dataset,optimizer,scheduler,state,config,device,stop_requested=None,*,batch=None):
    if stop_requested is not None and stop_requested():raise StopRequested()
    if 'epochs' not in config and state.get("spent_updates",state["cursor"])>=config.get("steps",float("inf")):raise BudgetExhausted()
    if 'epochs' in config and not batch:raise ValueError('Epoch training requires an explicit nonempty batch')
    rows=dataset.rows
    if config.get('e2_members'):
        if config['phase']!='E2':raise ValueError('E2 cohort cannot restrict E3 membership')
        rows=[r for r in rows if r['id'] in config['e2_members']]
        if len(rows)!=len(config['e2_members']):raise ValueError('E2 member missing from TRAIN')
    selected=select_rows(rows,config['seed'],state['cursor']) if batch is None else batch
    if any(row['role']!='TRAIN' for _,row in selected):raise ValueError('Training batch contains a held-out observation')
    lookup={r['id']:i for i,r in enumerate(dataset.rows)}
    optimizer.zero_grad(set_to_none=True);records=[]
    for family,row in selected:
        observation=dataset[lookup[row['id']]];channels=dataset.channels(observation,device)
        tasks={}
        for kind in ('A','B1'):
            view=make_view(observation,channels,kind,seed=config['seed'],update=config.get('mask_epoch',state['cursor']))
            tasks[kind]=view_loss(model,view,backward=True,loss_weight=.5/len(selected),
                tile_rows=config['tile_rows'],chunk_rows=config['chunk_rows'],chunk_pixels=config['chunk_pixels'],stop_requested=stop_requested)
        records.append(dict(source=family,id=row['id'],tasks=tasks))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
    if stop_requested is not None and stop_requested():raise StopRequested()
    sequence=state.get('spent_updates',state['cursor'])+1
    journal=Path(config['run_dir'])/'optimizer_attempts.jsonl' if config.get('run_dir') else None
    if journal:append_event(journal,dict(event='intent',sequence=sequence,cursor=state['cursor']+1))
    state['spent_updates']=sequence
    applied_lr=optimizer.param_groups[0]['lr']
    optimizer.step()
    if journal:append_event(journal,dict(event='applied',sequence=sequence,cursor=state['cursor']+1))
    scheduler.step();state['cursor']+=1
    values=[(r['source'],r['tasks']['A']['loss'],r['tasks']['B1']['loss']) for r in records]
    loss=sum(.5*a+.5*b for _,a,b in values)/len(values) if batch is not None else hierarchical_loss(values)
    return dict(cursor=state['cursor'],loss=loss,gradient_norm=float(norm),members=records,lr=applied_lr,next_lr=optimizer.param_groups[0]['lr'],spent_updates=state['spent_updates'],batch_size=len(records))


def validate(model,dataset,optimizer,scheduler,state,panel,config,device):
    values=[]
    with validation_transaction(model,optimizer,state,scheduler),torch.no_grad():
        model.eval();lookup={r['id']:i for i,r in enumerate(dataset.rows)}
        for item in panel:
            observation=dataset[lookup[item['id']]];channels=dataset.channels(observation,device);tasks={}
            for kind in ('A','B1'):
                mask_options=dict(seed=config.get('validation_seed',1729),update=0) if 'epochs' in config else {}
                tasks[kind]=view_loss(model,make_view(observation,channels,kind,**mask_options),tile_rows=config['tile_rows'],
                    chunk_rows=config['chunk_rows'],chunk_pixels=config['chunk_pixels'])['loss']
            family='S2' if item['sensor']=='S2' else 'Landsat'
            values.append((family,tasks['A'],tasks['B1']))
    metric=sum(.5*a+.5*b for _,a,b in values)/len(values) if 'epochs' in config else hierarchical_loss(values)
    per_source={source:sum(.5*a+.5*b for s,a,b in values if s==source)/sum(s==source for s,_,_ in values) for source in {s for s,_,_ in values}}
    result=dict(cursor=state['cursor'],completed=True,metric=metric,panels=len(values),per_source=per_source,
                reduction='observation_equal' if 'epochs' in config else 'source_equal')
    # Commit after validation restores live state; save must see this latest result.
    state['validation']=result
    return result


def main():
    if int(os.environ.get('WORLD_SIZE','1'))>1 or (torch.distributed.is_initialized() and torch.distributed.get_world_size()>1):
        raise RuntimeError('This training entry is single-process; distributed gradient synchronization is not implemented')
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--resume',type=Path);args=parser.parse_args()
    config=json.loads(args.config.read_text());approval=json.loads(args.authorization.read_text());require_training(approval,config)
    if args.run_dir.exists() and not args.resume:raise ValueError('New run requires a new directory')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    base,datasets,recipe,model,identity=build_inputs()
    require_execution_identity(approval,config,identity,datasets['TRAIN'].rows)
    evidence=json.loads(Path(config['preflight']).read_text())
    if evidence['status']!='PASS' or evidence['identity']!=identity:raise ValueError('Fresh exact-identity E1 evidence required')
    if config['phase']=='E3':
        gate=json.loads(Path(config['e2_receipt']).read_text())
        if gate.get('status')!='PASS' or gate.get('model_identity')!=identity:raise ValueError('Matching E2 PASS required')
    identity=dict(identity,training_config=digest(config),authorization=digest(approval))
    write(args.run_dir/'identity.json',identity);write(args.run_dir/'config.json',config)
    device=torch.device(config['device']);model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    batches=EpochBatches(datasets['TRAIN'].rows,config['seed'],config['batch_size']) if 'epochs' in config else None
    steps=config['epochs']*len(batches) if batches is not None else config['steps']
    warmup=min(steps-1,round(config.get('warmup_epochs',0)*len(batches))) if batches is not None else config['warmup_steps']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_multiplier(n,steps,warmup))
    state=dict(cursor=0,spent_updates=0,validation={},best=None,status='RUNNING')
    if args.resume:
        load_checkpoint(args.resume,model,optimizer,state,identity,scheduler)
        reconcile_logs(args.run_dir,state)
        state['status']='RUNNING'
    stop_event=threading.Event()
    signal.signal(signal.SIGINT,lambda *_:stop_event.set())
    signal.signal(signal.SIGTERM,lambda *_:stop_event.set())
    stop_requested=lambda:stop_event.is_set() or (args.run_dir/'STOP_REQUESTED').exists()
    runtime_config=dict(config,run_dir=str(args.run_dir))
    panel=datasets['VAL'].rows if batches is not None else [c for c in cases(datasets) if c['role']=='VAL']
    before=parameter_sha(model)
    updates_per_epoch=epoch_length(datasets['TRAIN'].rows,config)
    try:
        with training_writer(config,identity,state,args.run_dir,datasets['TRAIN'].rows,resuming=args.resume is not None) as writer:
            write(args.run_dir/'status.json',dict(state,parameter_sha=before,identity=identity))
            while state['cursor']<steps:
                batch=batches.batch(state['cursor']) if batches is not None else None
                if batches is not None:runtime_config['mask_epoch']=state['cursor']//len(batches)
                start=time.perf_counter();record=update(model,datasets['TRAIN'],optimizer,scheduler,state,runtime_config,device,stop_requested,batch=batch)
                record['seconds']=time.perf_counter()-start
                epoch=batches.progress(state['cursor']) if batches is not None else state['cursor']/updates_per_epoch
                if batches is not None:
                    state.update(epoch=epoch,completed_epochs=state['cursor']//len(batches),batch_in_epoch=state['cursor']%len(batches),
                        observations_seen=batches.observations_seen(state['cursor']))
                    state['epoch_loss_sum']=state.get('epoch_loss_sum',0.)+record['loss']*record['batch_size']
                    state['epoch_observations']=state.get('epoch_observations',0)+record['batch_size']
                    if state['batch_in_epoch']==0:
                        state['last_epoch_loss']=state['epoch_loss_sum']/state['epoch_observations']
                        state['epoch_loss_sum']=0.;state['epoch_observations']=0
                    record.update(epoch=epoch,observations_seen=state['observations_seen'])
                append_event(args.run_dir/'updates.jsonl',record)
                if writer is not None:
                    writer.log(step=state['cursor'],epoch=epoch,values=dict(train_loss=record['loss'],
                        learning_rate=record['lr'],gradient_norm=record['gradient_norm'],
                        update_seconds=record['seconds'],spent_updates=state['spent_updates']))
                    if batches is not None:
                        values=dict(batch_size=record['batch_size'],observations_seen=state['observations_seen'])
                        if state['batch_in_epoch']==0:values['epoch_train_loss']=state['last_epoch_loss']
                        writer.log(step=state['cursor'],epoch=epoch,values=values)
                validate_now=batches.validation_due(state['cursor'],config['validations_per_epoch']) if batches is not None else state['cursor']%config['validate_every']==0
                if validate_now or state['cursor']==steps:
                    result=validate(model,datasets['VAL'],optimizer,scheduler,state,panel,config,device)
                    if writer is not None:
                        values=dict(val_loss=result['metric'])
                        if batches is not None:values.update({('val_s2_loss' if key=='S2' else 'val_landsat_loss'):value for key,value in result['per_source'].items()})
                        writer.log(step=state['cursor'],epoch=epoch,values=values)
                    if state['best'] is None or result['metric']<state['best']:
                        state['best']=result['metric'];save_checkpoint(args.run_dir/'best.pt',model,optimizer,state,identity,scheduler)
                checkpoint_every=config['checkpoint_every_batches'] if batches is not None else config['checkpoint_every']
                if state['cursor']%checkpoint_every==0 or validate_now or state['cursor']==steps:
                    save_checkpoint(args.run_dir/'last.pt',model,optimizer,state,identity,scheduler)
                write(args.run_dir/'status.json',dict(state,parameter_sha=parameter_sha(model)))
                if batches is not None:print('epoch',round(epoch,6),'batch',(state['cursor']-1)%len(batches)+1,'/',len(batches),'observations',record['batch_size'],'loss',record['loss'],flush=True)
                else:print('update',state['cursor'],'loss',record['loss'],flush=True)
            state['status']='COMPLETE'
            write(args.run_dir/'status.json',dict(state,parameter_sha=parameter_sha(model)))
            write(args.run_dir/'receipt.json',dict(status='PASS',phase=config['phase'],updates=state['cursor'],
                parameter_sha_before=before,parameter_sha_after=parameter_sha(model),model_identity={k:v for k,v in identity.items() if k not in ('training_config','authorization')},
                validation=state['validation'],spent_updates=state['spent_updates']))
    except (StopRequested,BudgetExhausted) as exc:
        state['status']='STOPPED' if isinstance(exc,StopRequested) else 'BUDGET_EXHAUSTED'
        save_checkpoint(args.run_dir/'last.pt',model,optimizer,state,identity,scheduler)
        write(args.run_dir/'status.json',dict(state,parameter_sha=parameter_sha(model)))
    except Exception as exc:
        write(args.run_dir/'status.json',dict(state,status='FAILED',error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()

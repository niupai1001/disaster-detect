"""Fixed-cohort learning and actual Epoch-path preflight; never starts formal training."""
import argparse
import json
from pathlib import Path
import time
import torch
from .data import ROOT,load_data,identity,build_model,EpochPlan,validate_config
from .train import batch_loss,validate,update,writer_for,save,write
from .check import learning_cohort,metadata_audit
from encoder_v3.latent_response.checkpoint import parameter_sha
from encoder_v3.latent_response.run_control import lr_multiplier,append_event


def source_means(records):
    sources={}
    for r in records:sources.setdefault(r['sensor'],[]).append(r['loss'])
    return {k:sum(v)/len(v) for k,v in sources.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('fixed','epoch'),required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/config_shared_mae_v1.json')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh preflight directory required')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads(args.config.read_text());validate_config(config)
    datasets,resolver=load_data();binding=identity(config,datasets,resolver)
    admission=metadata_audit(datasets,resolver);write(args.output/'metadata.json',admission)
    if admission['status']!='PASS':raise ValueError('Physical admission failed')
    write(args.output/'config.json',config)
    model=build_model(config).to(config['device']);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    plan=EpochPlan(datasets['TRAIN'].rows,config['seed'],config['batch_size'])
    total=config['epochs']*len(plan);warmup=round(config['warmup_epochs']*len(plan))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:1. if args.phase=='fixed' else lr_multiplier(n,total,warmup))
    state=dict(cursor=0,spent_updates=0,epoch_index=0,status='RUNNING',best=None)
    rows=learning_cohort(datasets['TRAIN']);before=parameter_sha(model)
    initial_parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    with torch.no_grad():initial,initial_records=batch_loss(model,datasets['TRAIN'],rows,resolver,config)
    zero=sum(sum(t['zero_loss'] for t in r['tasks'].values())/len(r['tasks']) for r in initial_records)/len(initial_records)
    report=dict(status='RUNNING',phase=args.phase,identity=binding,initial_train_loss=initial,initial_per_source=source_means(initial_records),
                zero_baseline=zero,parameter_sha_before=before,diagnostic_members=[r['id'] for r in rows],updates=[],validations=[],training_release_ready=False)
    if args.phase=='epoch':
        start=time.perf_counter();result=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,config)
        result.update(cursor=0,seconds=time.perf_counter()-start);report['validations'].append(result);state['best']=result['metric']
    viewer=dict(config,viewer=dict(config['viewer'],experiment=dict(id='shared-mae-'+args.phase+'-preflight',name='共享MAE · '+('固定样本学习检查' if args.phase=='fixed' else 'Epoch入口稳定性检查'))))
    with writer_for(viewer,binding,state,args.output) as writer:
        if args.phase=='epoch':writer.log(step=0,epoch=0.,values=dict(val_loss=report['validations'][0]['metric']))
        for cursor in range(100 if args.phase=='fixed' else len(plan)):
            selected=rows if args.phase=='fixed' else plan.batch(state['cursor'])
            state['epoch_index']=0;start=time.perf_counter()
            record=update(model,datasets['TRAIN'],selected,resolver,optimizer,scheduler,state,dict(config,run_dir=str(args.output)))
            record['seconds']=time.perf_counter()-start;append_event(args.output/'updates.jsonl',record)
            report['updates'].append({k:v for k,v in record.items() if k!='members'})
            if state['cursor']%10==0:save(args.output/'last.pt',model,optimizer,scheduler,state,binding)
            progress=float(state['cursor']) if args.phase=='fixed' else plan.progress(state['cursor'])
            writer.log(step=state['cursor'],epoch=progress,values=dict(train_loss=record['loss'],learning_rate=record['lr'],gradient_norm=record['gradient_norm'],
                batch_size=len(selected),update_seconds=record['seconds'],observations_seen=len(rows)*state['cursor'] if args.phase=='fixed' else plan.observations_seen(state['cursor'])))
            print(args.phase,state['cursor'],record['loss'],record['seconds'],flush=True)
            if args.phase=='fixed' and state['cursor']>=20 and state['cursor']%10==0:
                with torch.no_grad():current,records=batch_loss(model,datasets['TRAIN'],rows,resolver,config)
                report.update(final_train_loss=current,final_per_source=source_means(records))
                stable=all(v<=report['initial_per_source'][k]*1.1 for k,v in report['final_per_source'].items())
                if current<=initial*.9 and current<zero and stable:break
            if args.phase=='epoch' and (state['cursor'] in (10,20) or plan.validation_due(state['cursor'],4)):
                start=time.perf_counter();result=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,config)
                result.update(cursor=state['cursor'],seconds=time.perf_counter()-start);report['validations'].append(result)
                writer.log(step=state['cursor'],epoch=progress,values=dict(val_loss=result['metric']))
                state['best']=min(state['best'],result['metric'])
                bad=result['metric']>state['best']*1.1
                state['regressions']=state.get('regressions',0)+1 if bad else 0
                if state['regressions']>=2:report['stop_reason']='Two full validations >10% worse than best';break
                if state['cursor']>=20 and state['best']<report['validations'][0]['metric']:break
            if state['cursor']%10==0:save(args.output/'last.pt',model,optimizer,scheduler,state,binding)
            write(args.output/'report.json',report)
        with torch.no_grad():final,records=batch_loss(model,datasets['TRAIN'],rows,resolver,config)
        report.update(final_train_loss=final,final_per_source=source_means(records),real_optimizer_steps=state['cursor'],parameter_sha_after=parameter_sha(model))
        changed={}
        for n,p in model.named_parameters():changed.setdefault(n.split('.')[0],[]).append(float((p.detach().cpu()-initial_parameters[n]).square().sum()))
        report['module_parameter_change_l2']={k:sum(v)**.5 for k,v in changed.items()}
        hook=model.encoder.register_forward_hook(lambda module,inputs,outputs:[torch.zeros_like(x) for x in outputs])
        with torch.no_grad():ablated,_=batch_loss(model,datasets['TRAIN'],rows,resolver,config)
        hook.remove();report['zero_encoder_diagnostic_loss']=ablated
        if args.phase=='fixed':
            report['passed']=final<=initial*.9 and final<zero and all(v<=report['initial_per_source'][k]*1.1 for k,v in report['final_per_source'].items())
        else:report['passed']=state['cursor']>=20 and state['best']<report['validations'][0]['metric'] and state.get('regressions',0)<2
        report['status']='PASS' if report['passed'] else 'FAIL';state['status']=report['status'];save(args.output/'last.pt',model,optimizer,scheduler,state,binding)
    write(args.output/'report.json',report)


if __name__=='__main__':main()

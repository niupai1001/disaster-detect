"""Metadata admission, real-data no-update checks and explicit short update verification."""
import argparse
import json
from pathlib import Path
import time
import torch
from encoder_v3.latent_response.checkpoint import parameter_sha, digest
from .data import ROOT, load_data, identity, validate_config, build_model, sample_from_dataset
from .train import write, batch_loss, validate, update, writer_for, save


def metadata_audit(datasets,resolver):
    unique={};blocked=[];count=0
    for role,ds in datasets.items():
        for row in ds.rows:
            for band in row['bands']:
                try:
                    value=resolver.resolve(row,band).as_dict();count+=1
                    unique[digest(value)]=value
                except ValueError as exc:
                    blocked.append(dict(id=row['id'],role=role,channel=band['band_key'],reason=str(exc)))
    return dict(status='PASS' if not blocked else 'BLOCKED',observations={role:len(ds.rows) for role,ds in datasets.items()},
                resolved_channels=count,unique_physical_conditions=list(unique.values()),blocked=blocked,catalog_sha256=resolver.catalog_sha)


def cohort(datasets,largest=False):
    chosen=[]
    for role,sensor in (('TRAIN','S2'),('TRAIN','L8'),('TRAIN','EnMAP'),('VAL','S2'),('VAL','L8')):
        members=[r for r in datasets[role].rows if r['sensor']==sensor]
        chosen.append(min(members,key=lambda r:max(b['shape'][0]*b['shape'][1] for b in r['bands'])))
    if largest:chosen.append(max((r for r in datasets['TRAIN'].rows if r['sensor']=='S2'),key=lambda r:max(b['shape'][0]*b['shape'][1] for b in r['bands'])))
    return chosen


def learning_cohort(dataset):
    """Eight fixed middle-range observations per source; engineering only, not an Epoch."""
    chosen=[]
    for sensor in ('S2','L8','EnMAP'):
        members=sorted((r for r in dataset.rows if r['sensor']==sensor),
                       key=lambda r:(max(b['shape'][0]*b['shape'][1] for b in r['bands']),r['id']))
        if len(members)<8:raise ValueError('Eight admitted training members per source required')
        indices=[round((len(members)-1)*(.25+.5*i/7)) for i in range(8)]
        if len(set(indices))!=8:raise ValueError('Learning cohort must contain eight distinct members per source')
        chosen.extend(members[i] for i in indices)
    return chosen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/config_shared_mae_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('audit','check','verify-updates'),default='audit')
    parser.add_argument('--patch-metres',type=float,help='Explicit engineering-run L; does not modify the project configuration')
    parser.add_argument('--include-largest',action='store_true')
    parser.add_argument('--updates',type=int,default=20)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.patch_metres is not None:config['patch_metres']=args.patch_metres
    validate_config(config,require_scale=args.mode!='audit')
    if args.output.exists():raise ValueError('Checks require a fresh evidence directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datasets,resolver=load_data();binding=identity(config,datasets,resolver)
    audit=metadata_audit(datasets,resolver);write(args.output/'metadata.json',audit);write(args.output/'config.json',config)
    if audit['status']!='PASS':raise ValueError('Unresolved physical metadata; see metadata.json. No guessed wavelengths or silent member drops.')
    if args.mode=='audit':
        write(args.output/'report.json',dict(status='PASS',mode='audit',identity=binding,real_optimizer_steps=0,training_release_ready=False));return
    model=build_model(config).to(config['device']);before=parameter_sha(model);results=[]
    rows=cohort(datasets,args.include_largest)
    for row in rows:
        data=datasets[row['role']];lookup={r['id']:i for i,r in enumerate(data.rows)}
        sample=sample_from_dataset(data,lookup[row['id']],resolver,config['device'])
        tasks={}
        for kind in model.available_tasks(sample):
            model.zero_grad(set_to_none=True)
            if str(config['device']).startswith('cuda'):torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.perf_counter();loss,detail=model([sample],kind=kind,seed=config['seed']);loss.mean().backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
            if str(config['device']).startswith('cuda'):torch.cuda.synchronize()
            if not bool(torch.isfinite(loss).all()):raise ValueError('Nonfinite engineering loss')
            memory_grad=float(model.memory.slots.grad.abs().sum())
            bias_grad=float(model.convolution.bias_bases.grad.abs().sum())
            if memory_grad<=0 or bias_grad<=0:raise ValueError('Shared memory or conditional bias is disconnected from the loss')
            tasks[kind]=dict(loss=float(loss.detach()),seconds=time.perf_counter()-start,gradient_norm=float(norm),memory_gradient_l1=memory_grad,
                conditional_bias_gradient_l1=bias_grad,peak_bytes=torch.cuda.max_memory_allocated() if str(config['device']).startswith('cuda') else None,
                census={k:v for k,v in detail[0].items() if k not in ('predictions','target_masks')})
        results.append(dict(id=row['id'],role=row['role'],sensor=row['sensor'],tasks=tasks))
        print(row['sensor'],row['id'],'PASS',flush=True)
        write(args.output/'engineering.json',dict(status='RUNNING',identity=binding,cases=results))
    assert parameter_sha(model)==before
    report=dict(status='PASS',mode=args.mode,identity=binding,real_optimizer_steps=0,cases=results,
                parameters=sum(p.numel() for p in model.parameters()),parameter_sha_before=before,parameter_sha_after=before,
                training_release_ready=False,scope='Engineering only; exact original target footprints, changed architecture/masks; not numerical equivalence to LR')
    write(args.output/'engineering.json',report)
    if args.mode=='verify-updates':
        if args.updates<1:raise ValueError('Positive explicit check update count required')
        train_rows=learning_cohort(datasets['TRAIN'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        state=dict(cursor=0,spent_updates=0,epoch_index=0,status='RUNNING',best=None)
        with torch.no_grad():initial_train,_=batch_loss(model,datasets['TRAIN'],train_rows,resolver,config)
        initial_validation=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,config)
        check_config=dict(config,viewer=dict(experiment=dict(id='encoder-v3-shared-mae-update-check',name='Encoder V3 · 共享MAE真实更新检查'),
            variant=dict(config['viewer']['variant'],description=config['viewer']['variant']['description']+'；固定工程成员、固定遮盖，常数学习率，仅检查可学习性与稳定性。')))
        with writer_for(check_config,binding,state,args.output) as writer:
            for _ in range(args.updates):
                start=time.perf_counter()
                record=update(model,datasets['TRAIN'],train_rows,resolver,optimizer,scheduler,state,dict(config,run_dir=str(args.output)))
                if writer:writer.log(step=state['cursor'],epoch=float(state['cursor']),values=dict(train_loss=record['loss'],learning_rate=record['lr'],
                    gradient_norm=record['gradient_norm'],batch_size=len(train_rows),observations_seen=state['cursor']*len(train_rows),update_seconds=time.perf_counter()-start))
                if state['cursor']%5==0 or state['cursor']==args.updates:
                    result=validate(model,datasets['VAL'],resolver,optimizer,scheduler,state,config)
                    if writer:writer.log(step=state['cursor'],epoch=float(state['cursor']),values=dict(val_loss=result['metric'],
                        **{('val_s2_loss' if k=='S2' else 'val_landsat_loss'):v for k,v in result['per_source'].items() if k in ('S2','Landsat')}))
                    save(args.output/'last.pt',model,optimizer,scheduler,state,binding)
                    print('check update',state['cursor'],'VAL',result['metric'],flush=True)
            with torch.no_grad():final_train,_=batch_loss(model,datasets['TRAIN'],train_rows,resolver,config)
            state['status']='COMPLETE';save(args.output/'last.pt',model,optimizer,scheduler,state,binding)
        report.update(real_optimizer_steps=state['cursor'],parameter_sha_after=parameter_sha(model),initial_train_loss=initial_train,final_train_loss=final_train,
            initial_validation=initial_validation,final_validation=state['validation'],
            training_release_ready=False,
            diagnostic_members=[r['id'] for r in train_rows],
            learning_check_passed=args.updates>=20 and final_train<=initial_train*.9 and state['validation']['metric']<=initial_validation['metric'])
        report['scope']='Fixed cohort update diagnostic; validation full frozen VAL. Not an epoch training result or proof of downstream transfer.'
    write(args.output/'report.json',report)


if __name__=='__main__':main()

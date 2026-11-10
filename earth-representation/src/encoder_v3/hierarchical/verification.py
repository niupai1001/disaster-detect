"""Exact complete-state recovery evidence; no optimizer updates in these helpers."""
import hashlib
import json
import copy
from pathlib import Path
import numpy as np
import torch


def fingerprint(value):
    result=hashlib.sha256()
    def add(item):
        if isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous()
            result.update(str((tensor.dtype,tuple(tensor.shape))).encode());result.update(tensor.numpy().tobytes())
        elif isinstance(item,np.ndarray):
            result.update(str((item.dtype,item.shape)).encode());result.update(item.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):add(key);add(item[key])
        elif isinstance(item,(list,tuple)):
            result.update(type(item).__name__.encode())
            for v in item:add(v)
        else:result.update(json.dumps(item,sort_keys=True).encode())
    add(value)
    return result.hexdigest()


def compare_snapshots(first,second):
    result={name:fingerprint(first[name])==fingerprint(second[name]) for name in ('model','optimizer','scheduler','runner','rng','modes')}
    result['ema']=fingerprint({k:v for k,v in first['model'].items() if k.startswith('target.') or k=='ema_updates'})==fingerprint({k:v for k,v in second['model'].items() if k.startswith('target.') or k=='ema_updates'})
    if not all(result.values()):raise AssertionError(result)
    return result


def migrate_single_parent_checkpoint(payload,reference_model,model,optimizer,binding):
    """Preserve effective parameters/momenta and runner/RNG when pruning provably dead Q/K."""
    from .model import canonical_weights
    if payload.get('schema')!='v3_h_checkpoint_v1':raise ValueError('V3-H source checkpoint required')
    old_names=[name for name,p in reference_model.named_parameters() if p.requires_grad]
    new_names=[name for name,p in model.named_parameters() if p.requires_grad]
    old_groups=payload['optimizer']['param_groups'];new_opt=optimizer.state_dict()
    if len(old_groups)!=1 or len(new_opt['param_groups'])!=1:raise ValueError('One AdamW parameter group required')
    if len(old_names)!=len(old_groups[0]['params']):raise ValueError('Source optimizer/name identity mismatch')
    old_ids=dict(zip(old_names,old_groups[0]['params']))
    old_parameters=dict(reference_model.named_parameters())
    states={}
    for name,new_id in zip(new_names,new_opt['param_groups'][0]['params']):
        old_name=name;offset=None
        if name.endswith(('value_weight','value_bias')) and '.cross.' in name:
            prefix,tail=name.rsplit('.',1)
            old_name=prefix+('.kv.weight' if tail=='value_weight' else '.kv.bias')
            offset=old_parameters[old_name].shape[0]//2
        old_id=old_ids[old_name]
        if old_id not in payload['optimizer']['state']:continue
        state=copy.deepcopy(payload['optimizer']['state'][old_id])
        if offset is not None:
            state={key:value[offset:].clone() if isinstance(value,torch.Tensor) and value.shape==old_parameters[old_name].shape else value for key,value in state.items()}
        states[new_id]=state
    group=copy.deepcopy(old_groups[0]);group['params']=new_opt['param_groups'][0]['params']
    result=copy.deepcopy(payload)
    result.update(identity=binding,model=canonical_weights(payload['model']),optimizer={'state':states,'param_groups':[group]})
    result['modes']={name:payload['modes'][name] for name,_ in model.named_modules()}
    result['migration']={'source_identity':payload['identity'],'operation':'Exact single-parent attention V-only reparameterization; Q/K/query norm had zero influence','runner_and_rng_preserved':True}
    model.load_state_dict(result['model'],strict=True)
    return result


def prepare_migrated_run(source_run,source_checkpoint,destination,reference_model,model,optimizer,binding):
    """Create an explicit new-identity continuation with inherited journals and a new Viewer Run."""
    source_run=Path(source_run).resolve();destination=Path(destination).resolve()
    if destination==source_run or destination.is_relative_to(source_run):raise ValueError('Migration must not overwrite the source Run')
    payload=torch.load(source_checkpoint,map_location='cpu',weights_only=False)
    if payload['runner'].get('status') not in ('STOPPED','COMPLETE'):raise ValueError('Safely stopped source checkpoint required')
    cursor=payload['runner']['cursor'];source_viewer=payload['runner'].get('viewer')
    if not source_viewer:raise ValueError('Source Viewer identity required')
    rows=[json.loads(line) for line in (source_run/'updates.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [r['cursor'] for r in rows if r['cursor']<=cursor]!=list(range(1,cursor+1)):raise ValueError('Source journal/checkpoint mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    def convert(source):
        result=migrate_single_parent_checkpoint(source,reference_model,model,optimizer,binding)
        viewer=result['runner'].pop('viewer',None)
        result['runner'].update(source_viewer=viewer,migration_source_cursor=source['runner']['cursor'])
        result['migration'].update(source_run=str(source_run),source_run_id=source_viewer['run_id'],new_viewer_run_required=True)
        return result
    for name in ('updates.jsonl','optimizer_attempts.jsonl'):
        records=[json.loads(line) for line in (source_run/name).read_text(encoding='utf-8').splitlines() if line.strip()]
        for record in records:record['inherited_from_run_id']=source_viewer['run_id']
        (destination/name).write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    if (source_run/'best.pt').exists():
        best=torch.load(source_run/'best.pt',map_location='cpu',weights_only=False)
        torch.save(convert(best),destination/'best.pt')
    result=convert(payload);checkpoint=destination/'initial_checkpoint.pt';torch.save(result,checkpoint)
    lineage={'source_run':str(source_run),'source_run_id':source_viewer['run_id'],'source_checkpoint':str(Path(source_checkpoint).resolve()),
        'source_checkpoint_sha256':hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest(),'source_identity':payload['identity'],
        'destination_identity':binding,'inherited_cursor':cursor,'new_viewer_run':True,'reason':'Equivalent execution optimization and V-only single-parent reparameterization; effective weights, AdamW moments, scheduler, RNG and sampling cursor preserved'}
    (destination/'migration_lineage.json').write_text(json.dumps(lineage,indent=2)+'\n',encoding='utf-8')
    return checkpoint


def gradient_contributions(model,sample,config,*,epoch=0):
    """Per-observation raw/weighted objectives and actual shared-Encoder gradient norms."""
    kinds=model.available_tasks(sample)
    denominator=sum(config['task_weights'][k] for k in kinds)
    names,parameters=zip(*model.online.named_parameters())
    result={'observation':sample.identifier,'normalization':'one observation; available task weights normalized; scale JEPA mean','tasks':{}}
    for kind in kinds:
        weight=config['task_weights'][kind]/denominator
        _,details=model([sample],kind=kind,seed=config['seed'],epoch=epoch,return_objectives=True)
        info=details[0];terms=info.pop('objective_tensors')
        metrics={}
        for name,objective in terms.items():
            factor=weight if name=='mae' else weight*config['lambda_jepa']
            if name.startswith('jepa_scale_'):factor/=max(1,len(info['jepa_counts']))
            grads=torch.autograd.grad(objective,parameters,retain_graph=True,allow_unused=True) if objective.requires_grad else [None]*len(parameters)
            squares={}
            for parameter_name,gradient in zip(names,grads):
                if gradient is not None:
                    group=parameter_name.split('.')[0]
                    squares[group]=squares.get(group,objective.new_zeros(()))+gradient.float().square().sum()
            total=sum(squares.values(),objective.new_zeros(())).sqrt()
            metrics[name]={'raw_loss':float(objective.detach()),'weight':factor,'weighted_loss':float(objective.detach())*factor,
                'encoder_gradient_norm':float(total),'weighted_encoder_gradient_norm':float(total)*factor,
                'encoder_group_gradient_norms':{name:float(v.sqrt()) for name,v in squares.items()}}
        result['tasks'][kind]={'terms':metrics,'native_target_count':info['targets'],'jepa_target_counts':info['jepa_counts']}
    return result

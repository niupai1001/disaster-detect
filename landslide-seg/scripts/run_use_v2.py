"""Release-bound USE v2 execution. Engineering never authorizes formal training."""
import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from pretrain_use_v2 import (ROOT,inputs,initialize,encoder,RecoveryProbe,ssl_update,lr,
    atomic_save,checkpoint,restore,state_sha256,deterministic)
from audit_use_independent_data import audit_cache,EXTERNAL,sha,read
import numpy as np
import torch
import yaml
from landslide_seg.models.use_v2 import FrozenReadout
from landslide_seg.data.segformer_dataset import read_manifest,load_stats,collate
from landslide_seg.data.use_dataset import USESegDataset
from landslide_seg.training.segformer_trainer import _seg_loss
from evaluate_use_v2 import calibrate,evaluate,selection,representations,full_diagnostics,efficiency,plot_features

BASE=ROOT/'outputs/use_v2'
CONTRACT=ROOT/'configs/use_v2'
DEPENDENCIES=('torch','numpy','scipy','PyYAML','rasterio','pyproj','shapely','matplotlib','transformers')
SCRIPTS=('run_use_v2.py','pretrain_use_v2.py','evaluate_use_v2.py','pretrain_use_independent.py',
         'audit_use_independent_data.py','check_research_protocol.py')


def write_json(path,value,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf8') as f: json.dump(value,f,indent=2,ensure_ascii=False)
    else:
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8'); os.replace(temporary,path)


def atomic_copy(source,target):
    target=Path(target); temporary=target.with_suffix(target.suffix+'.tmp')
    shutil.copy2(source,temporary); os.replace(temporary,target)


def states_equal(left,right):
    if torch.is_tensor(left): return torch.equal(left.cpu(),right.cpu())
    if isinstance(left,np.ndarray): return np.array_equal(left,right)
    if isinstance(left,dict): return left.keys()==right.keys() and all(states_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)): return len(left)==len(right) and all(states_equal(a,b) for a,b in zip(left,right))
    return left==right


def audit_data(out):
    _,_,audit,members_hash=inputs('multisensor')
    rows=[json.loads(line) for line in (ROOT/'outputs/use_independent_frozen/cache_audit_v1/members.jsonl').read_text(encoding='utf8').splitlines()]
    roles=Counter(); sensors=Counter(); failures=[]
    for row in rows:
        try:
            audit_cache(EXTERNAL,row); roles[row['role']]+=1; sensors[(row['role'],row['sensor'])]+=1
        except Exception as error: failures.append(dict(file=row['file'],error=str(error)))
    result=dict(passed=dict(roles)=={'TRAIN':11132,'PRETRAIN_VAL':599,'GUARD':239} and not failures,
        roles=dict(roles),sensor_counts={str(k):v for k,v in sensors.items()},failures=failures,
        original_audit_sha256=audit,members_sha256=members_hash,
        limitation='Every cached array rechecked; original product/geographic audits reused')
    write_json(out,result,True); return result


def matrix():
    jobs={}
    for candidate in ('mean','attention'):
        jobs['pretrain_'+candidate]=dict(mode='pretrain',candidate=candidate,steps=20000,warmup=1000,task=None,source=None,validation_interval=1000)
    sources=[(c,f'pretrain_{c}/u{s}.pt') for c in ('mean','attention') for s in (5000,10000,15000,20000)]
    sources += [('old',None),('mean',None),('attention',None)]
    for candidate,source in sources:
        name=f'{candidate}_{Path(source).stem}' if source else candidate+'_reference'
        for task in ('fine','context'):
            jobs[f'probe_{name}_{task}']=dict(mode='probe',candidate=candidate,steps=1000,warmup=50,task=task,source=source,validation_interval=1000)
    for candidate in ('old','mean','attention'):
        for fine in (False,True):
            jobs[f'downstream_{candidate}_{"context_fine" if fine else "context"}']=dict(mode='downstream',candidate=candidate,
                steps=5000,warmup=250,task=None,fine=fine,source=None if candidate=='old' else f'pretrain_{candidate}/best_unlabeled.pt',validation_interval=500)
    assert sum(j['steps'] for j in jobs.values())==92000
    return jobs


def prepare():
    CONTRACT.mkdir(parents=True,exist_ok=True)
    splits,stats,audit,members=inputs('multisensor')
    calibration=calibrate(splits,stats)
    write_json(CONTRACT/'calibration.json',calibration,True)
    base=yaml.safe_load((ROOT/'configs/segformer/use/physical_mixed.yaml').read_text())
    s=base['segformer']; s['model']=dict(type='use_v2',in_chans=10,freeze_use=True)
    s['data']['mixed_bands']=False
    s['train'].update(epochs=5000,stop_updates=5000,schedule_updates=5000,max_steps=5000,report_steps=[1000,2000,3000,4000,5000])
    s['optim'].update(schedule='cosine',warmup_frac=.05,encoder_lr=.0006,head_lr=.0006)
    s['experiment']='use_v2_frozen'; s['output_dir']='outputs/use_v2/runs'
    base['use_v2']=dict(seed=1234,precision='FP32',microbatch=2,accumulation=4,lr=.0006,weight_decay=.01,clip=1.,
        calibration_sha256=sha(CONTRACT/'calibration.json'),audit_sha256=audit,members_sha256=members,stats=stats,
        statistics_source='outputs/use/pretraining/resolved_config.json',initialization='new seed1234 USE; shared head seed1235',
        selection='sensor-equal task-equal normalized unlabeled MAE',masks='quality / artificial hidden / output support kept separately',
        readout='Fine32 + summed four canonical32 Context projections -> Conv64:32:32; target physical+known+visible mean/std -> Linear8:32:33',
        augmentation='stateless H/V flip and quarter rotation; spatial transform updated',
        sampling='seed1234+update, S2 six and Landsat two; uniform S2 locations preserves source proportions',
        area30=dict(resampling='area average into local UTM 30m',valid_area_min=.8,supported_latitude=[-80,84]),budget=92000,jobs=matrix())
    path=CONTRACT/'execution.yaml'
    with path.open('x',encoding='utf8') as f: yaml.safe_dump(base,f,allow_unicode=True,sort_keys=False)
    from check_research_protocol import audit_config
    proof=audit_config(path); write_json(CONTRACT/'protocol_preflight.json',proof,True)
    return dict(config=str(path),calibration=str(CONTRACT/'calibration.json'),preflight=proof['ok'])


def load_configuration():
    config=yaml.safe_load((CONTRACT/'execution.yaml').read_text(encoding='utf8')); cal=read(CONTRACT/'calibration.json')
    if sha(CONTRACT/'calibration.json')!=config['use_v2']['calibration_sha256']: raise ValueError('Calibration changed')
    splits,stats,audit,members=inputs('multisensor')
    if (stats,audit,members)!=(config['use_v2']['stats'],config['use_v2']['audit_sha256'],config['use_v2']['members_sha256']):
        raise ValueError('Data/statistics contract changed')
    from check_research_protocol import audit_config
    audit_config(CONTRACT/'execution.yaml')
    return config,cal,splits,stats


def source_checkpoint(job,engineering=False):
    if engineering or not job['source']: return None
    path=BASE/'runs'/job['source']; summary=read(path.parent/'summary.json')
    if not summary['complete'] or summary['updates']!=20000: raise ValueError('Source pretraining incomplete')
    if sha(path)!=summary['checkpoints'][path.name]: raise ValueError('Source checkpoint changed')
    return path


def build(job,device='cpu',engineering=False):
    if job['mode']=='pretrain': return initialize(job['candidate'],device)
    enc=encoder(job['candidate'],source_checkpoint(job,engineering),device)
    if job['mode']=='probe': return RecoveryProbe(enc,job['task']).to(device)
    torch.manual_seed(1235)
    return FrozenReadout(enc,fine=job['fine']).to(device)


def datasets(config):
    d=config['segformer']['data']; records=read_manifest(ROOT/d['manifest']); stats=load_stats(ROOT/d['stats'])
    train=USESegDataset([r for r in records if r['split']=='train'],stats,augment=True,seed=1234,preload=False)
    val=USESegDataset([r for r in records if r['split']=='val'],stats,augment=False,seed=1234,preload=False)
    if (len(train),len(val))!=(2223,669): raise ValueError('Downstream membership differs')
    return train,val


def downstream_update(model,optimizer,dataset,step,config,device):
    optimizer.zero_grad(set_to_none=True)
    indices=np.random.default_rng(1234+step).integers(len(dataset),size=8)
    dataset.set_epoch(step); losses=[]
    for start in range(0,8,2):
        batch=collate([dataset[int(i)] for i in indices[start:start+2]])
        pred=model(batch['image'].to(device))
        loss=_seg_loss(config,pred,batch['mask'].to(device),batch['valid'].to(device))
        (loss/4).backward(); losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm): raise ValueError('Nonfinite gradients')
    optimizer.step(); frozen=model.use.verify(optimizer)
    return dict(loss=float(np.mean(losses)),sample_ids=[dataset.records[int(i)]['sample_id'] for i in indices],frozen_sha256=frozen,gradient_norm=float(norm))


@torch.no_grad()
def downstream_evaluate(model,dataset,config,device,limit=None,prediction_dir=None):
    from landslide_seg.training.segformer_metrics import evaluate_tile_v2,summarize_tiles_v2,summarize_by_event_v2,pixel_average_precision
    model.eval(); rows=[]; probabilities=[]; targets=[]; tile_ap=[]
    for index in range(len(dataset) if limit is None else min(limit,len(dataset))):
        sample=dataset[index]; batch=collate([sample])
        prob=model(batch['image'].to(device))[0,0].sigmoid().cpu().numpy()
        gt=sample['mask'].numpy().astype(bool); valid=sample['valid'].numpy(); rec=dataset.records[index]
        row=evaluate_tile_v2(prob,gt,valid,threshold=.5,thresholds_px=(50.,200.),iou_threshold=.25,iou_strict=.5,boundary_tol=1)
        row.update(sample_id=sample['sample_id'],event_id=sample['event_id'],bbox=rec['bbox'],crs=rec['crs'])
        rows.append(row); tile_ap.append(pixel_average_precision(prob[valid],gt[valid]))
        probabilities.append(prob[valid]); targets.append(gt[valid])
        if prediction_dir:
            path=Path(prediction_dir)/f'{index:04d}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,prob=prob,mask=gt,valid=valid,sample_id=sample['sample_id'])
    by_event=summarize_by_event_v2(rows); score=float(np.mean([v['pixel']['iou'] for v in by_event.values()]))
    for event,values in by_event.items():
        indices=[i for i,row in enumerate(rows) if row['event_id']==event]
        values['pixel_ap']=pixel_average_precision(np.concatenate([probabilities[i] for i in indices]),np.concatenate([targets[i] for i in indices]))
    result=dict(overall=summarize_tiles_v2(rows),by_event=by_event,event_mean_iou=score,tiles=rows,
                pixel_ap=pixel_average_precision(np.concatenate(probabilities),np.concatenate(targets)),tile_ap=tile_ap)
    model.use.verify(); model.train(); return result


def train(job_id,out,device,engineering=False,engineering_steps=None,resume=False,release_hash=None,stop_at=None,gate=None):
    config,cal,splits,stats=load_configuration(); job=config['use_v2']['jobs'][job_id]
    total=job['steps']; steps=engineering_steps if engineering else total
    if engineering and (steps is None or not 1<=steps<=100): raise ValueError('Engineering range is 1..100 updates')
    deterministic(); model=build(job,device,engineering)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0006,weight_decay=.01)
    if hasattr(model.use,'verify'): model.use.verify(optimizer)
    identity=dict(job_id=job_id,job=job,release=release_hash,engineering=engineering,
        config_sha256=sha(CONTRACT/'execution.yaml'),calibration_sha256=sha(CONTRACT/'calibration.json'),initialization_sha256=state_sha256(model),
        source_sha256={**{name:sha(ROOT/'scripts'/name) for name in SCRIPTS},'models/use_v2.py':sha(ROOT/'src/landslide_seg/models/use_v2.py')},
        source_checkpoint_sha256=sha(source_checkpoint(job)) if job['source'] and not engineering else None)
    out=Path(out)
    if resume:
        saved=restore(model,optimizer,out/'last.pt',identity); start=saved['step']; best=saved['best']; history=saved['history']
        # A crash after curve append but before atomic checkpoint must not duplicate the update ledger.
        (out/'curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf8')
        # Reconcile the best alias if a crash saved a newer candidate before last.
        candidates=[r for r in history if r.get('selection_improved')]
        if candidates:
            selected=out/f"selected_{candidates[-1]['step']}.pt"
            atomic_copy(selected,out/('best.pt' if job['mode']=='downstream' else 'best_unlabeled.pt'))
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'resolved_config.json',dict(configuration=config,job=job,identity=identity,device_gate=gate,
            command=sys.argv,cwd=str(Path.cwd())),True)
        start=0; best=float('-inf') if job['mode']=='downstream' else float('inf'); history=[]
        atomic_save(checkpoint(model,optimizer,0,best,identity,history),out/'initialization.pt')
    train_ds,val_ds=datasets(config) if job['mode']=='downstream' else (None,None)
    write_json(out/'process.json',dict(pid=os.getpid(),command=sys.argv,device=device,device_gate=gate,started=time.time(),status='RUNNING'))
    began=time.perf_counter(); last_saved=start; step=start
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(start+1,(min(steps,stop_at) if stop_at is not None else steps)+1):
            for group in optimizer.param_groups: group['lr']=lr(step,total,job['warmup'])
            model.train()
            record=downstream_update(model,optimizer,train_ds,step,config,device) if train_ds is not None else ssl_update(model,optimizer,splits['TRAIN'],stats,step,cal,job['task'],device)
            record.update(step=step,lr=optimizer.param_groups[0]['lr']); history.append(record)
            with (out/'curve.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(record)+'\n')
            validate=step%job['validation_interval']==0 or step==steps or (engineering and step%25==0)
            if validate:
                if train_ds is not None:
                    evaluation=downstream_evaluate(model,val_ds,config,device,limit=4 if engineering else None)
                    score=evaluation['event_mean_iou']; improved=score>best
                else:
                    evaluation=evaluate(model,splits['PRETRAIN_VAL'],stats,device,limit=2 if engineering else None,calibration=cal,tasks=(job['task'],) if job['task'] else ('spectral','fine','context'))
                    score=selection(evaluation,cal,(job['task'],) if job['task'] else ('spectral','fine','context')); improved=score<best
                write_json(out/f'validation_{step}.json',evaluation)
                record.update(selection_score=score,selection_improved=bool(improved))
                if improved:
                    best=score
                    atomic_save(checkpoint(model,optimizer,step,best,identity,history),out/f'selected_{step}.pt')
                    atomic_copy(out/f'selected_{step}.pt',out/('best.pt' if train_ds is not None else 'best_unlabeled.pt'))
            if step%1000==0 or validate or step==stop_at:
                atomic_save(checkpoint(model,optimizer,step,best,identity,history),out/'last.pt'); last_saved=step
            if job['mode']=='pretrain' and step in (5000,10000,15000,20000):
                atomic_save(checkpoint(model,optimizer,step,best,identity,history),out/f'u{step}.pt')
                write_json(out/f'representation_{step}.json',full_diagnostics(model,splits['PRETRAIN_VAL'],stats,device,cal,limit=4 if engineering else None))
                plot_features(model,splits['PRETRAIN_VAL']['s2'][0],stats,out/f'features_{step}.png',device)
        if last_saved<step: atomic_save(checkpoint(model,optimizer,step,best,identity,history),out/'last.pt')
        (out/'curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history),encoding='utf8')
        elapsed=time.perf_counter()-began
        summary=dict(complete=step==steps,updates=step,formal_updates=0 if engineering else step,engineering_updates=step if engineering else 0,
            best_score=best,identity=identity,seconds_this_session=elapsed,seconds_per_update=elapsed/max(1,step-start),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
            parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            checkpoints={p.name:sha(p) for p in out.glob('*.pt')},initialization_reset_sha256=state_sha256(build(job,device,engineering)))
        if summary['initialization_reset_sha256']!=identity['initialization_sha256']: raise ValueError('Fresh reset mismatch')
        write_json(out/'summary.json',summary)
        write_json(out/'process.json',dict(pid=os.getpid(),status='COMPLETED' if summary['complete'] else 'STOPPED_FOR_RESUME',updated=time.time()))
        return summary
    except BaseException as error:
        write_json(out/'failure.json',dict(error=repr(error),successful_updates=len(history),saved_step=last_saved,command=sys.argv,pid=os.getpid())); raise


def verify_release(release,audit_marker=None):
    release=Path(release); manifest=read(release/'manifest.json'); digest=sha(release/'manifest.json')
    if (release/'manifest.sha256').read_text().strip()!=digest: raise ValueError('Release manifest changed')
    for name,expected in manifest['files'].items():
        if sha(ROOT/name)!=expected or sha(release/'snapshot'/name)!=expected: raise ValueError('Release-bound file changed: '+name)
    for package,version in manifest['environment']['dependencies'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('Dependency changed: '+package)
    if sys.version!=manifest['environment']['python']: raise ValueError('Python changed')
    if audit_marker:
        audit=read(Path(audit_marker))
        if audit.get('status')!='READY_FOR_TRAINING' or audit.get('release_sha256')!=digest: raise ValueError('Independent C approval missing or mismatched')
    if manifest['status']!='READY_FOR_AUDIT': raise ValueError('Incomplete release')
    return digest


GPU_MIN_FREE_MIB=4096
GPU_QUERY='index,uuid,display_active,memory.total,memory.used'


def gpu_cards():
    """Physical card facts from nvidia-smi; display use is recorded, not rejected."""
    rows=subprocess.check_output(['nvidia-smi',f'--query-gpu={GPU_QUERY}','--format=csv,noheader'],text=True)
    cards={}
    for row in rows.splitlines():
        index,uuid,display,total,used=(v.strip() for v in row.split(','))
        cards[int(index)]=dict(uuid=uuid,display=display,free_mib=int(total.split()[0])-int(used.split()[0]))
    return cards


def card_process_alive(pid):
    if os.name!='nt':
        try: os.kill(pid,0); return True
        except ProcessLookupError: return False
    output=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/NH'],capture_output=True,text=True).stdout
    return str(pid) in output


def claim_card(card):
    """One live process per physical card; a stale lock from a dead owner is replaced."""
    path=BASE/'runtime'/f"gpu{card}.lock"; path.parent.mkdir(parents=True,exist_ok=True)
    for _ in range(2):
        try:
            handle=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(handle,str(os.getpid()).encode()); os.close(handle)
            return path
        except FileExistsError:
            owner=int(path.read_text() or 0)
            if owner and card_process_alive(owner):
                raise ValueError(f'GPU {card} already hosts a USE v2 job (pid {owner}); pick the other card')
            path.unlink(missing_ok=True)
    raise ValueError(f'Could not claim GPU {card}')


def physical_index(device):
    """Logical torch index -> physical nvidia-smi index, honouring CUDA_VISIBLE_DEVICES."""
    index=int(torch.device(device).index or 0)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    order=[int(v) for v in visible.replace(' ','').split(',')] if visible and visible!='all' else None
    if order and index>=len(order): raise ValueError('CUDA_VISIBLE_DEVICES does not expose the requested index')
    return (order[index] if order else index),visible


def active_jobs():
    """In-flight runs recorded by their own process.json; used to refuse a duplicate card start."""
    result=[]
    for path in BASE.rglob('process.json'):
        try: data=read(path)
        except (OSError,ValueError): continue
        if data.get('status')=='RUNNING' and data.get('pid') and card_process_alive(int(data['pid'])):
            result.append((path.parent.relative_to(ROOT).as_posix(),int(torch.device(data.get('device','cpu')).index or 0)))
    return result


def probe_device(index):
    """Real allocation and backward on the card; a dead or unsupported device fails here."""
    with torch.cuda.device(index):
        probe=torch.ones(1024,1024,device='cuda',requires_grad=True)
        probe.square().mean().backward()
        if not torch.isfinite(probe.grad).all(): raise ValueError('Device probe produced nonfinite gradients')
        torch.cuda.synchronize(); free,total=torch.cuda.mem_get_info()
    del probe; torch.cuda.empty_cache()
    return dict(probe='1024x1024 forward+backward finite',free_mib_after_probe=round(free/2**20),
                total_mib=round(total/2**20))


def gpu_available(device):
    """Launch gate: free memory, no duplicate job on the card, then a real allocation/backward."""
    if device=='cpu': return dict(device='cpu')
    index=int(torch.device(device).index or 0)
    physical,visible=physical_index(device)
    cards=gpu_cards()
    if physical not in cards: raise ValueError(f'GPU {physical} not found')
    card=cards[physical]
    if card['free_mib']<GPU_MIN_FREE_MIB:
        raise ValueError(f"GPU {physical} has {card['free_mib']} MiB free; {GPU_MIN_FREE_MIB} MiB required")
    busy=[name for name,owned in active_jobs() if physical_index(f'cuda:{owned}')[0]==physical]
    if busy: raise ValueError(f'GPU {physical} already hosts a running USE v2 job: {busy[0]}')
    return dict(device=device,physical_index=physical,uuid=card['uuid'],display_active=card['display'],
        visible_devices=visible,free_mib_before=card['free_mib'],min_free_mib=GPU_MIN_FREE_MIB,**probe_device(index))


@contextmanager
def gpu_guard(device):
    """Hold the card for one process; the lock prevents a duplicate start on that card."""
    gate=gpu_available(device); lock=claim_card(gate['physical_index']) if 'physical_index' in gate else None
    try: yield gate
    finally:
        if lock: lock.unlink(missing_ok=True)


def allocation():
    """Two-card plan: V2-N and V2-A on separate cards, serial within a card, no DDP or batch change."""
    plan={}
    for job in matrix():
        if 'attention' in job: plan[job]='cuda:1'
        elif 'mean' in job: plan[job]='cuda:0'
        else: plan[job]='cuda:0' if job in ('probe_old_reference_fine','downstream_old_context') else 'cuda:1'
    assert set(plan)==set(matrix())
    steps={card:sum(matrix()[j]['steps'] for j in plan if plan[j]==card) for card in ('cuda:0','cuda:1')}
    assert steps=={'cuda:0':46000,'cuda:1':46000}, steps
    return plan


def waves():
    """Dependency waves; within a card the jobs of one wave run serially."""
    plan=allocation(); jobs=list(matrix())
    return [dict(wave=name,jobs={card:[j for j in jobs if plan[j]==card and j.startswith(prefix)]
                for card in ('cuda:0','cuda:1')})
            for name,prefix in (('pretrain','pretrain_'),('probe','probe_'),('downstream','downstream_'))]


def evaluate_job(job_id,path,out,device):
    config,cal,splits,stats=load_configuration(); job=config['use_v2']['jobs'][job_id]
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['identity']['job_id']!=job_id or saved['identity']['config_sha256']!=sha(CONTRACT/'execution.yaml'):
        raise ValueError('Evaluation job/config differs from saved state')
    model=build(job,device,saved['identity']['engineering']); model.load_state_dict(saved['model'],strict=True)
    if state_sha256(model)!=saved['model_sha256']: raise ValueError('Evaluation state hash differs')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    if job['mode']=='downstream':
        _,val=datasets(config)
        result=downstream_evaluate(model,val,config,device,prediction_dir=out/'predictions')
    elif job['mode']=='probe':
        result=evaluate(model,splits['PRETRAIN_VAL'],stats,device,calibration=cal,tasks=(job['task'],))
    else:
        result=full_diagnostics(model,splits['PRETRAIN_VAL'],stats,device,cal)
        plot_features(model,splits['PRETRAIN_VAL']['s2'][0],stats,out/'features.png',device)
    write_json(out/'evaluation.json',dict(checkpoint_sha256=sha(Path(path)),checkpoint_step=saved['step'],job=job_id,results=result),True)
    return dict(output=str(out/'evaluation.json'),checkpoint_step=saved['step'])


def snapshot(out,acceptance_path='outputs/use_v2/engineering/acceptance_v2.json',
             marker='outputs/use_v2/audit/READY_FOR_TRAINING.json'):
    acceptance=read(ROOT/acceptance_path)
    if acceptance.get('passed') is not True: raise ValueError('Engineering acceptance required')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    tests=ROOT/acceptance.get('tests_xml','outputs/use_v2/engineering/tests_v3.xml')
    test_files=[ROOT/'tests'/name for name in acceptance.get('tests_files',['test_use_v2.py'])]
    files=[*sorted((ROOT/'src/landslide_seg').rglob('*.py')),*[ROOT/'scripts'/name for name in SCRIPTS],*test_files,*sorted(CONTRACT.glob('*')),
        ROOT/'outputs/use_independent_frozen/cache_audit_v1/acceptance.json',ROOT/'outputs/use_independent_frozen/cache_audit_v1/members.jsonl',
        ROOT/'outputs/use/pretraining/resolved_config.json',ROOT/'outputs/use/quality/index.json',ROOT/'outputs/use_independent_frozen/pretraining/multisensor/best.pt',
        ROOT/'outputs/use_independent_frozen/pretraining/multisensor/summary.json',*[ROOT/'manifests/v4r1'/n for n in ('pool_100.jsonl','stats_core25.json','protocol.json')],
        ROOT/'docs/assets/research/research_policy.json',*sorted((BASE/'engineering').glob('*.json')),tests,ROOT/acceptance_path]
    suite=ROOT/acceptance['suite_directory']
    files += [suite/name for name in ('acceptance.json','diagnostic_smoke.json','efficiency.json','feature_smoke.png','reference_representations.json','source_at_start.json')]
    for job in acceptance['suite']['summaries']:
        files += [suite/job/name for name in ('initialization.pt','last.pt','summary.json','resolved_config.json','curve.jsonl')]
    for job in acceptance['suite']['resume']:
        files += [suite/(job+suffix)/'last.pt' for suffix in ('_whole','_split')]
    digests={}
    for source in dict.fromkeys(files):
        name=source.relative_to(ROOT).as_posix(); expected=sha(source)
        target=out/'snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        if sha(target)!=expected: raise ValueError('Changed during snapshot: '+name)
        digests[name]=expected
    manifest=dict(status='READY_FOR_AUDIT',files=digests,created=time.time(),engineering_acceptance=acceptance,
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),dependencies={n:importlib.metadata.version(n) for n in DEPENDENCIES}),
        cwd=str(ROOT),formal_updates=0,commands={j:[sys.executable,'-X','utf8','scripts/run_use_v2.py','train','--job',j,'--release',str(out),
        '--audit-marker',str(ROOT/marker),'--device',allocation()[j]] for j in matrix()},
        allocation=allocation(),waves=waves(),
        device_policy='Two physical cards, one formal process per card, serial within a card; display use and other users\' processes are checked but not rejected or terminated; every launch passes a free-memory and real allocation/backward gate and holds a per-card lock; no DDP, no batch change.',
        resume='Same command with --resume; restores optimizer, RNG, sampler update, and best selection',
        limits=['Independent C audit required','No real paired cross-sensor data or second labeled task','Engineering scores are not formal results'])
    write_json(out/'manifest.json',manifest,True); (out/'manifest.sha256').write_text(sha(out/'manifest.json')+'\n',encoding='utf8')
    verify_release(out); return dict(release=str(out),sha256=sha(out/'manifest.json'),status=manifest['status'])


def engineering_suite(out,device,gate=None):
    from pretrain_use_v2 import load_rows,task_view
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    config,cal,splits,stats=load_configuration()
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [*[ROOT/'scripts'/n for n in SCRIPTS],ROOT/'src/landslide_seg/models/use_v2.py']}
    write_json(out/'source_at_start.json',sources,True)
    summaries={}
    prototypes=['pretrain_mean','pretrain_attention']
    prototypes += [f'probe_{c}_reference_{task}' for c in ('old','mean','attention') for task in ('fine','context')]
    prototypes += [j for j in matrix() if j.startswith('downstream_')]
    for job in prototypes:
        summaries[job]=train(job,out/job,device,engineering=True,engineering_steps=100 if job.startswith('pretrain') else 2,gate=gate)
        print(json.dumps(dict(completed=job,steps=summaries[job]['updates'])),flush=True)
    # Check exact resumed updates on all three real execution modes.
    recovery={}
    for job in ('pretrain_mean','probe_mean_reference_context','downstream_mean_context_fine'):
        whole=train(job,out/(job+'_whole'),device,engineering=True,engineering_steps=2,gate=gate)
        train(job,out/(job+'_split'),device,engineering=True,engineering_steps=2,stop_at=1,gate=gate)
        split=train(job,out/(job+'_split'),device,engineering=True,engineering_steps=2,resume=True,gate=gate)
        left=torch.load(out/(job+'_whole')/'last.pt',map_location='cpu',weights_only=False)
        right=torch.load(out/(job+'_split')/'last.pt',map_location='cpu',weights_only=False)
        if left['model_sha256']!=right['model_sha256'] or left['history']!=right['history'] or not states_equal(left['optimizer'],right['optimizer']) or not states_equal(left['rng'],right['rng']):
            raise ValueError('Real resume differs: '+job)
        recovery[job]=dict(exact=True,optimizer_exact=True,rng_exact=True,model_sha256=left['model_sha256'],updates=2)
    model=initialize('attention',device)
    diag=full_diagnostics(model,splits['PRETRAIN_VAL'],stats,device,cal,limit=2)
    write_json(out/'diagnostic_smoke.json',diag,True)
    plot_features(model,splits['PRETRAIN_VAL']['s2'][0],stats,out/'feature_smoke.png',device)
    from pretrain_use_v2 import Reconstruction
    costs={}
    for candidate in ('old','mean','attention'):
        benchmark=Reconstruction(encoder(candidate,device=device)).to(device) if candidate=='old' else initialize(candidate,device)
        costs[candidate]=efficiency(benchmark,device)
        del benchmark
    write_json(out/'efficiency.json',costs,True)
    references={}
    for candidate in ('old','mean','attention'):
        references[candidate]=representations(encoder(candidate,device=device),splits['PRETRAIN_VAL'],stats,device,limit=4)
    write_json(out/'reference_representations.json',references,True)
    # Real S2 subsets and the six actual Landsat channels, including metadata/mask permutation.
    real_checks=[]
    for sensor,counts in (('s2',(1,3,4,6,10)),('landsat',(6,))):
        original=load_rows([splits['PRETRAIN_VAL'][sensor][0]],stats,device)
        c=original['x'].shape[1]
        for count in counts:
            o={k:v[:,:count] if torch.is_tensor(v) and v.ndim>=2 and v.shape[1]==c else v for k,v in original.items()}
            with torch.no_grad():
                reference=model.use(o)
                perm=torch.arange(count-1,-1,-1,device=device)
                other={k:v[:,perm] if torch.is_tensor(v) and v.ndim>=2 and v.shape[1]==count else v for k,v in o.items()}
                output=model.use(other)
                torch.testing.assert_close(reference['fine'],output['fine'],atol=2e-5,rtol=2e-4)
            real_checks.append(dict(sensor=sensor,channels=count,passed=True))
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()): raise ValueError('Source changed during engineering suite')
    receipt=dict(passed=True,summaries=summaries,resume=recovery,real_interface=real_checks,
        diagnostic_sha256=sha(out/'diagnostic_smoke.json'),source_sha256=sources,
        engineering_updates=sum(v['updates'] for v in summaries.values())+12,formal_updates=0,
        note='Milestone probe prototypes use fresh matching v2 encoders; actual formal source-checkpoint loading is covered by independent unit tests',
        fresh_initialization_hashes={j:s['initialization_reset_sha256'] for j,s in summaries.items()},
        output=str(out),commands=dict(cwd=str(ROOT),argv=sys.argv,python=sys.executable))
    write_json(out/'acceptance.json',receipt,True)
    return receipt


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('prepare','audit-data','engineering','engineering-suite','train','snapshot','verify','evaluate','compare'))
    p.add_argument('--out'); p.add_argument('--job',choices=tuple(matrix())); p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=100); p.add_argument('--resume',action='store_true'); p.add_argument('--release'); p.add_argument('--audit-marker')
    p.add_argument('--acceptance',default='outputs/use_v2/engineering/acceptance_v2.json'); p.add_argument('--marker',default='outputs/use_v2/audit/READY_FOR_TRAINING.json')
    p.add_argument('--checkpoint'); p.add_argument('--left'); p.add_argument('--right')
    a=p.parse_args(); torch.set_num_threads(4)
    if a.action=='prepare': result=prepare()
    elif a.action=='audit-data': result=audit_data(a.out)
    elif a.action=='snapshot': result=snapshot(a.out,a.acceptance,a.marker)
    elif a.action=='verify': result=verify_release(a.release,a.audit_marker)
    elif a.action=='engineering-suite':
        if not Path(a.out).resolve().is_relative_to((BASE/'engineering').resolve()): raise ValueError('Engineering output escapes scope')
        with gpu_guard(a.device) as gate: result=engineering_suite(a.out,a.device,gate)
    elif a.action=='compare':
        from evaluate_use_v2 import paired_bootstrap,geographic_bootstrap
        left=read(Path(a.left)); right=read(Path(a.right))
        left=left.get('results',left); right=right.get('results',right)
        result=geographic_bootstrap(left,right) if 'tiles' in left else paired_bootstrap(left.get('reconstruction',left),right.get('reconstruction',right))
        write_json(a.out,result,True)
    elif a.action=='evaluate':
        if a.release: verify_release(a.release)
        result=evaluate_job(a.job,a.checkpoint,a.out,a.device)
    elif a.action=='engineering':
        target=Path(a.out).resolve()
        if not target.is_relative_to((BASE/'engineering').resolve()): raise ValueError('Engineering outputs must stay in engineering')
        with gpu_guard(a.device) as gate:
            result=train(a.job,target,a.device,engineering=True,engineering_steps=a.steps,resume=a.resume,gate=gate)
    else:
        if not a.release or not a.audit_marker: p.error('Formal training requires release and independent audit marker')
        digest=verify_release(a.release,a.audit_marker)
        with gpu_guard(a.device) as gate:
            result=train(a.job,BASE/'runs'/a.job,a.device,resume=a.resume,release_hash=digest,gate=gate)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()


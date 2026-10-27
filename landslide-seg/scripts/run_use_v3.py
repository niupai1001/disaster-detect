"""Engineering runner. Formal launch remains subject to independent B audit."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
import yaml
import pretrain_use_v3 as pt
import run_use_v2 as rt
from audit_use_independent_data import sha
from landslide_seg.models.frozen_use import state_sha256
ROOT=pt.ROOT
BASE=ROOT/'outputs/use_v3'
CONFIG=ROOT/'configs/use_v3/execution.yaml'
PRETRAIN_MILESTONES=(5000,10000,20000,30000)


def prepare():
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((ROOT/'configs/use_v2/execution.yaml').read_text())
    config.pop('use_v2',None)
    config['segformer']['model']['type']='use_v3'
    config['segformer']['output_dir']='outputs/use_v3/runs'
    config['use_v3']=dict(seed=1234,head_seed=1235,precision='FP32',microbatch=2,accumulation=4,
        steps=30000,warmup=1500,lr=.0006,weight_decay=.01,clip=1.,engineering_steps=500,
        weights=dict(spectral=1,fine=1,context=1,gradient=.1,consistency=.1),
        source_v2='outputs/use_v2/runs/pretrain_attention_tail/best_unlabeled.pt',
        formal_budget=62000)
    with CONFIG.open('x',encoding='utf8') as f: yaml.safe_dump(config,f,sort_keys=False)
    from check_research_protocol import audit_config
    rt.write_json(BASE/'engineering/protocol_preflight.json',audit_config(CONFIG))


def engineering(device,resume=False):
    config=yaml.safe_load(CONFIG.read_text()); contract=config['use_v3']
    from check_research_protocol import audit_config
    preflight=audit_config(CONFIG)
    rt.write_json(BASE/'engineering/protocol_preflight.json',preflight)
    splits,stats,audit,members=pt.inputs('multisensor')
    model=pt.initialize(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
    out=BASE/'engineering/trial500'; out.mkdir(parents=True,exist_ok=True)
    calibration_path=BASE/'engineering/calibration.json'
    if calibration_path.exists(): calibration=rt.read(calibration_path)
    else:
        calibration=pt.calibrate(model,splits,stats,device)
        rt.write_json(calibration_path,calibration,True)
    identity=dict(config=sha(CONFIG),members=members,audit=audit,calibration=sha(calibration_path),
        source={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'scripts/pretrain_use_v3.py',ROOT/'src/landslide_seg/models/use_v3.py',ROOT/'src/landslide_seg/models/use_v3_reconstruction.py']})
    history=[]; start=0
    if resume:
        saved=pt.restore(model,optimizer,out/'last.pt',identity); history=saved['history']; start=saved['step']
    else:
        if (out/'initial.pt').exists(): raise ValueError('Trial already exists; use --resume')
        pt.atomic_save(pt.checkpoint(model,optimizer,0,None,identity,[]),out/'initial.pt')
    # Reuse the project's existing physical-card lock and device probe, read-only v2 checks.
    with rt.gpu_guard(device) as gate:
        rt.write_json(out/'device_gate.json',gate)
        began=time.time()
        for step in range(start+1,501):
            rate=pt.lr(step,30000,1500)
            for group in optimizer.param_groups: group['lr']=rate
            record=pt.update(model,optimizer,splits['TRAIN'],stats,step,calibration,device)
            record['lr']=rate; history.append(record)
            if step%10==0:
                rt.write_json(out/'curve.json',history)
                pt.atomic_save(pt.checkpoint(model,optimizer,step,None,identity,history),out/'last.pt')
                print(json.dumps(dict(step=step,gradient_norm=record['gradient_norm'])),flush=True)
        rt.write_json(out/'summary.json',dict(complete=True,updates=500,formal_updates=0,
            seconds=time.time()-began,identity=identity,use_sha256=state_sha256(model.use),
            initialization=sha(out/'initial.pt'),last=sha(out/'last.pt'),
            limitation='Engineering only; never resume these weights into formal training'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','engineering','run','verify'])
    parser.add_argument('--device',default='cuda:1'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--job',choices=list(matrix())); parser.add_argument('--release',type=Path); parser.add_argument('--marker',type=Path)
    args=parser.parse_args(); torch.set_num_threads(4)
    if args.action=='prepare': prepare()
    elif args.action=='engineering': engineering(args.device,args.resume)
    elif args.action=='verify': verify_release(args.release)
    else: run_job(args.job,args.device,args.release,args.marker,args.resume)



def load_encoder(kind,source=None):
    if kind=='v2':
        from landslide_seg.models.use_v2 import USEv2
        torch.manual_seed(1234); model=USEv2()
    else:
        torch.manual_seed(1234); model=pt.USEv3()
    if source:
        source=Path(source)
        model.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['use'],strict=True)
    return model


def probe_update(model,optimizer,split,stats,step,task,device):
    rows=pt.sample_rows(split,step); optimizer.zero_grad(set_to_none=True); losses=[]
    for micro in range(4):
        original=pt.load_rows(rows[micro*2:micro*2+2],stats,device,1234+step*8+micro*2)
        tasks=('fine',) if task=='fine' else ('spectral','context')
        for name in tasks:
            view,hidden,summary=pt.task_view(original,name,1234+step*100+micro*3,block=(16,32,64)[(step-1)%3],metadata_mode='known')
            if not hidden.any(): continue
            pred=model(view,view['band_metadata'],view['known_mask'],summary)
            valid=original['pixel_valid']&original['band_valid'][:,:,None,None]
            loss=pt.spatial_error(pred,original['x'],hidden,valid) if name=='context' else pt.mean_error(pred,original['x'],hidden)
            (loss/(4*len(tasks))).backward(); losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm): raise ValueError('Nonfinite probe gradient')
    optimizer.step(); model.use.verify(optimizer)
    return dict(loss=sum(losses)/len(losses),gradient_norm=float(norm))


def matrix():
    jobs={'pretrain':dict(mode='pretrain',encoder='v3',steps=30000,warmup=1500)}
    for kind,points in (('v2',('initial','selected')),('v3',('initial','10000','20000','30000'))):
        for point in points:
            for task in ('fine','context'):
                jobs[f'probe_{kind}_{point}_{task}']=dict(mode='probe',encoder=kind,point=point,task=task,steps=1000,warmup=50)
        for network in ('unet','segformer'):
            jobs[f'seg_{kind}_{network}']=dict(mode='segmentation',encoder=kind,network=network,point='selected',steps=5000,warmup=250)
    assert sum(j['steps'] for j in jobs.values())==62000
    return jobs

def run_job(job_id,device,release,marker,resume=False):
    """Only B's independent, hash-bound marker can enable a formal run."""
    verify_release(release)
    proof=rt.read(Path(marker))
    if proof.get('status')!='READY_FOR_TRAINING' or proof.get('release_sha256')!=sha(Path(release)/'manifest.json'):
        raise ValueError('Independent READY_FOR_TRAINING for this exact release required')
    job=matrix()[job_id]; out=BASE/'runs'/job_id; out.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load(CONFIG.read_text()); splits,stats,audit,members=pt.inputs('multisensor')
    source=None
    if job['mode']!='pretrain':
        point=job['point']
        if job['encoder']=='v2' and point=='selected': source=ROOT/config['use_v3']['source_v2']
        elif job['encoder']=='v3':
            source=BASE/'runs/pretrain'/('initial.pt' if point=='initial' else 'best_unlabeled.pt' if point=='selected' else f'u{point}.pt')
        encoder=load_encoder(job['encoder'],source)
        if job['mode']=='probe': model=pt.initialize(device,job['task'],encoder)
        else:
            from landslide_seg.models.use_v3_reconstruction import FullDecoder
            model=FullDecoder(encoder,job['network'],ROOT/'weights/segformer/segformer-b0-finetuned-ade-512-512').to(device)
    else: model=pt.initialize(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0006,weight_decay=.01)
    identity=dict(release_sha256=sha(Path(release)/'manifest.json'),job=job,source=sha(source) if source else None,members=members,audit=audit)
    history=[]; start=0; best=None
    if resume:
        saved=pt.restore(model,optimizer,out/'last.pt',identity); history=saved['history']; start=saved['step']; best=saved['best']
    else:
        if (out/'initial.pt').exists(): raise ValueError('Output exists; explicit resume required')
        pt.atomic_save(pt.checkpoint(model,optimizer,0,best,identity,history),out/'initial.pt')
    rt.write_json(out/'summary.json',dict(complete=False,updates=start,identity=identity,best=best))
    if resume:
        # Recover the old last-before-milestone interruption without another update.
        if job['mode']=='pretrain' and start in PRETRAIN_MILESTONES and not (out/f'u{start}.pt').exists():
            pt.atomic_save(saved,out/f'u{start}.pt')
        check_job_checkpoints(out,job,identity,start)
    if job['mode']=='segmentation': train,val=rt.datasets(config)
    calibration=rt.read(BASE/'engineering/calibration.json')
    with rt.gpu_guard(device) as gate:
        rt.write_json(out/'device_gate.json',gate)
        for step in range(start+1,job['steps']+1):
            model.train()
            rate=pt.lr(step,job['steps'],job['warmup'])
            for group in optimizer.param_groups: group['lr']=rate
            if job['mode']=='pretrain': record=pt.update(model,optimizer,splits['TRAIN'],stats,step,calibration,device)
            elif job['mode']=='probe': record=probe_update(model,optimizer,splits['TRAIN'],stats,step,job['task'],device)
            else: record=rt.downstream_update(model,optimizer,train,step,config,device)
            record.update(step=step,lr=rate); history.append(record)
            if job['mode']=='segmentation' and step%500==0:
                evaluation=rt.downstream_evaluate(model,val,config,device)
                rt.write_json(out/f'eval_{step}.json',evaluation)
                if best is None or evaluation['event_mean_iou']>best:
                    best=evaluation['event_mean_iou']; pt.atomic_save(pt.checkpoint(model,optimizer,step,best,identity,history),out/'best.pt')
            # Commit milestone first: last must never advance past an unsaved milestone.
            if job['mode']=='pretrain' and step in PRETRAIN_MILESTONES:
                pt.atomic_save(pt.checkpoint(model,optimizer,step,best,identity,history),out/f'u{step}.pt')
            if step%100==0 or step==job['steps']:
                rt.write_json(out/'curve.json',history)
                pt.atomic_save(pt.checkpoint(model,optimizer,step,best,identity,history),out/'last.pt')
        checkpoints=check_job_checkpoints(out,job,identity,job['steps'])
        rt.write_json(out/'summary.json',dict(complete=True,updates=job['steps'],identity=identity,best=best,
            last_sha256=checkpoints['last.pt'],checkpoints=checkpoints))


def check_job_checkpoints(out,job,identity,step):
    required={'initial.pt':0,'last.pt':step}
    if job['mode']=='pretrain':
        required.update({f'u{s}.pt':s for s in PRETRAIN_MILESTONES if s<=step})
    hashes={}
    for name,expected_step in required.items():
        path=out/name
        if not path.is_file(): raise ValueError('Missing required checkpoint: '+name)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['step']!=expected_step or saved['identity']!=identity:
            raise ValueError('Checkpoint step/identity mismatch: '+name)
        hashes[name]=sha(path)
    return hashes


def verify_release(release):
    release=Path(release); manifest=rt.read(release/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest or sha(release/'snapshot'/name)!=digest: raise ValueError('Release changed: '+name)
    return manifest




def freeze(name='engineering_v1'):
    import importlib.metadata
    import platform
    import shutil
    summary=rt.read(BASE/'engineering/trial500/summary.json')
    if not summary.get('complete') or summary['updates']!=500: raise ValueError('500-step trial required')
    release=BASE/'releases'/name
    release.mkdir(parents=True,exist_ok=False)
    files=list((ROOT/'src/landslide_seg').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    files+=list((ROOT/'tests').glob('test_use_v3*.py'))+list(CONFIG.parent.glob('*'))
    files+=[BASE/'engineering/calibration.json',BASE/'engineering/tests.xml',
            BASE/'engineering/protocol_preflight.json',BASE/'engineering/trial500/initial.pt',
            BASE/'engineering/trial500/summary.json',BASE/'engineering/trial500/curve.json',
            ROOT/'outputs/use_independent_frozen/cache_audit_v1/members.jsonl',
            ROOT/'outputs/use_independent_frozen/cache_audit_v1/acceptance.json',
            ROOT/'outputs/use/pretraining/resolved_config.json',
            ROOT/'outputs/use_v2/runs/pretrain_attention_tail/best_unlabeled.pt',
            ROOT/'outputs/use_v2/runs/pretrain_attention_tail/summary.json',
            ROOT/'docs/assets/research/research_policy.json']
    files+=list((BASE/'engineering').glob('*.json'))+list((BASE/'engineering').glob('*.py'))+list((BASE/'engineering').glob('*.md'))
    files+=list((ROOT/'manifests/v4r1').glob('*'))
    files+=list((ROOT/'weights/segformer/segformer-b0-finetuned-ade-512-512').glob('*'))
    hashes={}
    for source in sorted(set(p for p in files if p.is_file() and p.name!='READY_FOR_AUDIT.json')):
        relative=source.relative_to(ROOT).as_posix(); hashes[relative]=sha(source)
        target=release/'snapshot'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                     packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
    rt.write_json(release/'manifest.json',dict(status='FROZEN_FOR_REVIEW',files=hashes,environment=environment,
        trial_updates=500,formal_updates=0,initialization_sha256=sha(BASE/'engineering/trial500/initial.pt')) ,True)
    verify_release(release)
    return release

if __name__=='__main__': main()

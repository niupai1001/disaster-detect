"""Single v4 candidate. Engineering is separate from B's release-bound formal jobs."""
import argparse
import hashlib
from contextlib import contextmanager
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
import torch
import yaml
import pretrain_use_v4 as pt
import run_use_v2 as rt
from audit_use_independent_data import sha
from landslide_seg.models.use_v4_mit import V3,MIT,FRONTEND
from landslide_seg.models.use_v3_reconstruction import packed_state_sha256

ROOT=pt.ROOT
BASE=ROOT/'outputs/use_v4'
CONFIG=ROOT/'configs/use_v4/execution.yaml'
EXPECTED={V3:'e20e32f9915974c570d2ad9a064af3bd46b75f0e0d78bafbf698637895533886',MIT/'config.json':'209caa9091e4632f7c8883c11170cd08ad29af68b23c09590aa4a5befb1a2a7f',MIT/'model.safetensors':'6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9'}


def matrix():
    jobs={'pretrain':dict(mode='pretrain',steps=20000,warmup=1000)}
    for step in (10000,20000):
        for task in ('fine','context'):
            jobs[f'probe_{step}_{task}']=dict(mode='probe',point=step,task=task,steps=1000,warmup=50)
    jobs['segment']=dict(mode='segmentation',steps=5000,warmup=250)
    assert sum(j['steps'] for j in jobs.values())==29000
    return jobs


def prepare():
    for path,digest in EXPECTED.items():
        if sha(path)!=digest: raise ValueError('Source hash mismatch: '+str(path))
    config=yaml.safe_load((ROOT/'configs/use_v3/execution.yaml').read_text())
    config.pop('use_v3')
    config['segformer']['model']['type']='use_v4_mit'
    config['segformer']['output_dir']='outputs/use_v4/runs'
    config['segformer']['train']['eval_every_updates']=500
    config['use_v4']=dict(seed=1234,head_seed=1235,precision='FP32',tf32=False,metadata_mode='known',
        source_v3=V3.relative_to(ROOT).as_posix(),source_mit=MIT.relative_to(ROOT).as_posix(),source_sha256={p.relative_to(ROOT).as_posix():d for p,d in EXPECTED.items()},
        train_observations=11132,validation_observations=599,guard_locations=239,sensor_batch=dict(s2=6,landsat=2),
        microbatch=2,accumulation=4,engineering_steps=200,engineering_eval_every=50,
        steps=20000,warmup=1000,frontend_peak_lr=1e-4,new_peak_lr=6e-4,weight_decay=.01,clip=1.,
        weights=dict(spectral=1,fine=1,context=1,gradient=.1,consistency=.1),loss_weight_sum=3.2,
        masks=dict(fine_fraction=.25,context_blocks=[16,32,64],spectral='known wavelength, half bands'),
        calibration=dict(role='TRAIN',per_sensor=32,seed=1234),
        public_evaluation_calibration='outputs/use_v3/engineering/calibration.json',
        milestones=[5000,10000,20000],formal_budget=29000,jobs=matrix(),
        selection='known PRETRAIN_VAL; Fine pixel/Context spectral/Context spatial equal, sensors equal; earlier tie',
        gpu_rule='one physical-card lock per process; B verifies measured peak plus margin before formal start',
        comparisons=['Fine is early, without v3 Context feedback','Context consistency uses reconstruction fusion','external ImageNet/ADE20K MiT supervision','Context probe has 1024 more weights than v3','coarse spatial error retains visible pixels in pooled cells'])
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    with CONFIG.open('x',encoding='utf8') as f: yaml.safe_dump(config,f,sort_keys=False,allow_unicode=True)
    engineering=json.loads(json.dumps(config)); engineering['use_v4']['execution_mode']='engineering'
    with (CONFIG.parent/'engineering.yaml').open('x',encoding='utf8') as f: yaml.safe_dump(engineering,f,sort_keys=False,allow_unicode=True)
    preflight()


def preflight():
    from check_research_protocol import audit_config
    for path,digest in EXPECTED.items():
        if sha(path)!=digest: raise ValueError('Source changed: '+str(path))
    for name in ('execution.yaml','engineering.yaml'):
        rt.write_json(BASE/'engineering'/('protocol_'+name+'.json'),audit_config(CONFIG.parent/name))


def load_encoder(source=None):
    torch.manual_seed(1234); model=pt.USEv4()
    if source:
        saved=torch.load(source,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['use'],strict=True)
        if packed_state_sha256(model)!=saved['use_sha256']: raise ValueError('USE checkpoint byte hash mismatch')
    return model


def schedule(opt,step,job):
    rate=pt.lr(step,job['steps'],job['warmup'])
    for group in opt.param_groups: group['lr']=rate*group.get('peak_lr',6e-4)/6e-4
    return [g['lr'] for g in opt.param_groups]


@torch.no_grad()
def validate_reconstruction(model,splits,stats,device,step):
    model.eval(); result={}
    for sensor,rows in splits['PRETRAIN_VAL'].items():
        records=[]
        for i,row in enumerate(rows):
            raw,counts=pt.views(model,pt.load_rows([row],stats,device),800000+i*3,(16,32,64)[i%3])
            records.append(dict(file=row['file'],raw={k:float(v) for k,v in raw.items()},counts=counts))
        result[sensor]=dict(rows=records,means={key:sum(r['raw'][key] for r in records)/len(records) for key in records[0]['raw']})
    model.train()
    return dict(step=step,sensors=result,role='PRETRAIN_VAL',selection_eligible=False)


def engineering(device,resume=False):
    preflight(); splits,stats,audit,members=pt.inputs('multisensor')
    out=BASE/'engineering/trial200'; out.mkdir(parents=True,exist_ok=True)
    with rt.gpu_guard(device) as gate:
        rt.write_json(out/'device_gate.json',gate)
        model=pt.initialize(device); opt=pt.optimizer(model)
        reference=packed_state_sha256(model.use.mit)
        calibration_path=BASE/'engineering/calibration.json'
        if calibration_path.exists(): calibration=rt.read(calibration_path)
        else:
            calibration=pt.calibrate(model,splits,stats,device)
            rt.write_json(calibration_path,calibration,True)
        source={p.relative_to(ROOT).as_posix():sha(p) for p in protected_sources()}
        identity=dict(config=sha(CONFIG),source=source,members=members,audit=audit,calibration=sha(calibration_path))
        history=[]; start=0
        if resume:
            saved=pt.restore(model,opt,out/'last.pt',identity); history=saved['history']; start=saved['step']
        else:
            if (out/'initial.pt').exists(): raise ValueError('Trial exists; explicit --resume required')
            pt.atomic_save(pt.checkpoint(model,opt,0,None,identity,[]),out/'initial.pt')
        began=time.time(); job=matrix()['pretrain']
        for step in range(start+1,201):
            model.train(); rates=schedule(opt,step,job)
            if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
            record=pt.update(model,opt,splits['TRAIN'],stats,step,calibration,device)
            record.update(lr=rates,seconds=time.time()-began,
                          peak_allocated=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None)
            history.append(record)
            if step%50==0:
                if packed_state_sha256(model.use.mit)!=reference: raise ValueError('Frozen MiT changed')
                rt.write_json(out/f'validation_{step}.json',validate_reconstruction(model,splits,stats,device,step))
            if step%10==0:
                rt.write_json(out/'curve.json',history)
                pt.atomic_save(pt.checkpoint(model,opt,step,None,identity,history),out/'last.pt')
                print(json.dumps(dict(step=step,gradient_norm=record['gradient_norm'],seconds=record['seconds'])),flush=True)
        rt.write_json(out/'summary.json',dict(complete=True,updates=200,formal_updates=0,seconds=time.time()-began,
            identity=identity,mit_sha256=reference,mit_final_sha256=packed_state_sha256(model.use.mit),
            initial_sha256=sha(out/'initial.pt'),last_sha256=sha(out/'last.pt'),
            limitation='Engineering only; formal jobs reinitialize from locked v3/MiT and seed'))


def protected_sources():
    return sorted(list((ROOT/'src/landslide_seg').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('test_use_v4*.py')))


def verify_release(release):
    release=Path(release); manifest=rt.read(release/'manifest.json')
    for name,digest in manifest['files'].items():
        if sha(ROOT/name)!=digest or sha(release/'snapshot'/name)!=digest: raise ValueError('Release changed: '+name)
    return manifest


def freeze(name):
    trial=rt.read(BASE/'engineering/trial200/summary.json'); verification=rt.read(BASE/'engineering/verification.json')
    if trial.get('updates')!=200 or not trial.get('complete') or verification.get('status')!='PASS':
        raise ValueError('Completed 200 updates and verified tests required')
    release=BASE/'releases'/name; release.mkdir(parents=True,exist_ok=False)
    files=protected_sources()+list(CONFIG.parent.glob('*'))+list((BASE/'engineering').glob('*'))
    files+=list((BASE/'engineering/trial200').glob('*'))+list((ROOT/'manifests/v4r1').glob('*'))
    files+=list(MIT.glob('*'))+[V3,ROOT/'docs/assets/research/research_policy.json',ROOT/'docs/USE_V4_MIT_EXECUTION_HANDOFF.md',
        ROOT/'outputs/use_v3/engineering/calibration.json',ROOT/'outputs/use_independent_frozen/cache_audit_v1/members.jsonl',
        ROOT/'outputs/use_independent_frozen/cache_audit_v1/acceptance.json',ROOT/'outputs/use/pretraining/resolved_config.json']
    hashes={}
    for path in sorted(set(p for p in files if p.is_file() and p.name!='READY_FOR_AUDIT.json')):
        relative=path.relative_to(ROOT).as_posix(); hashes[relative]=sha(path)
        target=release/'snapshot'/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
    rt.write_json(release/'manifest.json',dict(status='FROZEN_FOR_REVIEW',files=hashes,environment=environment,
        jobs=matrix(),engineering_updates=200,formal_updates=0,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf-8').strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,encoding='utf-8')),True)
    verify_release(release)
    rt.write_json(release/'READY_FOR_AUDIT.json',dict(status='READY_FOR_AUDIT',release_sha256=sha(release/'manifest.json'),formal_updates=0),True)
    print(str(release),sha(release/'manifest.json'))


def statistics_identity(config,stats):
    paths=[ROOT/'outputs/use/pretraining/resolved_config.json',ROOT/config['segformer']['data']['stats'],
           BASE/'engineering/calibration.json',ROOT/'outputs/use_v3/engineering/calibration.json']
    return dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        actual_input_stats_sha256=hashlib.sha256(json.dumps(stats,sort_keys=True,separators=(',',':')).encode()).hexdigest())


@contextmanager
def guarded_device(device):
    lock=rt.claim_card(rt.physical_index(device)[0]) if torch.device(device).type=='cuda' else None
    try:
        yield rt.gpu_available(device)
    finally:
        if lock is not None: lock.unlink(missing_ok=True)


def check_frozen_boundary(model,opt,mode):
    encoder=model.use.mit if mode=='pretrain' else model.use.encoder
    if any(m.training for m in encoder.modules()) or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise RuntimeError('Frozen mode/parameter/gradient boundary violated')
    if mode!='pretrain' and model.use.training: raise RuntimeError('Frozen USE wrapper is training')
    frozen={id(p) for p in encoder.parameters()}
    if any(id(p) in frozen for group in opt.param_groups for p in group['params']):
        raise RuntimeError('Optimizer contains frozen parameters')
    return encoder


def verify_frozen(model,opt,mode,reference):
    digest=packed_state_sha256(check_frozen_boundary(model,opt,mode))
    if digest!=reference: raise RuntimeError('Frozen state changed')
    return digest


def save_checked(model,opt,step,best,identity,history,path,reference,mode):
    before=verify_frozen(model,opt,mode,reference)
    pt.atomic_save(pt.checkpoint(model,opt,step,best,identity,history),path)
    after=verify_frozen(model,opt,mode,reference)
    rt.write_json(path.with_suffix('.freeze.json'),dict(step=step,checkpoint_sha256=sha(path),
        frozen_before_sha256=before,frozen_after_sha256=after))


def segment_update(model,optimizer,dataset,step,config,device):
    # Same numerical update as v2; full frozen hashes belong at checkpoints, not every step.
    optimizer.zero_grad(set_to_none=True)
    indices=rt.np.random.default_rng(1234+step).integers(len(dataset),size=8)
    dataset.set_epoch(step); losses=[]
    for start in range(0,8,2):
        batch=rt.collate([dataset[int(i)] for i in indices[start:start+2]])
        pred=model(batch['image'].to(device))
        loss=rt._seg_loss(config,pred,batch['mask'].to(device),batch['valid'].to(device))
        (loss/4).backward(); losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm): raise ValueError('Nonfinite gradients')
    optimizer.step()
    return dict(loss=float(rt.np.mean(losses)),sample_ids=[dataset.records[int(i)]['sample_id'] for i in indices],
        frozen_sha256=model.use.reference_sha256,gradient_norm=float(norm))


def run_job(job_id,device,release,marker,resume=False):
    pt.deterministic()
    verify_release(release); proof=rt.read(marker)
    release_sha=sha(Path(release)/'manifest.json')
    if proof.get('status')!='READY_FOR_TRAINING' or proof.get('release_sha256')!=release_sha:
        raise ValueError('B independent READY_FOR_TRAINING for this exact release required')
    preflight(); job=matrix()[job_id]; out=BASE/'runs'/job_id; out.mkdir(parents=True,exist_ok=True)
    splits,stats,audit,members=pt.inputs('multisensor'); config=yaml.safe_load(CONFIG.read_text())
    with guarded_device(device) as gate:
        rt.write_json(out/'device_gate.json',gate)
        source=None
        if job['mode']=='pretrain': model=pt.initialize(device)
        else:
            source=BASE/'runs/pretrain'/('best_unlabeled.pt' if job['mode']=='segmentation' else f"u{job['point']}.pt")
            encoder=load_encoder(source)
            if job['mode']=='probe': model=pt.initialize(device,job['task'],encoder)
            else:
                from landslide_seg.models.use_v4_reconstruction import Segmenter
                torch.manual_seed(1235); model=Segmenter(encoder).to(device)
        opt=pt.optimizer(model) if job['mode']!='segmentation' else torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=6e-4,weight_decay=.01)
        identity=dict(release_sha256=release_sha,job=job,source=sha(source) if source else None,
            members=members,audit=audit,config_sha256=sha(CONFIG),statistics=statistics_identity(config,stats),overrides=[])
        reference=packed_state_sha256(check_frozen_boundary(model,opt,job['mode']))
        history=[]; start=0; best=None
        if resume:
            saved=pt.restore(model,opt,out/'last.pt',identity); history=saved['history']; start=saved['step']; best=saved['best']
            verify_frozen(model,opt,job['mode'],reference)
        else:
            if (out/'initial.pt').exists(): raise ValueError('Output exists; explicit resume required')
            save_checked(model,opt,0,None,identity,[],out/'initial.pt',reference,job['mode'])
        if job['mode']=='segmentation': train,val=rt.datasets(config)
        calibration=rt.read(BASE/'engineering/calibration.json')
        for step in range(start+1,job['steps']+1):
            model.train(); check_frozen_boundary(model,opt,job['mode'])
            rates=schedule(opt,step,job)
            cuda=torch.device(device).type=='cuda'
            if cuda:
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            began=time.time()
            if job['mode']=='pretrain': record=pt.update(model,opt,splits['TRAIN'],stats,step,calibration,device)
            elif job['mode']=='probe':
                from run_use_v3 import probe_update
                record=probe_update(model,opt,splits['TRAIN'],stats,step,job['task'],device)
            else: record=segment_update(model,opt,train,step,config,device)
            if cuda: torch.cuda.synchronize(device)
            record.update(step=step,lr=rates,seconds=time.time()-began,
                peak_allocated=torch.cuda.max_memory_allocated(device) if cuda else None,
                peak_reserved=torch.cuda.max_memory_reserved(device) if cuda else None)
            check_frozen_boundary(model,opt,job['mode']); history.append(record)
            if job['mode']=='segmentation' and step%500==0:
                evaluation=rt.downstream_evaluate(model,val,config,device)
                rt.write_json(out/f'eval_{step}.json',evaluation)
                if best is None or evaluation['event_mean_iou']>best:
                    best=evaluation['event_mean_iou']
                    save_checked(model,opt,step,best,identity,history,out/'best.pt',reference,job['mode'])
            if job['mode']=='pretrain' and step%1000==0:
                rt.write_json(out/f'validation_{step}.json',validate_reconstruction(model,splits,stats,device,step))
            if step%100==0 or step==job['steps']:
                if job['mode']=='pretrain' and step in (5000,10000,20000):
                    save_checked(model,opt,step,best,identity,history,out/f'u{step}.pt',reference,job['mode'])
                rt.write_json(out/'curve.json',history)
                save_checked(model,opt,step,best,identity,history,out/'last.pt',reference,job['mode'])
                print(json.dumps(dict(job=job_id,step=step)),flush=True)
        rt.write_json(out/'summary.json',dict(complete=True,updates=job['steps'],best=best,identity=identity,last_sha256=sha(out/'last.pt')))


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','verify','engineering','freeze','run','probe','segment','select','evaluate'])
    p.add_argument('--device',default='cuda:1'); p.add_argument('--resume',action='store_true'); p.add_argument('--release',type=Path); p.add_argument('--marker',type=Path)
    p.add_argument('--name',default='engineering_v1'); p.add_argument('--job',choices=list(matrix())); p.add_argument('--checkpoint',type=Path)
    args=p.parse_args(); torch.set_num_threads(4)
    if args.action=='prepare': prepare()
    elif args.action=='verify':
        if args.release: verify_release(args.release)
        else:
            from verify_use_v4_mit import verify
            verify(args.device)
    elif args.action=='engineering': engineering(args.device,args.resume)
    elif args.action=='freeze': freeze(args.name)
    elif args.action=='select':
        from evaluate_use_v4 import select
        select(BASE/'runs',BASE/'runs/selection.json')
    elif args.action=='evaluate':
        from evaluate_use_v4 import evaluate_job
        evaluate_job(args.job,args.checkpoint,args.device)
    else: run_job('segment' if args.action=='segment' else args.job,args.device,args.release,args.marker,args.resume)


if __name__=='__main__': main()


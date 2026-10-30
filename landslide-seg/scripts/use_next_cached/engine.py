"""Isolated P2 execution, including restart-safe state and exact engineering replay."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import yaml
from landslide_seg.data.use_integrated import IntegratedUSEData
from landslide_seg.data.use_dataset import USESegDataset
from landslide_seg.data.segformer_dataset import read_manifest,load_stats,collate
from landslide_seg.models.use_next_v2 import load_asset
from landslide_seg.models.use_next_v2.readout import state_digest
from landslide_seg.models.use_next_p2.tasks import ScaleTask
from landslide_seg.models.use_next_p2.readout import ScaleReadout
from landslide_seg.training.losses import bce_loss,dice_loss
from landslide_seg.training.segformer_metrics import evaluate_tile_v2,summarize_tiles_v2,summarize_by_event_v2,pixel_average_precision
from use_next_v2.device import claim
from pretrain_use_v2 import atomic_save
from use_next_p2.protocol import audit,sha


def move(asset,device):
    asset=copy.copy(asset)
    for k in ('image','pixel_valid','channel_metadata','metadata_known'):setattr(asset,k,getattr(asset,k).to(device))
    return asset


def stored_state(model):
    # Frozen official weights are restored from the hash-bound originals, not rewritten every update.
    return {k:v for k,v in model.state_dict().items() if not k.startswith(('use.dofa.','use.anysat.'))}


def tensor_digest(state):
    digest=hashlib.sha256()
    for key,value in state.items():digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_model(model,state):
    full=model.state_dict()
    if set(state)!={k for k in full if not k.startswith(('use.dofa.','use.anysat.'))}:raise ValueError('Incomplete trainable/teacher/consumer checkpoint')
    full.update(state);model.load_state_dict(full,strict=True)


def checkpoint(model,opt,step,history,identity,best,device):
    return dict(model=stored_state(model),optimizer=opt.state_dict(),step=step,history=history,identity=identity,best=best,
        rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)


def restore(model,opt,path,identity,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['identity']!=identity:raise ValueError('Resume identity changed')
    restore_model(model,saved['model']);opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng']);np.random.set_state(saved['numpy_rng'])
    if saved['cuda_rng'] is not None:torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved


@torch.no_grad()
def evaluate(model,dataset,device):
    model.eval();rows=[];probabilities=[];targets=[]
    for i in range(len(dataset)):
        sample=dataset[i];batch=collate([sample]);p=model(batch['image'].to(device))[0,0].sigmoid().cpu().numpy()
        r=evaluate_tile_v2(p,sample['mask'].numpy(),sample['valid'].numpy(),sample_id=sample['sample_id'],event_id=sample['event_id'])
        rows.append(r)
        valid=sample['valid'].numpy().astype(bool)
        probabilities.append(p[valid]);targets.append(sample['mask'].numpy().astype(bool)[valid])
    by_event=summarize_by_event_v2(rows);score=float(np.mean([r['pixel']['iou'] for r in by_event.values()]))
    for event,values in by_event.items():
        ids=[i for i,r in enumerate(rows) if r['event_id']==event]
        values['pixel_ap']=pixel_average_precision(np.concatenate([probabilities[i] for i in ids]),np.concatenate([targets[i] for i in ids]))
    model.train();return dict(event_mean_iou=score,by_event=by_event,overall=summarize_tiles_v2(rows),tiles=rows,
                             pixel_ap=pixel_average_precision(np.concatenate(probabilities),np.concatenate(targets)))


def main(task_factory=None,phase_key='use_next_p2',audit_fn=audit,marker_status='READY_P2',source_steps=6000):
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--out',required=True);ap.add_argument('--engineering',action='store_true');ap.add_argument('--steps',type=int,default=2)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--source');ap.add_argument('--release');ap.add_argument('--marker','--ready-p2',dest='ready_p2')
    args=ap.parse_args();torch.set_num_threads(1 if args.device=='cpu' else 4);torch.manual_seed(1234);torch.use_deterministic_algorithms(True)
    if args.device=='cpu':
        torch.backends.mkldnn.enabled=False
        torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=yaml.safe_load(Path(args.config).read_text());v=cfg[phase_key];receipt=audit_fn(args.config)
    if args.engineering:
        if not 1<=args.steps<=100:raise ValueError('Engineering budget exceeded')
    else:
        if not args.device.startswith('cuda:'):raise ValueError('Formal workflow requires a verified UUID-locked CUDA device')
        if not args.release or not args.ready_p2:raise ValueError('Independent READY_P2 required')
        from use_next_p2.verify import verify
        digest=verify(args.release)
        marker=json.loads(Path(args.ready_p2).read_text())
        if marker.get('status')!=marker_status or marker.get('release_sha256')!=digest:raise ValueError('Independent acceptance mismatch')
        if marker.get('data_manifest_sha256')!=v['data_manifest_sha256']:raise ValueError('Marker data binding missing')
        if marker.get('u0_sha256')!=sha(ROOT/v['u0']) or marker.get('schedule_sha256')!=sha(ROOT/v['schedule']):raise ValueError('Marker initialization/schedule binding missing')
        if marker.get('config_sha256',{}).get(Path(args.config).resolve().relative_to(ROOT).as_posix())!=sha(args.config):raise ValueError('Marker config binding missing')
        previous={'P2':'P1','P3':'P2','P5':'P3'}[v['stage']]
        previous_status=marker.get('previous_stage_status',previous+'_ACCEPTED')
        if previous_status not in (previous+'_ACCEPTED', 'P2_USER_STOPPED' if v['stage']=='P3' else previous+'_ACCEPTED'):raise ValueError('Invalid previous-stage status')
        previous_file=ROOT/('outputs/use_next_v2/audit/P2_USER_STOP_ACCEPTANCE.json' if previous_status=='P2_USER_STOPPED' else f'outputs/use_next_v2/audit/{previous}_FINAL_ACCEPTANCE.json')
        if json.loads(previous_file.read_text()).get('status')!=previous_status or sha(previous_file)!=marker.get('previous_acceptance_sha256'):
            raise ValueError('Previous stage not accepted or binding missing')
        args.steps=v['steps']
    out=Path(args.out)
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    (out/'status.json').write_text(json.dumps(dict(status='INITIALIZING',pid=os.getpid(),command=sys.argv,budget=args.steps,updated=time.time())))
    context=claim(args.device) if args.device.startswith('cuda') else nullcontext('cpu')
    with context as uuid:
        stats=load_stats(ROOT/cfg['segformer']['data']['stats'])
        if v['task']=='pretrain':
            dataset=IntegratedUSEData(ROOT/v['data_release']);index={r['id']:i for i,r in enumerate(dataset.rows)}
            schedule=[json.loads(l) for l in (ROOT/v['schedule']).read_text().splitlines()]
            calibration=json.loads((ROOT/v['calibration']).read_text())
            model=task_factory(ROOT,v) if task_factory else ScaleTask(ROOT,v['mode'])
            u0=torch.load(ROOT/v['u0'],map_location='cpu',weights_only=True)
            if u0.get('updates')!=0 or u0.get('seed')!=1234:raise ValueError('Encoder initialization is not fresh seed1234 u0')
            model.use.local.load_state_dict(u0['local'],strict=True)
            model.use.teacher_local.load_state_dict(u0['local'],strict=True)
            task_parameters=model.task_parameters() if hasattr(model,'task_parameters') else model.predictor.parameters()
            opt=torch.optim.AdamW([dict(params=model.use.local.parameters(),lr=.0003),dict(params=task_parameters,lr=.0006)],weight_decay=.01,foreach=False,fused=False)
            source_hash=None
        else:
            path=Path(args.source or ROOT/v['frozen_source'])
            source=torch.load(path,map_location='cpu',weights_only=False)
            if not args.engineering and (source['step']!=source_steps or source['identity']['engineering'] or source['identity']['mode']!=v['mode']
                or source['identity'].get('objective','scale')!=v.get('source_objective','scale') or source['identity']['data_manifest_sha256']!=v['data_manifest_sha256']
                or source['identity']['task']!='pretrain' or source['identity'].get('stage')!=('P2' if v['stage']=='P2' else 'P3')):
                raise ValueError('Consumer requires own same-data fixed formal endpoint')
            local={k[len('use.local.'):]:value for k,value in source['model'].items() if k.startswith('use.local.')}
            model=ScaleReadout(ROOT,v['mode'],stats,local)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0006,weight_decay=.01,foreach=False,fused=False)
            records=read_manifest(ROOT/cfg['segformer']['data']['manifest'])
            train=USESegDataset([r for r in records if r['split']=='train'],stats,augment=True)
            val=USESegDataset([r for r in records if r['split']=='val'],stats,augment=False)
            if (len(train),len(val))!=(2223,669):raise ValueError('Supervised frozen consumer membership differs')
            source_hash=sha(path)
        model.to(args.device);model.train()
        initialization=tensor_digest(stored_state(model) if v['task']=='pretrain' else {k:x for k,x in model.state_dict().items() if not k.startswith('use.')})
        frozen=(state_digest(model.use.dofa),state_digest(model.use.anysat))
        frozen_local=state_digest(model.use.local) if v['task']=='head' else None
        opt_ids={id(p) for g in opt.param_groups for p in g['params']}
        assert not any(id(p) in opt_ids for anchor in (model.use.dofa,model.use.anysat) for p in anchor.parameters())
        if v['task']=='head':assert not any(id(p) in opt_ids for p in model.use.parameters())
        identity=dict(config_sha256=sha(args.config),bindings=receipt['stage_contract']['bindings'],mode=v['mode'],task=v['task'],engineering=args.engineering,
                      stage=v['stage'],objective=v.get('objective',v.get('source_objective','scale')),
                      cpu_threads=torch.get_num_threads(),mkldnn=torch.backends.mkldnn.enabled,mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
                      code_sha256=sha(__file__),release_sha256=sha(Path(args.release)/'manifest.json') if args.release else None,
                      data_manifest_sha256=v['data_manifest_sha256'],source_sha256=source_hash,anchor_hashes=frozen)
        # JSON roundtrip makes tuple/list representation identical across serialized checkpoints.
        identity=json.loads(json.dumps(identity));history=[];start=0;best=-1.
        if args.resume:
            saved=restore(model,opt,out/'last.pt',identity,args.device);start=saved['step'];history=saved['history'];best=saved['best']
        executed_updates=0
        def update(step):
            nonlocal executed_updates
            warm=v['warmup_updates'];ratio=min((step+1)/warm,1.)*.5*(1+math.cos(math.pi*max(0,step+1-warm)/(v['steps']-warm)))
            for i,g in enumerate(opt.param_groups):g['lr']=(.0003 if v['task']=='pretrain' and i==0 else .0006)*ratio
            opt.zero_grad(set_to_none=True);losses=[];exposures=[]
            if v['task']=='pretrain':
                for begin in range(0,8,2):
                    terms=[]
                    for cursor in range(step*8+begin,step*8+begin+2):
                        event=schedule[cursor];asset=move(load_asset(dataset,index[event['asset_id']]),args.device)
                        loss,detail=model.loss(asset,event['geometry_mask_seed'],event['band_mask_seed'],calibration)
                        exposures.append(dict(asset_id=event['asset_id'],cursor=cursor,**detail))
                        if loss is not None:terms.append(loss);losses.append(float(loss.detach()))
                    if terms:(sum(terms)/8).backward()
            else:
                train.set_epoch(step);ids=np.random.default_rng(1234+step).integers(len(train),size=8)
                for begin in range(0,8,2):
                    batch=collate([train[int(i)] for i in ids[begin:begin+2]])
                    logits=model(batch['image'].to(args.device))[:,0]
                    loss=.5*bce_loss(logits,batch['mask'].to(args.device),batch['valid'].to(args.device))+.5*dice_loss(logits,batch['mask'].to(args.device),batch['valid'].to(args.device))
                    (loss/4).backward();losses.append(float(loss.detach()))
                exposures=[dict(sample_id=train.records[int(i)]['sample_id']) for i in ids]
            if not losses:raise ValueError('No supported loss in fixed batch; do not silently resample')
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm) or norm==0:raise ValueError('Nonfinite or zero gradient')
            opt.step()
            executed_updates+=1
            if hasattr(model,'after_optimizer'):model.after_optimizer(step,v['steps'])
            return dict(step=step+1,loss=float(sum(losses)/8 if v['task']=='pretrain' else np.mean(losses)),
                        available_loss_mean=float(np.mean(losses)),loss_observations=len(losses),gradient_norm=float(norm),exposures=exposures)
        began=time.perf_counter()
        if args.device.startswith('cuda:'):torch.cuda.reset_peak_memory_stats(args.device)
        try:
            for step in range(start,args.steps):
                history.append(update(step))
                if not args.engineering and v['task']=='head' and (step+1)%500==0:
                    (out/'status.json').write_text(json.dumps(dict(status='EVALUATING',pid=os.getpid(),step=step+1,budget=args.steps,updated=time.time(),uuid=uuid)))
                    result=evaluate(model,val,args.device);(out/f'evaluation_{step+1}.json').write_text(json.dumps(result))
                    if result['event_mean_iou']>best:
                        best=result['event_mean_iou'];atomic_save(checkpoint(model,opt,step+1,history,identity,best,args.device),out/'best.pt')
                atomic_save(checkpoint(model,opt,step+1,history,identity,best,args.device),out/'last.pt')
                if not args.engineering and step+1 in v.get('milestones',[]):
                    atomic_save(checkpoint(model,opt,step+1,history,identity,best,args.device),out/f'u{step+1}.pt')
                if args.engineering and step==0:atomic_save(checkpoint(model,opt,1,history,identity,best,args.device),out/'step1.pt')
                (out/'curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
                (out/'status.json').write_text(json.dumps(dict(status='RUNNING',pid=os.getpid(),step=step+1,budget=args.steps,command=sys.argv,updated=time.time(),uuid=uuid)))
            replay=False
            if args.engineering and args.steps==2:
                expected=state_digest(model);optim=copy.deepcopy(opt.state_dict());expected_state=copy.deepcopy(stored_state(model))
                restore(model,opt,out/'step1.pt',identity,args.device);row=update(1)
                actual=state_digest(model)
                differences={k:dict(count=int((v!=stored_state(model)[k]).sum()),maximum=float((v-stored_state(model)[k]).abs().max()))
                             for k,v in expected_state.items() if not torch.equal(v,stored_state(model)[k])}
                (out/'replay_comparison.json').write_text(json.dumps(dict(expected_row=history[1],actual_row=row,expected_sha256=expected,actual_sha256=actual,
                    state_differences=differences,anchor_before=frozen,anchor_after=(state_digest(model.use.dofa),state_digest(model.use.anysat))),indent=2))
                assert row==history[1] and actual==expected
                for k,values in optim['state'].items():
                    for field,value in values.items():assert torch.equal(value.cpu(),opt.state_dict()['state'][k][field].cpu())
                replay=True
            assert frozen==(state_digest(model.use.dofa),state_digest(model.use.anysat))
            if frozen_local is not None:assert frozen_local==state_digest(model.use.local)
            summary=dict(complete=True,identity=identity,updates=args.steps,formal_updates=0 if args.engineering else args.steps,
                         successful_updates_including_replay=executed_updates,exact_disk_replay=replay,frozen_unchanged=True,
                         seconds=time.perf_counter()-began,uuid=uuid,final_local_sha256=state_digest(model.use.local),
                         initialization_sha256=initialization,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda:') else None)
            (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'status.json').write_text(json.dumps(dict(status='COMPLETED',step=args.steps,pid=os.getpid())))
            print(json.dumps(summary))
        except BaseException as error:
            (out/'failure.json').write_text(json.dumps(dict(error=repr(error),successful_updates_this_session=executed_updates,last_saved=(out/'last.pt').exists(),command=sys.argv)))
            raise


if __name__=='__main__':main()

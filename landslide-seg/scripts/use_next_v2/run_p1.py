"""Release-bound P1 execution. Formal runs require independent P0 acceptance."""
import argparse
import copy
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
from landslide_seg.models.use_next_v2.readout import NativeReadout
from landslide_seg.training.losses import bce_loss,dice_loss
from landslide_seg.data.segformer_dataset import read_manifest,load_stats,collate
from landslide_seg.data.use_dataset import USESegDataset
from check_research_protocol import audit_config
from run_use_v2 import downstream_evaluate,atomic_save
from device import claim


def datasets(cfg):
    data=cfg['segformer']['data'];stats=load_stats(ROOT/data['stats']);rows=read_manifest(ROOT/data['manifest'])
    return [USESegDataset([r for r in rows if r['split']==split],stats,augment=split=='train',preload=False)
            for split in ('train','val')],stats


def update(model,opt,train,step,device,cfg):
    train.set_epoch(step);opt.zero_grad(set_to_none=True)
    indices=np.random.default_rng(1234+step).integers(len(train),size=8)
    lr=.0006*min((step+1)/150,1.)*(.5*(1+math.cos(math.pi*max(0,step+1-150)/2850)))
    for group in opt.param_groups:group['lr']=lr
    losses=[]
    micro=cfg['use_next_v2']['microbatch']
    for start in range(0,8,micro):
        batch=collate([train[int(i)] for i in indices[start:start+micro]]);prediction=model(batch['image'].to(device))[:,0]
        target=batch['mask'].to(device);valid=batch['valid'].to(device)
        loss=.5*bce_loss(prediction,target,valid)+.5*dice_loss(prediction,target,valid)
        (loss/(8/micro)).backward();losses.append(float(loss.detach()))
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm) or norm==0:raise ValueError('Bad gradient')
    opt.step();model.verify(opt)
    return dict(step=step+1,loss=float(np.mean(losses)),lr=lr,gradient_norm=float(norm),sample_indices=indices.tolist())


def checkpoint(model,opt,step,history,config_sha,best):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,history=history,config_sha256=config_sha,best=best,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--out',required=True);ap.add_argument('--engineering',action='store_true');ap.add_argument('--steps',type=int,default=2)
    ap.add_argument('--ready-p0');ap.add_argument('--release');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());receipt=audit_config(args.config)
    if args.engineering:
        if not 1<=args.steps<=100:raise ValueError('Engineering budget exceeded')
    else:
        from preflight import verify
        if not args.ready_p0 or not args.release:raise ValueError('Independent P0 marker and bound release required')
        verified=verify(args.release)
        marker=json.loads(Path(args.ready_p0).read_text())
        from audit_data import sha
        if not verified.get('integrity_pass') or not verified.get('independent_audit_ready') or marker.get('status')!='READY_P0' or marker.get('release_sha256')!=sha(Path(args.release)/'release_manifest.json'):
            raise ValueError('Independent acceptance missing or mismatched')
        args.steps=3000
    out=Path(args.out)
    if not args.resume:out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1234);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    with claim(args.device) as uuid:
        (train,val),stats=datasets(cfg)
        model=NativeReadout(ROOT,cfg['use_next_v2']['anchor'],stats).to(args.device)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0006,weight_decay=.01)
        model.verify(opt);history=[];start=0;best=-1.
        if args.resume:
            saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            if saved['config_sha256']!=receipt['effective_config_sha256']:raise ValueError('Resume config changed')
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);start=saved['step'];history=saved['history'];best=saved['best']
            torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']]);np.random.set_state(saved['numpy_rng'])
        torch.cuda.reset_peak_memory_stats(args.device);began=time.perf_counter()
        for step in range(start,args.steps):
            history.append(update(model,opt,train,step,args.device,cfg))
            if not args.engineering and (step+1)%500==0:
                result=downstream_evaluate(model,val,cfg,args.device)
                (out/f'evaluation_{step+1}.json').write_text(json.dumps(result))
                if result['event_mean_iou']>best:
                    best=result['event_mean_iou'];atomic_save(checkpoint(model,opt,step+1,history,receipt['effective_config_sha256'],best),out/'best.pt')
            atomic_save(checkpoint(model,opt,step+1,history,receipt['effective_config_sha256'],best),out/'last.pt')
            if args.engineering and step==0:
                atomic_save(checkpoint(model,opt,step+1,history,receipt['effective_config_sha256'],best),out/'step1.pt')
            (out/'curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        replay=False
        if args.engineering and args.steps==2:
            from landslide_seg.models.use_next_v2.readout import state_digest
            expected=state_digest(model);expected_opt=copy.deepcopy(opt.state_dict())
            saved=torch.load(out/'step1.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng'])
            replay_row=update(model,opt,train,1,args.device,cfg)
            assert expected==state_digest(model) and replay_row==history[1]
            for k,v in expected_opt['state'].items():
                for field,value in v.items():assert torch.equal(value.cpu(),opt.state_dict()['state'][k][field].cpu())
            replay=True
        summary=dict(updates=args.steps,formal_updates=0 if args.engineering else args.steps,uuid=uuid,
            peak_bytes=torch.cuda.max_memory_allocated(args.device),seconds=time.perf_counter()-began,protocol=receipt,
            exact_disk_replay=replay,successful_updates_including_replay=args.steps+int(replay),microbatch=cfg['use_next_v2']['microbatch'],
            frozen_sha256=model.verify(),parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()

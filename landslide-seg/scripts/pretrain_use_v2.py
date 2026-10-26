"""USE v2 objectives and deterministic data streams, independent of task labels."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import os
import random

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F
from landslide_seg.models.use_v2 import initialize_pair,canonical_context,LegacyPyramid
from landslide_seg.models.frozen_use import state_sha256
from landslide_seg.data.optical_observation import OpticalObservation,collate_optical
from pretrain_use_independent import inputs,masked_mae
from audit_use_independent_data import EXTERNAL,sha


class Reconstruction(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.use=encoder
        self.context_projection=nn.ModuleList([nn.Conv2d(32,32,1) for _ in range(4)])
        self.decode=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1))
        # Unknown targets are distinguished by visible-only mean/std summaries.
        self.condition=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,33))

    def forward(self,observation,metadata,known,summary):
        f=self.use(observation)
        context=sum(F.interpolate(proj(canonical_context(t)),size=f['fine'].shape[-2:],mode='bilinear',align_corners=False)
                    for proj,t in zip(self.context_projection,f['context']))
        feature=self.decode(torch.cat((f['fine'],context),1))
        cond=torch.cat((torch.log1p(metadata),known.float(),summary),-1)
        weights=self.condition(cond)
        return torch.einsum('bcd,bdhw->bchw',weights[...,:32],feature)+weights[...,32,None,None]


def initialize(variant,device='cpu'):
    deterministic()
    pair=initialize_pair()
    torch.manual_seed(1235)
    return Reconstruction(pair[variant=='attention']).to(device)


def deterministic():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def encoder(candidate,checkpoint=None,device='cpu'):
    if candidate=='old':
        from pretrain_use_independent import initialize as old_initialize
        path=ROOT/'outputs/use_independent_frozen/pretraining/multisensor/best.pt'
        summary=json.loads((path.parent/'summary.json').read_text())
        if sha(path)!=summary['best_sha256']: raise ValueError('Old selected state changed')
        model=old_initialize('multisensor','cpu')
        model.use.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['use'],strict=True)
        result=LegacyPyramid(model.use)
    else:
        result=initialize(candidate).use
        if checkpoint:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            result.load_state_dict(saved['use'],strict=True)
    return result.to(device)


class RecoveryProbe(Reconstruction):
    def __init__(self,encoder,task):
        from landslide_seg.models.frozen_use import FrozenUSE
        torch.manual_seed(1235)
        super().__init__(FrozenUSE(encoder))
        self.task=task
        self.context_projection=nn.ModuleList([])
        self.projection=nn.Conv2d(32,32,1)

    def forward(self,observation,metadata,known,summary):
        f=self.use(observation)
        if self.task=='fine':
            feature=f['fine']
        else:
            feature=sum(F.interpolate(canonical_context(t),size=f['fine'].shape[-2:],mode='bilinear',align_corners=False)
                        for t in f['context'])/4
        # Same trainable capacity in both tasks; no input-value bypass.
        feature=self.projection(feature)
        feature=self.decode(torch.cat((feature,torch.zeros_like(feature)),1))
        weights=self.condition(torch.cat((torch.log1p(metadata),known.float(),summary),-1))
        return torch.einsum('bcd,bdhw->bchw',weights[...,:32],feature)+weights[...,32,None,None]


def load_rows(rows,stats,device,augmentation_seed=None):
    observations=[]
    for index,row in enumerate(rows):
        path=EXTERNAL/row['file']
        if sha(path)!=row['sha256']: raise ValueError('Cache hash changed')
        with np.load(path,allow_pickle=False) as arrays:
            x,v,m,k=(arrays[key].copy() for key in ('x','pixel_valid','metadata','known'))
        mean,std=(np.asarray(stats[row['sensor']][key],dtype=np.float32)[:,None,None] for key in ('mean','std'))
        x=np.where(v,(x-mean)/std,0).astype(np.float32)
        transform=np.array([row['transform'][:3],row['transform'][3:],[0,0,1]],dtype=float)
        if augmentation_seed is not None:
            from landslide_seg.data.use_dataset import augmentation_grid_map
            from landslide_seg.data.segformer_dataset import _augment
            seed=augmentation_seed+index
            mapping,_=augmentation_grid_map(x.shape[-2],x.shape[-1],seed,True)
            x,_,_= _augment(x,np.zeros(x.shape[-2:]),np.zeros(x.shape[-2:]),np.random.default_rng(seed))
            v,_,_= _augment(v,np.zeros(v.shape[-2:]),np.zeros(v.shape[-2:]),np.random.default_rng(seed))
            transform=transform@mapping
        observations.append(OpticalObservation(torch.from_numpy(x),tuple(row['band_ids']),torch.from_numpy(m),
            torch.from_numpy(k),torch.ones(len(x),dtype=torch.bool),torch.from_numpy(v),
            dict(crs=row['crs'],transform=transform[:2].flatten().tolist())))
    return collate_optical(observations).to(device)


def sample_rows(split,step):
    rng=np.random.default_rng(1234+step)
    return [pool[i] for sensor,count in (('s2',6),('landsat',2))
            for pool in [split[sensor]] for i in rng.integers(len(pool),size=count)]


def task_view(original,task,seed,block=32,metadata_mode='mixed'):
    rng=np.random.default_rng(seed)
    metadata_rng=np.random.default_rng(seed+10000000)
    o={k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in original.items()}
    x=o['x']; b,c,h,w=x.shape
    original_valid=o['pixel_valid'] & o['band_valid'][:,:,None,None]
    hidden=torch.zeros_like(original_valid)
    # Stream is local and never consumes model RNG.
    for i in range(b):
        choice=metadata_rng.choice(3,p=(.5,.25,.25)) if metadata_mode=='mixed' else {'known':0,'partial':1,'unknown':2}[metadata_mode]
        if choice==1:
            drop=torch.as_tensor(metadata_rng.random((c,3))<.5,device=x.device)
            o['known_mask'][i]&=~drop
        elif choice==2: o['known_mask'][i]=False
        o['band_metadata'][i].masked_fill_(~o['known_mask'][i],0)
        if task=='spectral':
            indices=torch.where(o['band_valid'][i])[0].cpu().numpy()
            chosen=rng.permutation(indices)[:len(indices)//2]
            for j in chosen:
                if o['known_mask'][i,j,0]: hidden[i,j]=original_valid[i,j]
        elif task=='fine':
            mask=torch.as_tensor(rng.random((h,w))<.25,device=x.device)
            hidden[i]=original_valid[i]&mask
        elif task=='context':
            size=min(block,h-1,w-1)
            if size<1: raise ValueError('Spatial masking needs H,W >=2')
            y=int(rng.integers(h-size+1)); z=int(rng.integers(w-size+1))
            hidden[i,:,y:y+size,z:z+size]=original_valid[i,:,y:y+size,z:z+size]
        else: raise ValueError(task)
    visible=original_valid & ~hidden
    o['pixel_valid']=visible
    o['x']=torch.where(visible,x,0)
    count=visible.sum((-1,-2)).clamp_min(1)
    mean=o['x'].sum((-1,-2))/count
    var=(torch.where(visible,o['x']-mean[...,None,None],0).square().sum((-1,-2))/count)
    summary=torch.stack((mean,var.sqrt()),-1)
    return o,hidden,summary


def objective(model,original,task,seed,coefficient=1.):
    o,hidden,summary=task_view(original,task,seed)
    usable=hidden.flatten(1).any(-1)
    if not usable.any(): return None
    # Some metadata-unknown samples have no identifiable spectral target.
    selected={k:v[usable] if torch.is_tensor(v) else v for k,v in o.items()}
    pred=model(selected,o['band_metadata'][usable],o['known_mask'][usable],summary[usable])
    coefficients=torch.as_tensor(coefficient,device=pred.device).expand(len(usable))[usable]
    loss=torch.stack([masked_mae(pred[i:i+1],original['x'][usable][i:i+1],hidden[usable][i:i+1])/coefficients[i]
                      for i in range(len(pred))]).mean()
    return loss,pred,hidden[usable],original['x'][usable]


def lr(step,total=20000,warmup=1000):
    return 6e-4*(step/warmup if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total-warmup))))


def atomic_save(payload,path):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary); os.replace(temporary,path)


def rng_state():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu()); np.random.set_state(state['numpy']); random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda']])


def checkpoint(model,optimizer,step,best,identity,history):
    underlying=model.use.encoder if hasattr(model.use,'encoder') else model.use
    return dict(model=model.state_dict(),use=underlying.state_dict(),use_sha256=state_sha256(underlying),
                optimizer=optimizer.state_dict(),rng=rng_state(),step=step,best=best,
                identity=identity,history=history,model_sha256=state_sha256(model))


def restore(model,optimizer,path,identity):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['identity']!=identity: raise ValueError('Resume release/config/data/initialization identity mismatch')
    model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer'])
    if state_sha256(model)!=saved['model_sha256']: raise ValueError('Restored model hash differs')
    restore_rng(saved['rng'])
    if hasattr(model.use,'verify'): model.use.verify(optimizer)
    return saved


def ssl_update(model,optimizer,split,stats,step,calibration,task=None,device='cpu'):
    rows=sample_rows(split,step)
    optimizer.zero_grad(set_to_none=True); losses=[]
    tasks=(task,) if task else ('spectral','fine','context')
    exposure=[]
    for micro in range(4):
        current=rows[micro*2:micro*2+2]
        batch=load_rows(current,stats,device,augmentation_seed=1234+step*8+micro*2)
        for ti,name in enumerate(tasks):
            coefficients=[calibration['coefficients'][r['sensor']][name] for r in current]
            result=objective(model,batch,name,1234+step*100+micro*3+('spectral','fine','context').index(name),coefficients)
            if result is not None:
                (result[0]/(4*len(tasks))).backward(); losses.append(float(result[0].detach()))
                exposure.append(dict(task=name,usable_samples=int(result[1].shape[0]),hidden_pixels=int(result[2].sum())))
    if not losses: raise ValueError('No usable targets')
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm): raise ValueError('Nonfinite gradients')
    optimizer.step()
    if hasattr(model.use,'verify'): model.use.verify(optimizer)
    return dict(loss=float(np.mean(losses)),files=[r['file'] for r in rows],
                sensors=[r['sensor'] for r in rows],views=exposure,gradient_norm=float(norm))


def engineering(variant,out,steps=100,device='cpu'):
    from run_use_v2 import train
    return train('pretrain_'+variant,out,device,engineering=True,engineering_steps=steps)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('mean','attention'),required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--engineering-steps',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.engineering_steps<=100: parser.error('Engineering budget is 1..100')
    torch.set_num_threads(4)
    print(json.dumps(engineering(args.variant,args.out,args.engineering_steps,args.device)))

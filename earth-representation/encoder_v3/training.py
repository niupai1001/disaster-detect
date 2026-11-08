"""Training components only; no automatic training launcher."""
import copy
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn


class JEPAPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Query receives metadata only: relative position, support, SRF and GSD.
        self.condition=nn.Linear(38,128)
        self.attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.norm=nn.LayerNorm(128)
        self.ffn=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,128))

    def forward(self, student_z, *, relative_xy, support_xy, srf_condition, gsd_xy):
        if bool((support_xy<=0).any() | (gsd_xy<=0).any()):
            raise ValueError('Positive support and GSD required')
        metadata=torch.cat((relative_xy,support_xy.log(),srf_condition,gsd_xy.log()),-1)
        q=self.condition(metadata)[None]
        z=student_z[None]
        q=q+self.attention(q,z,z,need_weights=False)[0]
        return (q+self.ffn(self.norm(q)))[0]


def make_teacher(student):
    teacher=copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_teacher(teacher, student, coefficient):
    if not 0<=coefficient<1:
        raise ValueError('EMA coefficient must lie in [0,1)')
    for target,source in zip(teacher.parameters(),student.parameters(),strict=True):
        target.lerp_(source,1-coefficient)
    for target,source in zip(teacher.buffers(),student.buffers(),strict=True):
        target.copy_(source)


class CAdamW(torch.optim.AdamW):
    """Cautious AdamW: filter the momentum update, not the moment estimates."""
    def __setstate__(self, state):
        super().__setstate__(state)
        # Adam converts counters to tensors on load; this implementation uses integers.
        for parameter_state in self.state.values():
            if 'step' in parameter_state:
                parameter_state['step']=int(parameter_state['step'])

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closure execution is not supported')
        for group in self.param_groups:
            beta1,beta2=group['betas']
            for p in group['params']:
                if p.grad is None: continue
                g=p.grad
                if g.is_sparse: raise ValueError('Sparse gradients unsupported')
                state=self.state[p]
                if not state:
                    state.update(step=0,exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                state['step']+=1
                m,v=state['exp_avg'],state['exp_avg_sq']
                m.lerp_(g,1-beta1); v.mul_(beta2).addcmul_(g,g,value=1-beta2)
                mask=(m*g>0).to(g.dtype)
                mask=mask/mask.mean().clamp_min(1e-3)
                update=(m/(1-beta1**state['step']))/(v.sqrt()/(1-beta2**state['step'])**0.5+group['eps'])
                p.mul_(1-group['lr']*group['weight_decay'])
                p.add_(update*mask,alpha=-group['lr'])


class CheckedPlateau:
    def __init__(self, optimizer, *, threshold, patience=4, factor=.5, minimum=1e-5):
        if threshold<0 or patience<1 or not 0<factor<1 or minimum<=0:
            raise ValueError('Invalid plateau configuration')
        self.optimizer=optimizer
        self.threshold=threshold; self.patience=patience; self.factor=factor; self.minimum=minimum
        self.best=float('inf'); self.wait=0

    def step(self, metric):
        if not np.isfinite(metric): raise ValueError('Nonfinite fixed validation metric')
        before=[g['lr'] for g in self.optimizer.param_groups]
        improved=metric<self.best-self.threshold
        if improved: self.best=metric; self.wait=0
        else: self.wait+=1
        trigger=self.wait>=self.patience
        expected=[max(self.minimum,lr*self.factor) if lr>self.minimum else lr for lr in before] if trigger else before
        if trigger:
            for group,lr in zip(self.optimizer.param_groups,expected): group['lr']=lr
            self.wait=0
        after=[g['lr'] for g in self.optimizer.param_groups]
        if after!=expected: raise RuntimeError('Scheduler LR update failed')
        return dict(metric=metric,best=self.best,threshold=self.threshold,wait=self.wait,before=before,after=after,trigger=trigger)

    def state_dict(self):
        return {key:value for key,value in vars(self).items() if key!='optimizer'}

    def load_state_dict(self, state):
        for key in ('threshold','patience','factor','minimum','best','wait'): setattr(self,key,state[key])


def save_checkpoint(path, *, modules, optimizer, scheduler, logical_update, sampler, frozen_ids):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    devices=sorted({value.device.index for module in modules.values() for value in list(module.parameters())+list(module.buffers()) if value.device.type=='cuda'})
    state=dict(modules={k:v.state_dict() for k,v in modules.items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),logical_update=logical_update,sampler=sampler,frozen_ids=frozen_ids,
               rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda={str(device):torch.cuda.get_rng_state(device) for device in devices} if devices else None))
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def restore_checkpoint(path, *, modules, optimizer, scheduler, frozen_ids):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['frozen_ids']!=frozen_ids: raise ValueError('Frozen data/config/code identity mismatch')
    devices={str(value.device.index) for module in modules.values() for value in list(module.parameters())+list(module.buffers()) if value.device.type=='cuda'}
    cuda_rng=state['rng']['cuda']
    if (cuda_rng is not None and not isinstance(cuda_rng,dict)) or set(cuda_rng or {})!=devices:raise ValueError('CUDA RNG device layout mismatch; no implicit device migration')
    for name,module in modules.items(): module.load_state_dict(state['modules'][name])
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    # torch.load(map_location='cpu') leaves optimizer moments on CPU; move them
    # back to each parameter's device before the next step.
    for group in optimizer.param_groups:
        for parameter in group['params']:
            for key,value in optimizer.state.get(parameter,{}).items():
                if torch.is_tensor(value) and value.shape == parameter.shape and value.device != parameter.device:
                    optimizer.state[parameter][key]=value.to(parameter.device)
    rng=state['rng']; random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    if cuda_rng is not None:
        for device,value in cuda_rng.items():torch.cuda.set_rng_state(value,device=int(device))
    return state

"""Identity-bound atomic checkpoints and transactional validation/restoration."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
import random
import tempfile
import time
import numpy as np
import torch


def atomic_replace(source,target):
    # Windows readers can briefly deny replacement even after the writer closes its file.
    for attempt in range(6):
        try:
            os.replace(source,target)
            return
        except PermissionError:
            if attempt == 5:raise
            time.sleep(.02*2**attempt)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def parameter_sha(model):
    h=hashlib.sha256()
    for name,p in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def snapshot(model,optimizer,state,scheduler=None):
    return copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        scheduler=None if scheduler is None else scheduler.state_dict(),runner=state,rng=rng_state(),
        modes={n:m.training for n,m in model.named_modules()}))


def restore(payload,model,optimizer,state,scheduler=None):
    model.load_state_dict(payload['model'],strict=True); optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(payload['scheduler'])
    state.clear(); state.update(copy.deepcopy(payload['runner']))
    for name,module in model.named_modules():
        module.training=payload['modes'][name]
    restore_rng(payload['rng'])


@contextmanager
def validation_transaction(model,optimizer,state,scheduler=None):
    live=snapshot(model,optimizer,state,scheduler)
    gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    try:
        yield
    finally:
        restore(live,model,optimizer,state,scheduler)
        for n,p in model.named_parameters():
            p.grad=gradients[n]


def save_checkpoint(path,model,optimizer,state,identity,scheduler=None):
    if not {'code','config','registry','catalog','data','sampling'}<=set(identity):
        raise ValueError('Incomplete checkpoint identity')
    if identity['registry']!=model.registry.identity_sha256():
        raise ValueError('Registry identity differs from live model')
    payload=dict(schema='lr_checkpoint_v1.3',identity=copy.deepcopy(identity),**snapshot(model,optimizer,state,scheduler))
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=path.parent,suffix='.tmp'); os.close(fd)
    try:
        torch.save(payload,temp); atomic_replace(temp,path)
    finally:
        if os.path.exists(temp):os.remove(temp)


def load_checkpoint(path,model,optimizer,state,identity,scheduler=None):
    # Only locally produced trusted checkpoints are accepted; never load external pickle files here.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!='lr_checkpoint_v1.3' or payload.get('identity')!=identity:
        raise ValueError('Checkpoint identity mismatch; live state untouched')
    if identity['registry']!=model.registry.identity_sha256():
        raise ValueError('Live registry identity mismatch')
    live=snapshot(model,optimizer,state,scheduler)
    try:
        restore(payload,model,optimizer,state,scheduler)
    except Exception:
        restore(live,model,optimizer,state,scheduler)
        raise

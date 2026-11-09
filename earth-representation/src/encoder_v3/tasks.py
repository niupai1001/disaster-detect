"""Value-independent task plans keyed by logical update, not worker RNG order."""
from dataclasses import dataclass
import hashlib
import random
import torch
from .model import spatial_hide


@dataclass
class Target:
    band: object
    mask: torch.Tensor


@dataclass
class Task:
    kind: str
    visible: list
    targets: list
    metadata: dict
    rectangle: tuple | None = None


def task_rng(seed, update, observation_id, stream):
    identity=f'{seed}:{update}:{observation_id}:{stream}'.encode()
    return random.Random(int.from_bytes(hashlib.sha256(identity).digest()[:16],'big'))


def choose_patch(seed, update, observation_id):
    return task_rng(seed,update,observation_id,'patch').choice((60.,120.))


def spatial_task(observation, rectangle):
    if len(rectangle)!=4 or rectangle[0]>=rectangle[2] or rectangle[1]>=rectangle[3]: raise ValueError('Explicit rectangle required')
    visible=[]; targets=[]; protection=0
    for band in observation.bands:
        hidden,mask=spatial_hide(band,rectangle)
        protection+=int((band.keep() & ~hidden.keep() & ~mask).sum())
        visible.append(hidden); targets.append(Target(band,mask))
    return Task('A',visible,targets,dict(observation.metadata,protection_pixels=protection),tuple(rectangle))


def spectral_task(observation, *, seed, update):
    count=len(observation.bands)
    hidden=hidden_band_indices(count,seed=seed,update=update,observation_id=observation.identifier)
    return _spectral_view(observation,hidden)


def fixed_spectral_task(observation,hidden_band_ids):
    names=[band.record_id for band in observation.bands]
    if len(set(names))!=len(names) or not hidden_band_ids or len(set(hidden_band_ids))!=len(hidden_band_ids) or not set(hidden_band_ids)<set(names):raise ValueError('Explicit unique hidden bands must leave visible support')
    return _spectral_view(observation,{i for i,name in enumerate(names) if name in hidden_band_ids})


def _spectral_view(observation,hidden):
    visible=[b for i,b in enumerate(observation.bands) if i not in hidden]
    targets=[Target(b,b.valid_mask.clone()) for i,b in enumerate(observation.bands) if i in hidden]
    return Task('B1',visible,targets,dict(observation.metadata,hidden_band_ids=[observation.bands[i].record_id for i in sorted(hidden)]))


def hidden_band_indices(count, *, seed,update,observation_id):
    if count<2: raise ValueError('B1 requires at least two bands')
    rng=task_rng(seed,update,observation_id,'B1')
    hidden_count=1 if count==2 or rng.random()<.5 else rng.randint(2,count-1)
    return set(rng.sample(range(count),hidden_count))


def cross_source_task(source, target, admission):
    required=('space','registration','time_difference','surface_change','qa','srf','role_isolation')
    if admission.get('source_id')!=source.identifier or admission.get('target_id')!=target.identifier:
        raise ValueError('Pair admission identity mismatch')
    if any(admission.get('checks',{}).get(key) is not True for key in required):
        raise ValueError('B2 requires V3 pair review, not historical qualified=true')
    if source.metadata['role']!=target.metadata['role']: raise ValueError('Pair crosses frozen roles')
    if len({b.crs for b in source.bands+target.bands})!=1: raise ValueError('Common metric geometry not provided')
    return Task('B2',source.bands,[Target(b,b.valid_mask.clone()) for b in target.bands],dict(source.metadata,target_id=target.identifier,pair_admission=admission))


def curriculum(update, duration=2000):
    if update<0 or duration<=0: raise ValueError('Invalid curriculum progress')
    x=min(update/duration,1.); smooth=x*x*(3-2*x)
    return 1-.75*smooth,.25+.75*smooth

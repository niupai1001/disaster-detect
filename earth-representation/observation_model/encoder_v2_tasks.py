"""Physical A/B tasks and balanced reflectance losses; no training entry point."""
from collections import defaultdict
from dataclasses import dataclass,replace
import hashlib
import math

import numpy as np
import torch

from .encoder_v2 import OpticalImage,target_from_band


@dataclass
class Target:
    values: torch.Tensor
    valid: torch.Tensor
    grid: object
    band_key: str


@dataclass
class Task:
    kind: str
    visible: OpticalImage
    targets: list
    meta: dict


def seed_for(member,kind,draw=0):
    return int(hashlib.sha256(f'0:{member}:{kind}:{draw}'.encode()).hexdigest()[:15],16)


def curriculum(progress):
    if not 0<=progress<=1:raise ValueError('Relative progress outside0..1')
    if progress<=.35:return 1.,0.
    if progress>=.75:return .25,1.
    t=(progress-.35)/.40;r=t*t*(3-2*t)
    return 1-.75*r,r


def spatial_task(image,seed,excluded=(),patch_m=60.,fraction=.60):
    bands=[b for b in image.bands if b.band_key not in excluded]
    coarse=max(bands,key=lambda b:b.support[0]*b.support[1])
    left=max(b.bounds[0] for b in bands);bottom=max(b.bounds[1] for b in bands)
    right=min(b.bounds[2] for b in bands);top=min(b.bounds[3] for b in bands)
    _,_,cx,_,_,cy=coarse.transform
    left=cx+math.ceil((left-cx)/patch_m-1e-10)*patch_m
    top=cy+math.floor((top-cy)/patch_m+1e-10)*patch_m
    nx=int(math.floor((right-left)/patch_m+1e-10));ny=int(math.floor((top-bottom)/patch_m+1e-10))
    if nx*ny<4:raise ValueError('Too little real support for multi-patch A task')
    generator=np.random.default_rng(seed);count=max(2,min(nx*ny-1,round(fraction*nx*ny)))
    selected=np.zeros(nx*ny,dtype=bool);selected[generator.choice(nx*ny,count,replace=False)]=True
    plan=torch.as_tensor(selected.reshape(ny,nx),device=bands[0].values.device)
    visible=[];targets=[];exposure={}
    for band in bands:
        gx,gy=band.support;_,_,x0,_,_,y0=band.transform;h,w=band.values.shape;device=band.values.device
        x=torch.arange(w,device=device,dtype=torch.float64)*gx+x0
        y=y0-torch.arange(h,device=device,dtype=torch.float64)*gy
        c0=torch.floor((x-left)/patch_m+1e-10).long();c1=torch.ceil((x+gx-left)/patch_m-1e-10).long()-1
        r0=torch.floor((top-y)/patch_m+1e-10).long();r1=torch.ceil((top-y+gy)/patch_m-1e-10).long()-1
        hidden=torch.zeros_like(band.valid)
        for rows in [r0,r1]:
            for cols in [c0,c1]:
                inside=(rows[:,None]>=0)&(rows[:,None]<ny)&(cols[None]>=0)&(cols[None]<nx)
                hidden |= inside&plan[rows.clamp(0,ny-1)[:,None],cols.clamp(0,nx-1)[None]]
        # All intersecting native pixels are removed. Border/padding pixels are
        # excluded from supervision, even if deletion extends beyond a patch edge.
        interior=(x[None]>=left-1e-7)&(x[None]+gx<=left+nx*patch_m+1e-7)&(y[:,None]<=top+1e-7)&(y[:,None]-gy>=top-ny*patch_m-1e-7)
        keep=band.valid&~hidden;target=band.valid&hidden&interior
        if not target.any():raise ValueError('No real valid A targets for '+band.band_key)
        visible.append(replace(band,values=torch.where(keep,band.values,0),valid=keep,visible=~hidden))
        targets.append(Target(band.values,target,target_from_band(band),band.band_key))
        exposure[band.band_key]=dict(native_pixels=h*w,source_valid_pixels=int(band.valid.sum()),
            visible_pixels=int(keep.sum()),hidden_native_pixels=int(hidden.sum()),target_valid_pixels=int(target.sum()),
            target_fraction_of_valid=float(target.sum()/band.valid.sum()),gsd_m=band.support)
    return Task('A',OpticalImage(visible,image.product_id),targets,dict(seed=int(seed),patch_m=patch_m,
        selected_patches=np.flatnonzero(selected).tolist(),grid_shape=[ny,nx],grid_origin_xy=[left,top],
        eligible_area_m2=nx*ny*patch_m**2,hidden_area_m2=count*patch_m**2,mask_area_fraction=count/(nx*ny),
        excluded_bands=list(excluded),exposure=exposure))


def band_task(image,key):
    target=next(b for b in image.bands if b.band_key==key)
    visible=[b for b in image.bands if b.band_key!=key]
    if not visible or not target.valid.any():raise ValueError('B needs a visible source and real target')
    return Task('B',OpticalImage(visible,image.product_id),[Target(target.values,target.valid,target_from_band(target),key)],
        dict(target_band=key,target_scope='entire_native_band',exposure={key:dict(native_pixels=target.values.numel(),
            target_valid_pixels=int(target.valid.sum()),visible_target_pixels=0,gsd_m=target.support)}))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def input_hash(task):
    h=hashlib.sha256()
    for b in task.visible.bands:
        for value in [b.band_key,str(b.transform),tensor_hash(b.values),tensor_hash(b.valid),tensor_hash(b.srf)]:h.update(value.encode())
    return h.hexdigest()


def balanced_loss(rows):
    """mean_sensor mean_band mean_observation mean_valid_pixels, never pixel pooling."""
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:groups[row['sensor']][row['band']].append(row['loss'])
    if not groups:raise ValueError('No valid supervision; not a zero loss')
    return torch.stack([torch.stack([torch.stack(values).mean() for values in bands.values()]).mean() for bands in groups.values()]).mean()


def run_task(model,task,observation,statistics,save_predictions=False):
    representation=model.encode(task.visible);rows=[];cases=[]
    for target in task.targets:
        prediction,coverage=model.decoder(representation,target.grid)
        q=target.valid&coverage
        # Missing prediction support cannot quietly reduce the declared task.
        if not torch.equal(q,target.valid):raise ValueError('Declared real targets are not all predicted')
        if not q.any():raise ValueError('No valid target')
        stat=statistics[observation['instrument']+':'+target.band_key]
        error=prediction[q]-target.values[q];loss=(error/stat['std']).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reflectance objective')
        metrics=dict(nmse=float(loss.detach()),rmse=float(error.detach().square().mean().sqrt()),
            mae=float(error.detach().abs().mean()),bias=float(error.detach().mean()),
            prediction_variance=float(prediction[q].detach().var(unbiased=False)),
            target_variance=float(target.values[q].var(unbiased=False)),valid_pixels=int(q.sum()))
        rows.append(dict(sensor=observation['sensor'],instrument=observation['instrument'],band=target.band_key,
            observation=observation['id'],loss=loss,metrics=metrics))
        if save_predictions:cases.append(dict(band=target.band_key,prediction=prediction.detach().cpu(),
            target=target.values.detach().cpu(),valid=q.detach().cpu(),output_hash=tensor_hash(prediction)))
    return rows,dict(input_hash=input_hash(task),tokens=representation.features.shape[1],
        valid_tokens=int(representation.valid.sum()),native_input_pixels=sum(b.values.numel() for b in task.visible.bands),
        visible_source_pixels=sum(int(b.keep().sum()) for b in task.visible.bands),
        target_pixels=sum(r['metrics']['valid_pixels'] for r in rows),views=1,
        token_budget_capped=representation.grids[0]['token_budget_capped'],grids=representation.grids),cases


def aggregate_metrics(rows):
    result={}
    for sensor in sorted({r['sensor'] for r in rows}):
        bands=sorted({r['band'] for r in rows if r['sensor']==sensor})
        for band in bands:
            selected=[r['metrics'] for r in rows if r['sensor']==sensor and r['band']==band]
            for key in ['nmse','rmse','mae','bias','prediction_variance','target_variance']:
                result[f'{sensor}/{band}/{key}']=sum(r[key] for r in selected)/len(selected)
    result['score']=float(balanced_loss(rows).detach())
    return result

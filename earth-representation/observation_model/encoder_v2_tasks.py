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


def training_spatial_task(image,member,update,excluded=()):
    if update<1:raise ValueError('Training mask needs a positive logical update')
    task=spatial_task(image,seed_for(member,'A_train',update),excluded=excluded)
    task.meta['sampling']=dict(role='TRAIN',logical_update=update,reproducible_after_resume=True)
    return task


def validation_spatial_task(image,member,draw=0):
    return spatial_task(image,seed_for(member,'A',draw))


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


def spectral_task(image,hidden_keys):
    hidden=set(hidden_keys);visible=[b for b in image.bands if b.band_key not in hidden]
    targets=[b for b in image.bands if b.band_key in hidden]
    if not visible or len(targets)!=len(hidden):raise ValueError('Missing spectral task member or no visible input')
    return Task('B',OpticalImage(visible,image.product_id),
        [Target(b.values,b.valid,target_from_band(b),b.band_key) for b in targets],
        dict(target_bands=sorted(hidden),target_scope='entire_native_bands',exposure={b.band_key:dict(
            target_valid_pixels=int(b.valid.sum()),visible_target_pixels=0,native_pixels=b.values.numel(),gsd_m=b.support) for b in targets}))


def held_out_combinations(image):
    names={b.band_key for b in image.bands}
    candidates=[('SWIR_pair',('B11','B12')),('rededge_group',('B05','B06','B07')),
        ('visible_NIR_pair',('B04','B08')),('SWIR_pair',('SR_B6','SR_B7')),
        ('visible_NIR_group',('SR_B3','SR_B4','SR_B5'))]
    return [(name,keys) for name,keys in candidates if set(keys)<=names and len(keys)<len(names)]


def training_spectral_keys(image,member,update,anchor=None):
    names=[b.band_key for b in image.bands];rng=np.random.default_rng(seed_for(member,'B_train',update))
    anchor=anchor or names[int(rng.integers(len(names)))];forbidden=[set(keys) for _,keys in held_out_combinations(image)]
    for _ in range(50):
        maximum=min(len(names)//2,3)
        n=1 if maximum<2 or rng.random()<.5 else int(rng.integers(2,maximum+1))
        others=[n for n in names if n!=anchor]
        hidden={anchor,*rng.choice(others,n-1,replace=False).tolist()}
        if not any(group<=hidden for group in forbidden):return sorted(hidden)
    return [anchor]


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


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


def sample_target_indices(target,member,kind,limit,logical_update):
    indices=target.valid.flatten().nonzero().flatten()
    if len(indices)>limit:
        generator=torch.Generator(device=target.values.device).manual_seed(seed_for(member,kind+':pixels:'+target.band_key,logical_update))
        indices=indices[torch.randperm(len(indices),device=indices.device,generator=generator)[:limit]]
    return indices


def run_task(model,task,observation,statistics,save_predictions=False,target_sample_pixels=None,logical_update=0):
    representation=model.encode(task.visible);rows=[];cases=[]
    for target in task.targets:
        pixel_indices=None
        if target_sample_pixels is not None:
            pixel_indices=sample_target_indices(target,observation['id'],task.kind,target_sample_pixels,logical_update)
        prediction,coverage=model.decoder(representation,target.grid,pixel_indices=pixel_indices)
        truth=target.values if pixel_indices is None else target.values.flatten()[pixel_indices]
        declared=target.valid if pixel_indices is None else torch.ones_like(coverage)
        q=declared&coverage
        # Missing prediction support cannot quietly reduce the declared task.
        if not torch.equal(q,declared):raise ValueError('Declared real targets are not all predicted')
        if not q.any():raise ValueError('No valid target')
        stat=statistics[observation['instrument']+':'+target.band_key]
        error=prediction[q].float()-truth[q].float();loss=(error/stat['std']).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reflectance objective')
        metrics=dict(nmse=float(loss.detach()),rmse=float(error.detach().square().mean().sqrt()),
            mae=float(error.detach().abs().mean()),bias=float(error.detach().mean()),
            prediction_variance=float(prediction[q].detach().var(unbiased=False)),
            target_variance=float(truth[q].var(unbiased=False)),valid_pixels=int(q.sum()),declared_full_valid_pixels=int(target.valid.sum()))
        rows.append(dict(sensor=observation['sensor'],instrument=observation['instrument'],band=target.band_key,
            observation=observation['id'],loss=loss,metrics=metrics))
        if save_predictions:cases.append(dict(band=target.band_key,prediction=prediction.detach().cpu(),
            target=truth.detach().cpu(),valid=q.detach().cpu(),pixel_indices=None if pixel_indices is None else pixel_indices.cpu(),output_hash=tensor_hash(prediction)))
    banks=representation.optional_local_features
    local_vectors=sum(len(b.features) for b in banks[0]) if banks is not None else 0
    local_elements=sum(b.features.numel() for b in banks[0]) if banks is not None else 0
    return rows,dict(input_hash=input_hash(task),tokens=representation.features.shape[1],local_feature_vectors=local_vectors,
        encoder_feature_elements=representation.features.numel()+local_elements,
        full_valid_target_pixels=sum(r['metrics']['declared_full_valid_pixels'] for r in rows),target_estimator='uniform-valid-pixel mean' if target_sample_pixels else 'full valid target mean',
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

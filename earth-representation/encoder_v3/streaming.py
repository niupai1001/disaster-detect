"""Exact full-target losses with bounded Decoder graphs and one Encoder backward."""
from dataclasses import dataclass, replace
from contextlib import nullcontext
import math
import hashlib
import json
import torch
from .layout import native_layout
from .spatial_index import SpatialIndex


@dataclass
class Geometry:
    cells: list
    centers: torch.Tensor
    bounds: torch.Tensor
    valid: torch.Tensor
    crs: str
    spatial_index: object | None = None


@dataclass
class BandPlan:
    target: object
    indices: torch.Tensor
    supported_count: int


@dataclass
class TaskPlan:
    geometry: Geometry
    bands: list
    patch_m: float
    origin_xy: tuple
    kind: str
    observation_id: str
    signature: str


def task_signature(task):
    digest=hashlib.sha256(json.dumps(dict(kind=task.kind,rectangle=task.rectangle,id=task.metadata['id'],source=task.metadata.get('frozen_source_ids')),sort_keys=True).encode())
    for band,mask in [(b,None) for b in task.visible]+[(t.band,t.mask) for t in task.targets]:
        digest.update(json.dumps([band.record_id,band.crs,band.affine,list(band.reflectance.shape)]).encode())
        for tensor in (band.valid_mask,band.visible,mask,band.wavelengths_nm,band.response):
            if tensor is None:digest.update(b'none')
            else:digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def geometry_for_task(task, *, patch_m, origin_xy):
    """CPU geometry only. Never read input or target reflectance values."""
    if not task.visible or len({b.crs for b in task.visible})!=1: raise ValueError('Visible bands with common CRS required')
    cells=[];centers=[];bounds=[]
    for band in task.visible:
        band.validate();layout=native_layout(band,patch_m,origin_xy)
        if layout is None:continue
        cells.extend(torch.from_numpy(layout['represented']).split(layout['represented_lengths']))
        centers.append(torch.from_numpy(layout['centers']));bounds.append(torch.from_numpy(layout['bounds']))
    if not centers: return None
    return Geometry(cells,torch.cat(centers),torch.cat(bounds),torch.ones(len(cells),dtype=torch.bool),task.visible[0].crs)


def prepare_task(task, *, patch_m, origin_xy, chunk_pixels, cell_indexed=False):
    if chunk_pixels<1 or not math.isfinite(patch_m) or patch_m<=0 or len(origin_xy)!=2 or not all(math.isfinite(v) for v in origin_xy): raise ValueError('Positive execution chunk/scale and finite origin required')
    geometry=geometry_for_task(task,patch_m=patch_m,origin_xy=origin_xy)
    if geometry is None:return None
    geometry.spatial_index=SpatialIndex(geometry)
    mask=None if task.rectangle is None else torch.tensor(task.rectangle,dtype=torch.float64)
    plans=[]
    for target in task.targets:
        band=target.band
        if band.crs!=geometry.crs or target.mask.shape!=band.reflectance.shape or target.mask.dtype!=torch.bool:
            raise ValueError('Target grid/validity does not match task geometry')
        indices=torch.where((target.mask & band.valid_mask).detach().cpu().flatten())[0]
        bounds=band.cells().detach().cpu();count=0
        for ids in indices.split(chunk_pixels):
            q,_,_,_=geometry.spatial_index.edges(geometry,bounds[ids],task.kind,mask,cell_indexed=cell_indexed)
            count+=len(torch.unique(q))
        plans.append(BandPlan(target,indices,count))
    return TaskPlan(geometry,plans,patch_m,tuple(origin_xy),task.kind,task.metadata['id'],task_signature(task))


def stream_task_roots(encoder, decoder, task, plan, *, train_stds, chunk_pixels, loss_weight, backward, precision='fp32', representation=None):
    """Backpropagate into Decoder per chunk, then VJP accumulated Z/SRF gradients once.

    loss_weight is the already-frozen observation/source/curriculum multiplier.
    This function never steps an optimizer. Returned losses are detached metrics.
    """
    if chunk_pixels<1 or not math.isfinite(loss_weight) or loss_weight<=0:raise ValueError('Positive explicit execution chunk and loss weight required')
    if plan is None:
        return dict(status='NO_VALID_SUPPORTED_TARGET',loss=None,bands=[]),[],[]
    if plan.kind!=task.kind or plan.observation_id!=task.metadata['id']:raise ValueError('Task identity differs from geometry plan')
    if plan.signature!=task_signature(task):raise ValueError('Task geometry/metadata changed after planning')
    if not any(b.supported_count for b in plan.bands):return dict(status='NO_VALID_SUPPORTED_TARGET',loss=None,bands=[]),[],[]
    if precision not in ('fp32','bf16'):raise ValueError('Explicit FP32/BF16 execution mode required')
    device=next(encoder.parameters()).device
    if precision=='bf16' and device.type!='cuda':raise ValueError('BF16 candidate requires CUDA')
    context=lambda: torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
    if representation is None:
        with torch.set_grad_enabled(backward),context():
            representation=encoder(task.visible,patch_m=plan.patch_m,origin_xy=plan.origin_xy)
    if len(representation.tokens)!=len(plan.geometry.valid) or representation.crs!=plan.geometry.crs:
        raise ValueError('Encoder geometry differs from supervision plan')
    torch.testing.assert_close(representation.bounds.detach().cpu(),plan.geometry.bounds,atol=1e-6,rtol=0)
    latent=representation.tokens.detach().requires_grad_(backward)
    readout=replace(representation,tokens=latent,spatial_index=None)
    active=sum(b.supported_count>0 for b in plan.bands);roots=[];gradients=[];reports=[];task_losses=[]
    mask=None if task.rectangle is None else representation.bounds.new_tensor(task.rectangle)
    for band_plan,target in zip(plan.bands,task.targets,strict=True):
        band=target.band;count=band_plan.supported_count
        if not count:
            reports.append(dict(band=band.record_id,status='NO_VALID_SUPPORTED_TARGET',count=0,unsupported=len(band_plan.indices),nmse=None));continue
        std=float(train_stds[band.record_id])
        if not math.isfinite(std) or std<=0:raise ValueError('Positive finite TRAIN-only standard deviation required')
        with torch.set_grad_enabled(backward),context():condition=encoder.srf(band.wavelengths_nm,band.response)
        condition_leaf=condition.detach().requires_grad_(backward)
        totals=torch.zeros(7,dtype=torch.float64,device=device);seen=0
        bounds=band.cells()
        for cpu_ids in band_plan.indices.split(chunk_pixels):
            ids=cpu_ids.to(device); truth=band.reflectance.flatten()[ids]
            with torch.set_grad_enabled(backward),context():
                prediction,supported=decoder(readout,bounds[ids],condition_leaf,kind=task.kind,mask_bounds=mask)
                loss_dtype=torch.float64 if prediction.dtype==torch.float64 else torch.float32
                p,y=prediction[supported].to(loss_dtype),truth[supported].to(loss_dtype)
                if not bool(torch.isfinite(p).all() & torch.isfinite(y).all()):raise FloatingPointError('Nonfinite supported target/prediction')
                if not len(p):continue
                error=p-y
                weighted=error.square().sum()*(loss_weight/(active*count*std**2))
            if backward:weighted.backward()
            seen+=len(p)
            p,y,error=p.detach().double(),y.detach().double(),error.detach().double()
            totals+=torch.stack((error.square().sum(),error.abs().sum(),error.sum(),p.sum(),p.square().sum(),y.sum(),y.square().sum()))
        if seen!=count:raise RuntimeError('Decoder support changed after geometry planning')
        sse,sae,bias,psum,p2sum,ysum,y2sum=totals.cpu().tolist()
        nmse=sse/(count*std**2);task_losses.append(nmse)
        reports.append(dict(band=band.record_id,status='OK',count=count,unsupported=len(band_plan.indices)-count,nmse=nmse,mae=sae/count,rmse=math.sqrt(sse/count),bias=bias/count,prediction_variance=max(0.,p2sum/count-(psum/count)**2),target_variance=max(0.,y2sum/count-(ysum/count)**2),target_mean=ysum/count))
        if backward:
            if condition_leaf.grad is None:raise RuntimeError('Missing target SRF gradient')
            roots.append(condition);gradients.append(condition_leaf.grad)
    if backward:
        if latent.grad is None:raise RuntimeError('Missing Z gradient')
        roots.insert(0,representation.tokens);gradients.insert(0,latent.grad)
    report=dict(status='OK',loss=sum(task_losses)/len(task_losses),weighted_loss=loss_weight*sum(task_losses)/len(task_losses),bands=reports,tokens=len(latent),all_target_pixels=True,decoder_chunk_pixels=chunk_pixels)
    return report,roots,gradients


def stream_task(encoder,decoder,task,plan,**kwargs):
    report,roots,gradients=stream_task_roots(encoder,decoder,task,plan,**kwargs)
    if roots:torch.autograd.backward(roots,gradients)
    return report

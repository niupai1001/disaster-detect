"""Equivalent target backward with one condition per channel and cached pure geometry."""
from dataclasses import replace
import math
import torch
from .data import target_chunks
from .retrieval import ViewRetrievalCache
from .memory_read import read_memory


class StopRequested(Exception):
    pass


def packed_targets(targets,size,stop_requested=None):
    parts=[];filled=0;grid=None;gsd=None
    def joined():return tuple(torch.cat([part[i] for part in parts]) for i in range(5))
    for index,(channel,mask,truth) in enumerate(targets):
        if stop_requested is not None and stop_requested():raise StopRequested()
        if parts and channel.grid!=grid:
            yield grid,gsd,joined();parts=[];filled=0
        grid,gsd=channel.grid,channel.gsd_xy;rc=mask.nonzero();count=len(rc)
        if not count:raise ValueError('Empty scoring channel')
        for points,y,linear in target_chunks(channel,mask,truth,size,with_indices=True,coordinates=rc):
            offset=0
            while offset<len(y):
                take=min(size-filled,len(y)-offset);end=offset+take
                ids=torch.full((take,),index,device=y.device,dtype=torch.long)
                weights=y.new_full((take,),1./count/len(targets))
                parts.append((points[offset:end],y[offset:end],linear[offset:end],ids,weights));filled+=take;offset=end
                if filled==size:
                    yield grid,gsd,joined();parts=[];filled=0
    if parts:yield grid,gsd,joined()


def view_loss(model,view,*,backward=False,tile_rows=128,chunk_rows=2,chunk_pixels=128,loss_weight=1.,stop_requested=None):
    if chunk_pixels<1:raise ValueError('Positive target chunk size required')
    if not math.isfinite(loss_weight) or loss_weight<0:raise ValueError('Finite nonnegative objective weight required')
    representation=model.encode(view.channels,view.geometry,tile_rows=tile_rows,chunk_rows=chunk_rows)
    roots=(representation.fine,representation.latent,representation.global_latent,*representation.memory)
    if backward:
        leaves=tuple(t.detach().requires_grad_(True) for t in roots)
        read=replace(representation,fine=leaves[0],latent=leaves[1],global_latent=leaves[2],memory=leaves[3:])
    else:read=representation
    projections=model.decoder.prepare(read)
    projected_leaves=tuple(t.detach().requires_grad_(True) for t in projections) if backward else projections
    read._decoder_projection=projected_leaves
    retrieval=ViewRetrievalCache(read,view.targets)
    read._retrieval_index=retrieval.index
    value=read.fine.new_zeros((),dtype=torch.float64)
    scans=sum(m['distance_evaluations'] for m in retrieval.metrics)
    seconds=sum(m['seconds'] for m in retrieval.metrics)
    candidates=len(read.centers);targets=0
    channels=[c for c,_,_ in view.targets]
    for channel in channels:
        if channel.channel_meta!=model.registry.meta(read.sensor_key,channel.channel_meta.channel_id):raise ValueError('Channel metadata disagrees with registry')
    queries=model.module_query.forward_many('decoder',[c.channel_meta for c in channels],[c.gsd_xy for c in channels])
    conditions=read_memory(queries,*read.memory)[1]
    z=conditions.detach().requires_grad_(True) if backward else conditions
    for grid,gsd,(points,y,linear,ids,weights) in packed_targets(view.targets,chunk_pixels,stop_requested):
        if stop_requested is not None and stop_requested():raise StopRequested()
        targets+=len(y)
        prediction,metrics=model.decoder(read,points,gsd,z[ids],chunk_pixels,retrieval.get(grid,linear))
        loss=((prediction-y).square()*weights).sum()
        if backward:(loss*loss_weight).backward()
        value=value+loss.detach().double()
        scans+=sum(m['distance_evaluations'] for m in metrics)
        seconds+=sum(m['seconds'] for m in metrics)
    if backward and z.grad is not None:conditions.backward(z.grad)
    if not bool(torch.isfinite(value)):raise ValueError('Nonfinite loss; no optimizer update permitted')
    if backward:
        projected_active=[(root,leaf.grad) for root,leaf in zip(projections,projected_leaves) if leaf.grad is not None]
        torch.autograd.backward([r for r,g in projected_active],[g for r,g in projected_active])
        active=[(r,l.grad) for r,l in zip(roots,leaves) if l.grad is not None]
        torch.autograd.backward([r for r,g in active],[g for r,g in active])
        flags=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if flags and not bool(torch.stack(flags).all()):raise ValueError('Nonfinite parameter gradient')
    return dict(loss=float(value),targets=targets,census=representation.census,distance_evaluations=scans,candidate_count=candidates,
                retrieval_seconds=seconds,retrieval_grid_caches=len(retrieval.maps),
                distance_evaluation_scope='explicit_canonical_refinement_only; library_internal_traversal_unavailable')


def hierarchical_loss(observations):
    sources={}
    for source,a,b in observations:sources.setdefault(source,[]).append(.5*a+.5*b)
    if not sources:raise ValueError('Empty objective is unavailable, not zero')
    return sum(sum(v)/len(v) for v in sources.values())/len(sources)

"""Exact streamed target backward; never performs a real optimizer step."""
from dataclasses import replace
import torch
from .data import target_chunks


def view_loss(model, view, *, backward=False, tile_rows=128, chunk_rows=2, chunk_pixels=128):
    representation=model.encode(view.channels,view.geometry,tile_rows=tile_rows,chunk_rows=chunk_rows)
    roots=(representation.fine,representation.latent,representation.global_latent,*representation.memory)
    if backward:
        leaves=tuple(t.detach().requires_grad_(True) for t in roots)
        read=replace(representation,fine=leaves[0],latent=leaves[1],global_latent=leaves[2],memory=leaves[3:])
    else:
        read=representation
    value=0.; scans=0; seconds=0.; candidates=0; targets=0
    for channel,mask,truth in view.targets:
        count=int(mask.sum()); targets+=count
        for points,y in target_chunks(channel,mask,truth,chunk_pixels):
            prediction,metrics=model.predict(read,channel.channel_meta,channel.gsd_xy,points,chunk_pixels=chunk_pixels)
            loss=(prediction-y).square().sum()/count/len(view.targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            if backward:
                loss.backward()
            value+=float(loss.detach())
            scans+=sum(m['distance_evaluations'] for m in metrics)
            seconds+=sum(m['seconds'] for m in metrics)
            candidates=max(candidates,max(m['candidates'] for m in metrics))
    if backward:
        active=[(r,l.grad) for r,l in zip(roots,leaves) if l.grad is not None]
        torch.autograd.backward([r for r,g in active],[g for r,g in active])
        if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite parameter gradient')
    return dict(loss=value,targets=targets,census=representation.census,distance_evaluations=scans,
                candidate_count=candidates,retrieval_seconds=seconds)


def hierarchical_loss(observations):
    """observations = [(source, loss_A, loss_B1)]; each task already averages channels."""
    sources={}
    for source,a,b in observations:
        sources.setdefault(source,[]).append(.5*a+.5*b)
    if not sources:
        raise ValueError('Empty objective is unavailable, not zero')
    return sum(sum(v)/len(v) for v in sources.values())/len(sources)

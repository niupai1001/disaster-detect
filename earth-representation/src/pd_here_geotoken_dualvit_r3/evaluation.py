"""Native-grid metrics and visible-input reconstruction references."""
import torch
import torch.nn.functional as F
from .data import unpatchify


def metric_records(prediction,values,qa,keep,bands,task):
    records=[];offset=0
    for j,(truth,q) in enumerate(zip(values,qa)):
        width=truth.shape[-2]*truth.shape[-1]
        pred=prediction[...,offset:offset+width].reshape_as(truth);offset+=width
        selected=q & ((~keep)[:,:,None,None] if task=='A' else (~bands[:,j])[:,None,None,None])
        for i in range(len(truth)):
            mask=selected[i]
            if not mask.any():continue
            x=pred[i][mask].double();y=truth[i][mask].double();error=x-y
            px,py,valid=unpatchify(pred[i]),unpatchify(truth[i]),unpatchify(mask)
            dx=torch.cat(((px[:,1:]-px[:,:-1])[valid[:,1:]&valid[:,:-1]],(px[1:]-px[:-1])[valid[1:]&valid[:-1]])).double()
            dy=torch.cat(((py[:,1:]-py[:,:-1])[valid[:,1:]&valid[:,:-1]],(py[1:]-py[:-1])[valid[1:]&valid[:-1]])).double()
            record=dict(parent=i,band=j,mse=error.square().mean().item(),mae=error.abs().mean().item(),
                        bias_squared=error.mean().square().item(),centered_mse=(error-error.mean()).square().mean().item())
            if dx.numel():
                record['gradient_mse'] = (dx-dy).square().mean().item()
                truth_rms = dy.square().mean().sqrt()
                record['truth_gradient_energy'] = truth_rms.square().item()
                if truth_rms >= 1e-6:
                    record['gradient_amplitude'] = (dx.square().mean().sqrt()/truth_rms).item()
                xc = dx-dx.mean()
                yc = dy-dy.mean()
                if min(xc.square().mean().sqrt(),yc.square().mean().sqrt()) >= 1e-6:
                    record['gradient_correlation'] = ((xc*yc).sum()/(xc.norm()*yc.norm())).item()
            records.append(record)
    return records


def reduce_records(records, counts=False):
    from collections import defaultdict
    sums = defaultdict(list)
    for record in records:
        for key,value in record.items():
            if key not in ('parent','band','truth_gradient_energy'):
                sums[key,record['parent']].append(value)
    parents = defaultdict(list)
    for (key,_),values in sums.items():
        parents[key].append(sum(values)/len(values))
    result = {key:sum(values)/len(values) for key,values in parents.items()}
    return (result,{key:len(values) for key,values in parents.items()}) if counts else result


def metrics(prediction,values,qa,keep,bands,task):
    return reduce_records(metric_records(prediction,values,qa,keep,bands,task))


def a_baseline(values,qa,keep,method):
    result=[];coords=torch.stack(torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij'),-1).reshape(16,2).to(keep.device)
    dist=torch.cdist(coords.float(),coords.float())
    for value,q in zip(values,qa):
        observed=q & keep[:,:,None,None]
        means=torch.where(observed,value,0.).flatten(2).sum(-1)/observed.flatten(2).sum(-1).clamp_min(1)
        available=observed.flatten(2).any(-1)
        weights=available[:,None,:].float()/dist.clamp_min(.1)[None].square()
        if method=='nearest':
            ids=dist[None].expand(len(value),-1,-1).masked_fill(~available[:,None,:],float('inf')).argmin(-1)
            estimate=means.gather(1,ids)
        elif method=='constant':
            estimate=torch.where(observed,value,0.).flatten(1).sum(-1)/observed.flatten(1).sum(-1).clamp_min(1)
            estimate=estimate[:,None].expand(-1,16)
        else:estimate=(weights*means[:,None,:]).sum(-1)/weights.sum(-1).clamp_min(1e-12)
        result.append(estimate[:,:,None].expand(-1,-1,value.shape[-2]*value.shape[-1]))
    return torch.cat(result,-1)


def coarse(values,qa):
    # Baseline only: 10m -> 20m area means; model inputs remain native.
    x=[];valid=[]
    for v,q in zip(values,qa):
        b,p,h,w=v.shape
        if h==12:
            vv=F.avg_pool2d(torch.where(q,v,0.).reshape(b*p,1,h,w),2)*4
            qq=F.avg_pool2d(q.float().reshape(b*p,1,h,w),2)*4
            v=(vv/qq.clamp_min(1)).reshape(b,p,6,6);q=(qq==4).reshape(b,p,6,6)
        x.append(v);valid.append(q)
    return torch.stack(x,-1),torch.stack(valid,-1)


def fit_ols(values,qa):
    x,valid=coarse(values,qa);x=x[valid.all(-1)].double();coeff=[]
    for j in range(x.shape[-1]):
        features=torch.cat((torch.ones_like(x[:,:1]),x[:,:j],x[:,j+1:]),-1)
        coeff.append(torch.linalg.lstsq(features.cpu(),x[:,j:j+1].cpu()).solution)
    return coeff


def b_baseline(values,qa,bands,coeff):
    x,_=coarse(values,qa);result=[]
    for j,v in enumerate(values):
        # The target band is never included in its own predictors.
        f=torch.cat((torch.ones_like(x[...,:1]),x[...,:j],x[...,j+1:]),-1)
        pred=(f.double()@coeff[j].to(f.device)).squeeze(-1).float()
        if v.shape[-1]==12:pred=pred.repeat_interleave(2,-1).repeat_interleave(2,-2)
        result.append(pred.flatten(2))
    return torch.cat(result,-1)

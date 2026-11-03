"""Same A targets on full raw originals: spatial and spectral controls."""
import torch
from .operators import resample_band


def features(items,record):
    j=record['target_index'];b=items[j];y,x=record['mask_patch_yx'];phase=b.phase_xy_m+torch.tensor([[x*60,y*60]],device=b.value.device)
    values=[];valid=[]
    for k,other in enumerate(items):
        if k==j:continue
        v,q=resample_band(other.value,other.valid,other.native_gsd_m,other.phase_xy_m,b.native_gsd_m,phase,footprint=60)
        values.append(v[0]);valid.append(q[0]>=.8)
    return torch.stack(values,-1),torch.stack(valid,-1)


def fit(observations,stats):
    from .train_full_image_a import inputs
    store={}
    for o in observations:
        r=o['record'];items,_=inputs(o,stats);b=items[r['target_index']];y,x=r['mask_patch_yx'];p=60//b.native_gsd_m
        a,v=features(items,r);q=v.all(-1)&b.valid[0,y*p:(y+1)*p,x*p:(x+1)*p];target=b.value[0,y*p:(y+1)*p,x*p:(x+1)*p]
        xs,ys=store.setdefault(b.name,([],[]));xs.append(a[q].double());ys.append(target[q].double())
    fits={}
    for name,(xs,ys) in store.items():
        a=torch.cat(xs);b=torch.cat(ys)
        if len(a)<2:raise ValueError('insufficient genuine spectral control training pixels')
        mean=a.mean(0);ym=b.mean();xc=a-mean;w=torch.linalg.solve(xc.T@xc/len(a)+.01*torch.eye(a.shape[1],device=a.device,dtype=torch.float64),xc.T@(b-ym)/len(a))
        fits[name]=dict(mean=mean,ym=ym,w=w,train_pixels=len(a),regularization=.01)
    return fits


def summarize(rows):
    bands={}
    for row in rows:bands.setdefault(row['band'],[]).append(row['mse'])
    means={b:sum(x)/len(x) for b,x in bands.items()}
    score=sum(sum(v for b,v in means.items() if b.startswith(s+':'))/sum(b.startswith(s+':') for b in means) for s in ['S2','L8'])/2
    return dict(score=score,bands=means,rows=rows)


def evaluate(observations,stats,fits):
    from .train_full_image_a import inputs
    rows={'spatial':[],'spectral':[]}
    for o in observations:
        r=o['record'];items,_=inputs(o,stats);b=items[r['target_index']];y,x=r['mask_patch_yx'];p=60//b.native_gsd_m;ys=slice(y*p,(y+1)*p);xs=slice(x*p,(x+1)*p)
        valid=b.valid[0,ys,xs];truth=b.value[0,ys,xs]
        yy,xx=torch.meshgrid(torch.arange(b.value.shape[-2],device='cuda'),torch.arange(b.value.shape[-1],device='cuda'),indexing='ij');xy=torch.stack([yy,xx],-1).float();keep=b.valid[0].clone();keep[ys,xs]=False
        distances,idx=torch.cdist(xy[ys,xs].reshape(-1,2),xy[keep]).topk(4,largest=False);weight=distances.clamp_min(1e-6).pow(-2);spatial=((b.value[0][keep][idx]*weight).sum(1)/weight.sum(1)).reshape(p,p)
        a,v=features(items,r);f=fits[b.name];a=torch.where(v,a,f['mean'].float());spectral=((a.double()-f['mean'])@f['w']+f['ym']).float()
        for method,pred in [('spatial',spatial),('spectral',spectral)]:
            e=(pred-truth)[valid];rows[method].append(dict(id=r['observation_id'],band=b.name,mse=float(e.square().mean()),rmse=float(e.square().mean().sqrt())*stats[b.name]['std'],bias=float(e.mean())*stats[b.name]['std']))
    return {k:summarize(v) for k,v in rows.items()}

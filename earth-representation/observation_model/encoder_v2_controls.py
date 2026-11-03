"""Small-data reflectance controls, fitted only on frozen TRAIN observations."""
import torch

from .encoder_v2 import axis_overlap


def area_features(image,target):
    columns=[];availability=[];keys=[]
    h,w=target.values.shape;gx,gy=target.support;_,_,left,_,_,top=target.transform
    ex=torch.arange(w+1,device=target.values.device,dtype=torch.float32)*gx
    ey=torch.arange(h+1,device=target.values.device,dtype=torch.float32)*gy
    for band in image.bands:
        if band.band_key==target.band_key:continue
        sx,sy=band.support;hh,ww=band.values.shape;_,_,cx,_,_,cy=band.transform
        px=torch.arange(ww,device=ex.device,dtype=torch.float32)*sx+(cx-left)
        py=torch.arange(hh,device=ey.device,dtype=torch.float32)*sy+(top-cy)
        wx=axis_overlap(ex,px,px+sx);wy=axis_overlap(ey,py,py+sy)
        valid=band.keep();area=(wy@valid.float())@wx.T
        values=(wy@torch.where(valid,band.values,0))@wx.T/area.clamp_min(1e-10)
        columns.append(values);availability.append(area>=.8*gx*gy);keys.append(band.band_key)
    return torch.stack(columns,-1),torch.stack(availability,-1),keys


def fit_ridge(data,regularization=1e-3):
    """Covariance ridge, following the existing control; no target-band predictor."""
    groups={};fits={}
    with torch.no_grad():
        for meta,image in data:
            if meta['role']!='TRAIN':continue
            for target in image.bands:
                x,q,keys=area_features(image,target);valid=q.all(-1)&target.valid
                x=x[valid].double();y=target.values[valid].double()
                if len(y)<16:raise ValueError('Insufficient real TRAIN pixels for ridge')
                # Deterministic sampling bounds baseline fit cost, not model inputs.
                indices=torch.linspace(0,len(y)-1,min(len(y),8192),device=y.device).long()
                groups.setdefault((meta['sensor'],target.band_key),[]).append((x[indices],y[indices],keys))
        for (sensor,band),parts in groups.items():
            keys=parts[0][2];assert all(p[2]==keys for p in parts)
            x=torch.cat([p[0] for p in parts]);y=torch.cat([p[1] for p in parts])
            mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-5);z=(x-mean)/scale;ym=y.mean()
            weight=torch.linalg.solve(z.T@z/len(z)+regularization*torch.eye(z.shape[1],device=z.device,dtype=z.dtype),z.T@(y-ym)/len(z))
            fits[sensor+':'+band]=dict(mean=mean,scale=scale,weight=weight,ym=ym,keys=keys,n=len(y))
    return fits


def ridge_prediction(image,target,fit):
    x,valid,keys=area_features(image,target)
    if keys!=fit['keys']:raise ValueError('Ridge predictor band mismatch')
    x=torch.where(valid,x.double(),fit['mean'])
    return (((x-fit['mean'])/fit['scale'])@fit['weight']+fit['ym']).float()


def spatial_prediction(band,mean):
    """Four-neighbor inverse-distance interpolation, as in the prior A control."""
    valid=band.keep();result=torch.full_like(band.values,float(mean))
    source=torch.nonzero(valid).float();targets=torch.nonzero(~valid).float()
    if len(source)==0:return result
    values=band.values[valid]
    for begin in range(0,len(targets),256):
        coords=targets[begin:begin+256]
        distance,indices=torch.cdist(coords,source).topk(min(4,len(source)),largest=False)
        weight=distance.clamp_min(1e-6).pow(-2)
        prediction=(values[indices]*weight).sum(1)/weight.sum(1)
        at=coords.long();result[at[:,0],at[:,1]]=prediction
    result[valid]=values
    return result

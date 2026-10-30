"""Identical physical low-frequency reconstruction target for both P2 encoders."""
import copy

import numpy as np
import torch
from torch import nn

from .encoder import ScaleUSE,geometry,grid_for,mask_asset


def masks(asset,seed):
    grid=grid_for(asset);h,w=grid['shape'];rng=np.random.default_rng(seed)
    n=(h//4)*(w//4);hidden=np.zeros(n,dtype=bool)
    hidden[rng.choice(n,size=max(1,min(n-1,round(.4*n))),replace=False)]=True
    return torch.from_numpy(hidden.reshape(h//4,w//4)).repeat_interleave(4,0).repeat_interleave(4,1),grid


def band_view(asset,seed):
    if asset.modality not in ('s2','landsat') or len(asset.image)<3:return asset
    rng=np.random.default_rng(seed)
    if rng.random()>=.5:return asset
    drop=max(1,int(round(len(asset.image)*rng.uniform(.2,.4))))
    ids=sorted(rng.choice(len(asset.image),size=max(1,len(asset.image)-drop),replace=False).tolist())
    result=copy.copy(asset)
    for key in ('image','pixel_valid','channel_metadata','metadata_known'):setattr(result,key,getattr(asset,key)[ids])
    result.channel_names=tuple(asset.channel_names[i] for i in ids)
    for key in ('native_resolution','effective_resolution'):
        if getattr(asset,key) is not None:setattr(result,key,tuple(getattr(asset,key)[i] for i in ids))
    return result


def patch_targets(asset,grid,calibration):
    if asset.modality=='s1':return None,'NOT_AVAILABLE_SAR_NO_VERIFIED_NOISE_CALIBRATION'
    matrix,(h,w),_,_=geometry(asset,grid);matrix=matrix.to(asset.image.device)
    valid=asset.pixel_valid & asset.metadata_known[:,2,None,None] & (asset.channel_metadata[:,2,None,None]<=120)
    x=torch.where(valid,asset.image,0)
    sums=torch.sparse.mm(matrix,x.flatten(1).T).T.reshape(len(x),h//4,4,w//4,4).sum((2,4))
    support=torch.sparse.mm(matrix,valid.flatten(1).float().T).T.reshape(len(x),h//4,4,w//4,4).sum((2,4))
    target=sums/support.clamp_min(1e-8)
    key=asset.modality+'/'+','.join(asset.provenance['band_ids'])
    stats=calibration[key]
    mean=target.new_tensor(stats['mean'])[:,None,None];std=target.new_tensor(stats['std'])[:,None,None]
    return ((target-mean)/std,support),None


class ScaleTask(nn.Module):
    def __init__(self,root,mode):
        super().__init__();self.use=ScaleUSE(root,mode)
        torch.manual_seed(1235)
        self.predictor=nn.Sequential(nn.Linear(200,192),nn.GELU(),nn.Linear(192,1))

    def loss(self,asset,spatial_seed,band_seed,calibration):
        hidden,grid=masks(asset,spatial_seed)
        target,reason=patch_targets(asset,grid,calibration)
        if target is None:return None,dict(reason=reason,modality=asset.modality)
        view=band_view(asset,band_seed)
        output=self.use(view,grid,hidden)
        if output.global_embedding is None:return None,dict(reason='NO_VISIBLE_CONTEXT',modality=asset.modality)
        values,support=target;c,ph,pw=values.shape
        yy,xx=torch.meshgrid(torch.arange(ph,device=values.device),torch.arange(pw,device=values.device),indexing='ij')
        # Both groups predict the same physical 120m targets using the same query coordinates.
        query=torch.stack((xx+.5,yy+.5),-1).float()
        metadata=torch.cat((torch.where(asset.metadata_known,asset.channel_metadata,0),asset.metadata_known.float()),-1)
        latent=output.global_embedding[None,None,None].expand(c,ph,pw,-1)
        decoder=torch.cat((latent,query[None].expand(c,-1,-1,-1),metadata[:,None,None].expand(-1,ph,pw,-1)),-1)
        prediction=self.predictor(decoder).squeeze(-1)
        weights=support*hidden[::4,::4].to(support.device)[None]
        if weights.sum()==0:return None,dict(reason='NO_SUPPORTED_HIDDEN_TARGET',modality=asset.modality)
        loss=((prediction-values.detach()).square()*weights).sum()/weights.sum()
        return loss,dict(modality=asset.modality,low_frequency_support=float(weights.sum()),high_frequency='NOT_AVAILABLE_EFFECTIVE_SUPPORT_UNKNOWN',
                         hidden_cells=int(hidden[::4,::4].sum()),target_variance=float(values.var()),loss=float(loss.detach()))

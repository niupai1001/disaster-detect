import copy
import math
from collections import OrderedDict

import numpy as np
from pyproj import Transformer
import torch
from torch.nn import functional as F

from landslide_seg.models.use_next_v2 import FeatureLevel,USEOutput
from landslide_seg.models.use_next_v2.physical import PhysicalLocal
from landslide_seg.models.use_next_v2.pipeline import USEModel
from .geometry import area_grid

_geometry=OrderedDict()


def geometry(asset,grid=None):
    key=(asset.sample_grid['crs'],tuple(asset.sample_grid['transform']),tuple(asset.image.shape[-2:]),str(grid))
    if key not in _geometry:
        _geometry[key]=area_grid(asset,grid)
        while len(_geometry)>16:_geometry.popitem(last=False)
    else:_geometry.move_to_end(key)
    return _geometry[key]


def grid_for(asset):
    value=geometry(asset);_,shape,t,crs=value
    grid=dict(crs=crs,shape=shape,origin=(t[2],t[5]))
    key=(asset.sample_grid['crs'],tuple(asset.sample_grid['transform']),tuple(asset.image.shape[-2:]),str(grid))
    _geometry[key]=value
    return grid


def mask_asset(asset,hidden,grid):
    matrix,shape,_,_=geometry(asset,grid)
    if tuple(hidden.shape)!=shape:raise ValueError('Physical mask mismatch')
    pixels=torch.sparse.mm(matrix.T,hidden.float().cpu().flatten()[:,None]).squeeze(1)>0
    result=copy.copy(asset)
    result.pixel_valid=asset.pixel_valid & ~pixels.reshape(asset.image.shape[-2:]).to(asset.image.device)[None]
    result.image=torch.where(result.pixel_valid,asset.image,0)
    return result


def block_four(asset):
    result=copy.copy(asset);_,h,w=asset.image.shape
    valid=F.pad(asset.pixel_valid.float(),(0,(-w)%4,0,(-h)%4))
    values=F.pad(torch.where(asset.pixel_valid,asset.image,0),(0,(-w)%4,0,(-h)%4))
    fraction=F.avg_pool2d(valid,4,4)
    result.image=F.avg_pool2d(values,4,4)/fraction.clamp_min(1e-8)
    result.pixel_valid=fraction>0
    a,b,c,d,e,f=asset.sample_grid['transform']
    result.sample_grid=dict(asset.sample_grid,transform=(a*4,b*4,c,d*4,e*4,f))
    return result,fraction


def scalar_spacing(transform):return math.hypot(transform[0],transform[3])


class ScaleEncoder(PhysicalLocal):
    def __init__(self,mode):
        super().__init__()
        if mode not in ('pixel','physical'):raise ValueError('Unknown scale mode')
        self.mode=mode

    def forward(self,asset,hidden_cells=None,grid=None):
        asset.validate()
        if asset.modality=='hr':raise ValueError('HR read only')
        grid=grid or grid_for(asset)
        if hidden_cells is not None:asset=mask_asset(asset,hidden_cells,grid)
        matrix,(h,w),t,crs=geometry(asset,grid);matrix=matrix.to(asset.image.device)
        valid=asset.pixel_valid.flatten(1).float()
        support=torch.sparse.mm(matrix,valid.T).T
        source,fraction=block_four(asset) if self.mode=='pixel' else (asset,asset.pixel_valid.float())
        coarse,_,_,_=geometry(source,grid);coarse=coarse.to(asset.image.device)
        denom=torch.sparse.mm(coarse,fraction.flatten(1).T).T
        means=torch.sparse.mm(coarse,(torch.where(fraction>0,source.image,0)*fraction).flatten(1).T).T/denom.clamp_min(1e-8)
        visible=(support>0)&(denom>0)
        if hidden_cells is not None:visible &= ~hidden_cells.to(visible.device).flatten()[None]
        proj=self.sar if asset.modality=='s1' else self.optical
        features=proj(means[:,:,None])
        if asset.modality=='s1':
            ids=torch.tensor([('vv','vh').index(n.lower()) for n in asset.channel_names],device=features.device)
            channel=self.polarization(ids)
        else:
            metadata=torch.cat((torch.where(asset.metadata_known,asset.channel_metadata,0),asset.metadata_known.float()),-1)
            channel=self.channel(metadata)
        features=features+channel[:,None]
        features=(features*visible[:,:,None]).sum(0)/visible.sum(0).clamp_min(1)[:,None]
        yy,xx=np.mgrid[:h,:w];X=xx*30+t[2]+15;Y=yy*30+t[5]+15
        a,b,c,d,e,f=asset.sample_grid['transform']
        if self.mode=='physical':
            positions=np.stack((X,Y,np.full_like(X,math.sqrt(abs(a*e-b*d))),np.full_like(X,30),np.full_like(X,120)),-1)/120
        else:
            px,py=Transformer.from_crs(crs,asset.sample_grid['crs'],always_xy=True).transform(X,Y)
            uv=np.linalg.inv(np.array([[a,b],[d,e]]))@np.stack((px-c,py-f)).reshape(2,-1)
            positions=np.stack((uv[0].reshape(h,w),uv[1].reshape(h,w),np.ones_like(X),np.full_like(X,4),np.full_like(X,16)),-1)/16
        pos=torch.as_tensor(positions,dtype=features.dtype,device=features.device).reshape(-1,5)
        features=features+self.position(pos)
        mask=visible.any(0).reshape(h,w)
        patches=features.reshape(h//4,4,w//4,4,192).permute(0,2,1,3,4).reshape(-1,16,192)
        pmask=mask.reshape(h//4,4,w//4,4).permute(0,2,1,3).reshape(-1,16)
        nonempty=pmask.any(1);encoded=torch.zeros_like(patches)
        if nonempty.any():encoded[nonempty]=self.aggregate(patches[nonempty],src_key_padding_mask=~pmask[nonempty])
        weights=(support.max(0).values.reshape(h,w)*mask).reshape(h//4,4,w//4,4).permute(0,2,1,3).reshape(-1,16)
        pooled=(encoded*weights[:,:,None]).sum(1)/weights.sum(1).clamp_min(1e-8)[:,None]
        local=encoded.reshape(h//4,w//4,4,4,192).permute(0,2,1,3,4).reshape(h,w,192).permute(2,0,1)
        local_level=FeatureLevel(local,mask,t,30.,120.,asset.modality,asset.native_resolution,True,crs)
        patch=FeatureLevel(pooled.reshape(h//4,w//4,192).permute(2,0,1),nonempty.reshape(h//4,w//4),
                           (120.,0.,t[2],0.,120.,t[5]),120.,120.,asset.modality,asset.native_resolution,True,crs)
        count=asset.pixel_valid.sum(0).clamp_min(1)
        detail=proj((torch.where(asset.pixel_valid,asset.image,0).sum(0)/count)[:,:,None])
        detail=detail+(channel[:,:,None,None]*asset.pixel_valid[:,None]).sum(0).permute(1,2,0)/count[:,:,None]
        detail=FeatureLevel(detail.permute(2,0,1)*asset.pixel_valid.any(0),asset.pixel_valid.any(0),tuple(asset.sample_grid['transform']),
                            scalar_spacing(asset.sample_grid['transform']),None,asset.modality,asset.native_resolution,True,asset.sample_grid['crs'])
        return USEOutput(native_detail=detail,metric_local=local_level,metric_patch=patch,shared_metric_tokens=patch,
                         global_embedding=pooled[nonempty].mean(0) if nonempty.any() else None)


class ScaleUSE(USEModel):
    def __init__(self,root,mode):
        super().__init__(root)
        torch.manual_seed(1234);self.local=ScaleEncoder(mode)
        self.teacher_local=copy.deepcopy(self.local).requires_grad_(False).eval()

    def forward(self,asset,grid,hidden=None,teacher=False):
        if teacher and hidden is not None:raise ValueError('Teacher must use full observation')
        if hidden is not None:asset=mask_asset(asset,hidden,grid)
        result=super().forward(asset,grid,teacher=teacher)
        for level in (result.semantic_levels or {}).values():level.grid_spacing_m=scalar_spacing(level.transform)
        return result

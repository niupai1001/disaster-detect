"""Parallel frozen native semantics and trainable metric path with one physical mask."""
import copy

import torch
from torch import nn

from . import FeatureLevel
from .anchors import NativeAnySat,NativeDOFA
from .physical import PhysicalLocal
from .masking import mask_asset,update_ema


class USEModel(nn.Module):
    def __init__(self,root):
        super().__init__()
        self.local=PhysicalLocal()
        self.dofa=NativeDOFA(root)
        self.anysat=NativeAnySat(root)
        self.teacher_local=copy.deepcopy(self.local).requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(mode); self.teacher_local.eval()
        return self

    def forward(self,asset,grid,hidden=None,teacher=False):
        if hidden is not None and teacher:raise ValueError('Teacher target must use full valid observation')
        pixels=None
        if hidden is not None:asset,pixels=mask_asset(asset,hidden,grid)
        encoder=self.teacher_local if teacher else self.local
        with torch.set_grad_enabled(not teacher and torch.is_grad_enabled()):
            result=encoder(asset,grid=grid)
        result.semantic_levels={}
        if asset.modality in ('s2','landsat') and asset.metadata_known[:,0].all():
            features=self.dofa(asset)[-1][0]
            a,b,c,d,e,f=asset.sample_grid['transform']; _,h,w=asset.image.shape
            result.optical_semantic=FeatureLevel(features,torch.ones(features.shape[-2:],dtype=torch.bool,device=features.device),
                (a*w/14,b*h/14,c,d*w/14,e*h/14,f),abs(a*w/14),None,asset.modality,
                asset.native_resolution,asset.image.shape[-2:]!=(224,224),asset.sample_grid['crs'])
            result.semantic_levels['dofa']=result.optical_semantic
        # Illegal AnySat inputs retain independent metric/DOFA outputs; no synthetic channels.
        if asset.modality=='s2':
            try:self.anysat.prepare(asset)
            except ValueError:pass
            else:
                result.semantic_levels['anysat']=FeatureLevel(self.anysat(asset)[0].permute(2,0,1),asset.pixel_valid.all(0),
                    tuple(asset.sample_grid['transform']),10.,None,'s2',asset.native_resolution,False,asset.sample_grid['crs'])
        return result

    @torch.no_grad()
    def ema(self,momentum):update_ema(self.teacher_local,self.local,momentum)

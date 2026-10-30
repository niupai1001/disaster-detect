"""Fixed GroupNorm consumer of official frozen native levels."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from . import EOAsset
from .anchors import NativeDOFA,NativeAnySat,S2_BANDS


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class NativeReadout(nn.Module):
    @property
    def use(self):return self

    def __init__(self,root,kind,stats):
        super().__init__()
        self.kind=kind;self.stats=stats
        self.anchor=NativeDOFA(root) if kind=='dofa' else NativeAnySat(root)
        self.anchor_digest=state_digest(self.anchor)
        torch.manual_seed(1235)
        self.projections=nn.ModuleList([nn.Conv2d(768,32,1) for _ in range(4 if kind=='dofa' else 2)])
        torch.manual_seed(1235)
        self.head=nn.Sequential(nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,1,1))

    def verify(self,optimizer=None):
        if optimizer:
            ids={id(p) for g in optimizer.param_groups for p in g['params']}
            if any(id(p) in ids for p in self.anchor.parameters()):raise ValueError('Frozen anchor in optimizer')
        value=state_digest(self.anchor)
        if value!=self.anchor_digest:raise ValueError('Frozen anchor changed')
        return value

    def forward(self,batch):
        outputs=[]
        for i,x in enumerate(batch['x']):
            spatial=batch['spatial_metadata'][i]
            mean=x.new_tensor(self.stats['mean'])[:,None,None]
            std=x.new_tensor(self.stats['std'])[:,None,None]
            physical=(x*std+mean)*.0001
            valid=batch['pixel_valid'][i]
            asset=EOAsset('s2',S2_BANDS,torch.where(valid,physical,0),batch['band_metadata'][i],batch['known_mask'][i],
                {k:spatial[k] for k in ('crs','transform')},None,None,valid,spatial['date'],'surface_reflectance',spatial)
            if self.kind=='dofa':levels=self.anchor(asset)
            else:
                dense=self.anchor(asset).permute(0,3,1,2)
                levels=dense.split(768,dim=1)
            projected=[F.interpolate(proj(level),size=x.shape[-2:],mode='bilinear',align_corners=False)
                       for proj,level in zip(self.projections,levels)]
            outputs.append(self.head(torch.stack(projected).mean(0)))
        return torch.cat(outputs)

"""Area-weighted fusion of the same physical slot across visible bands."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from encoder_v3.shared_mae.geometry import AreaCache
from .frontend import Frontend, pack


@dataclass(frozen=True)
class SpatialSlots:
    """[H*W,4,8] in parent row/column then slot row/column order (south to north)."""
    values: Tensor
    observed: Tensor
    structural: Tensor
    shape: tuple

    def backbone_input(self):
        return self.values.reshape(*self.shape,32).permute(2,0,1).unsqueeze(0)


def relative_centres(grid, geometry, mapping, edge_slice, dtype):
    source = mapping.source[edge_slice]
    destination = mapping.destination[edge_slice]
    ax,_,x0,_,ey,y0 = grid.affine
    h,w = grid.shape
    s = geometry.patch_size/2
    px = (source % w + .5).double()*ax + (x0-geometry.origin[0])
    py = (source // w + .5).double()*ey + (y0-geometry.origin[1])
    parent, slot = destination//4,destination%4
    sx = (2*(parent%geometry.shape[1]) + slot%2 + .5)*s
    sy = (2*(parent//geometry.shape[1]) + slot//2 + .5)*s
    return ((px-sx)/s).to(dtype), ((py-sy)/s).to(dtype)


def add_attention_state(state, scores, values, valid):
    # Use a finite placeholder for empty slots and never exponentiate an invalid logit.
    block_valid = valid.any(0)
    top = torch.where(block_valid,scores.masked_fill(~valid,-torch.inf).amax(0),0.)
    if state is None:
        factors = torch.where(valid, scores-top, 0.).exp()*valid
        return top,factors.sum(0),(values*factors[...,None]).sum(0),block_valid
    old_top, old_mass, old_total, old_valid = state
    top = torch.where(old_valid & block_valid,torch.maximum(top,old_top),
                      torch.where(block_valid,top,old_top))
    factors = torch.where(valid, scores-top, 0.).exp()*valid
    previous = torch.where(old_valid,old_top-top,0.).exp()
    return top,old_mass*previous+factors.sum(0),old_total*previous[...,None]+(
        values*factors[...,None]).sum(0),old_valid | block_valid


class PhaseAwareFusion(nn.Module):
    def __init__(self, blocks=2, film=True):
        super().__init__()
        self.frontend = Frontend(blocks,film)
        self.phase = nn.ModuleDict({str(r):nn.Conv2d(32,r*r*8,1) for r in (1,2,4)})
        with torch.no_grad():
            for r in (1,2,4):
                layer = self.phase[str(r)]
                for a in range(r*r):
                    layer.weight[a*8].zero_()
                    layer.weight[a*8,a,0,0] = 1
                    layer.bias[a*8] = 0
        self.moments = nn.Linear(32,8)
        self.query = nn.Linear(34,8)
        self.key = nn.Linear(41,8)
        self.value = nn.Linear(40,8)
        self.area = AreaCache(limit_bytes=256*1024**2,construction_device="auto")

    def phase_features(self, packed, feature):
        r = packed.factor
        return self.phase[str(r)](feature).reshape(len(feature),r*r,8,*feature.shape[-2:])

    def aggregate(self, grid, geometry, mapping, clean, keeps, descriptor, edge_chunk):
        packed = [pack(b) for b in clean]
        features = self.frontend(packed,descriptor)
        phase = self.phase_features(packed[0],features).permute(0,1,3,4,2).flatten(1,3)
        r = packed[0].factor
        hp, wp = features.shape[-2:]
        w = grid.shape[1]
        c = len(clean)
        visible = keeps.flatten(1)
        # Accumulate 8-dimensional positioned content per band and slot, not an image skip.
        total = features.new_zeros(c,mapping.slots,8)
        support = features.new_zeros(c,mapping.slots)
        for start in range(0,len(mapping.source),edge_chunk):
            edges = slice(start,start+edge_chunk)
            src,dst = mapping.source[edges],mapping.destination[edges]
            row,col = src//w,src%w
            phase_index = ((row%r)*r+col%r)*(hp*wp)+(row//r)*wp+col//r
            z = phase[:,phase_index]
            dx,dy = (d[None,:,None] for d in relative_centres(grid,geometry,mapping,edges,z.dtype))
            moment = self.moments(torch.cat((z,dx*z,dy*z,dx*dy*z),dim=-1))
            weight = mapping.area[edges].to(z.dtype)[None]*visible[:,src]
            total = total.index_add(1,dst,moment*weight[...,None])
            support = support.index_add(1,dst,weight)
        value = total / torch.where(support>0,support,1.)[...,None]
        return value,support/mapping.slot_area

    def forward(self, bands, descriptor, geometry, *, channel_chunk=8, edge_chunk=262144, metrics=None):
        if channel_chunk<1 or edge_chunk<1 or descriptor.shape!=(len(bands),32):
            raise ValueError("Invalid fusion chunk or descriptor shape")
        geometry.check_channels(bands)
        groups = {}
        for i,band in enumerate(bands):
            groups.setdefault(band.grid,[]).append(i)
        h,w = geometry.shape
        device = descriptor.device
        structural = torch.zeros(h*w*4,dtype=torch.bool,device=device)
        present = torch.stack([b.visible.any() for b in bands]).tolist()
        active = [i for i,flag in enumerate(present) if flag]
        summary = descriptor[active].sum(0)/max(1,len(active))
        offsets = descriptor.new_tensor([[-.25,-.25],[.25,-.25],[-.25,.25],[.25,.25]])
        query = self.query(torch.cat((summary.expand(4,-1),offsets),dim=1)).repeat(h*w,1)
        state = None
        total_edges = 0
        for grid,indices in groups.items():
            mapping = self.area.get(grid,geometry,2,device)
            structural[mapping.destination] = True
            total_edges += len(mapping.source)
            indices = [i for i in indices if present[i]]
            chunk = min(channel_chunk,max(1,262144//mapping.slots),max(1,262144//mapping.pixels))
            for start in range(0,len(indices),chunk):
                ids = indices[start:start+chunk]
                values = torch.stack([bands[i].clean for i in ids])
                keeps = torch.stack([bands[i].visible for i in ids])
                def terms(values,keeps,desc,query,grid=grid,mapping=mapping):
                    local = [type(bands[0])(v,m,grid) for v,m in zip(values,keeps,strict=True)]
                    u,rho = self.aggregate(grid,geometry,mapping,local,keeps,desc,edge_chunk)
                    conditions = torch.cat((u,desc[:,None,:].expand(-1,mapping.slots,-1)),dim=-1)
                    valid = rho>0
                    keys = self.key(torch.cat((conditions,rho[...,None]),dim=-1))
                    scores = (query*keys).sum(-1)/math.sqrt(8)+torch.where(valid,rho,1.).log()
                    return scores,self.value(conditions),valid
                args = values,keeps,descriptor[ids],query
                scores,value,valid = (checkpoint(terms,*args,use_reentrant=False)
                                      if self.training and torch.is_grad_enabled() and len(active)>chunk
                                      else terms(*args))
                state = add_attention_state(state,scores,value,valid)
        if state is None:
            slots,observed = descriptor.new_zeros(h*w*4,8),torch.zeros_like(structural)
        else:
            _,mass,total,observed = state
            slots = total/torch.where(observed,mass,1.)[...,None]
        if metrics is not None:
            metrics.update(slot_overlap_edges=total_edges,active_bands=len(active),parent_slots=h*w*4)
        return SpatialSlots(slots.reshape(h*w,4,8),observed.reshape(h*w,4),structural.reshape(h*w,4),(h,w))

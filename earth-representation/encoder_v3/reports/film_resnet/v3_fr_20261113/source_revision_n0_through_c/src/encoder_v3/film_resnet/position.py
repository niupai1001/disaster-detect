"""Continuous complete-scene position features, for queries and keys only."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PhysicalPosition(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.project = nn.Linear(17,width,bias=False)
        nn.init.normal_(self.project.weight,std=.01)

    @staticmethod
    def coordinates(geometry, support, level, reduction):
        h,w = support.shape[-2:]
        side = geometry.patch_size
        y,x = torch.meshgrid(torch.arange(h,device=support.device),torch.arange(w,device=support.device),
                             indexing="ij")
        points = (torch.stack((x,y),dim=0).float()*(2**level)+.5)*side
        mask = support.float()
        pad = (0,(-w)%reduction,0,(-h)%reduction)
        mass = F.avg_pool2d(F.pad(mask,pad),reduction)
        pooled = F.avg_pool2d(F.pad(points[None]*mask,pad),reduction)/torch.where(mass>0,mass,1.)
        return points.permute(1,2,0),pooled[0].permute(1,2,0),mass

    def forward(self, geometry, support, level, reduction):
        fine,key,mass = self.coordinates(geometry,support,level,reduction)
        side = geometry.patch_size
        def features(points,step):
            extent = points.new_tensor(geometry.extent)
            base = points/extent.clamp_min(side)
            scales = torch.cat((extent/side,points.new_tensor([step/side]))).log().expand(*points.shape[:-1],3)
            periods = points.new_tensor([4.,16.,64.])*side
            phase = 2*math.pi*points[...,None]/periods
            return self.project(torch.cat((base,scales,phase.sin().flatten(-2),phase.cos().flatten(-2)),dim=-1))
        return features(fine,side*2**level),features(key,side*2**level*reduction),mass


def positioned_block(block, x, footprint, query_position, key_position, coverage):
    """The frozen reduced attention computation with offsets on Q and K inputs."""
    _,c,h,w = x.shape
    r = block.reduction
    z = block.norm(x.permute(0,2,3,1))
    clean = torch.where(footprint,z.permute(0,3,1,2),0.)
    reduced = block.reduce(F.pad(clean,(0,(-w)%r,0,(-h)%r)))
    reduced = reduced/torch.where(coverage>0,coverage,1.)
    reduced = block.reduce_norm(reduced.flatten(2).transpose(1,2))
    q = block.query(z+query_position)
    kw,vw = block.kv.weight.chunk(2,dim=0)
    kb,vb = block.kv.bias.chunk(2,dim=0)
    k = F.linear(reduced+key_position.reshape(1,-1,c),kw,kb)
    v = F.linear(reduced,vw,vb)
    q,k,v = [a.reshape(1,-1,block.heads,c//block.heads).transpose(1,2) for a in (q,k,v)]
    mask = coverage.flatten(2)[:,:,None]>0
    attended = torch.cat([F.scaled_dot_product_attention(part,k,v,attn_mask=mask)
                          for part in q.split(block.query_chunk,dim=2)],dim=2)
    out = block.output(attended.transpose(1,2).reshape(1,h,w,c)).permute(0,3,1,2)
    x = torch.where(footprint,x+out,0.)
    z = block.ff_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
    z = torch.where(footprint,block.expand(z),0.)
    return torch.where(footprint,x+block.contract(F.gelu(block.mix(z))),0.)

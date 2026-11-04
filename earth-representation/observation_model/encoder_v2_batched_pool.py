"""Reuse fixed integration geometry within a microbatch, never observed values."""
import math
import torch

from . import encoder_v2_microbatch as batched
from .encoder_v2 import axis_overlap


class CachedPoolObservationEncoderV2(batched.BatchedObservationEncoderV2):
    def encode_views(self,images):
        cache={};original=batched.pool_native
        def pooled(band,grid,config):
            with torch.autocast(band.values.device.type,enabled=False):
                x=band.values;keep=band.keep();h,w=x.shape;ny,nx=grid['shape'];left,bottom,right,top=grid['bounds'];gx,gy=band.support
                key=(tuple(x.shape),tuple(band.transform),tuple(grid['bounds']),tuple(grid['shape']),x.dtype)
                if key not in cache:
                    _,_,c,_,_,f=band.transform
                    px=(torch.arange(w,device=x.device,dtype=torch.float64)*gx+(c-left)).to(x.dtype);py=(torch.arange(h,device=x.device,dtype=torch.float64)*gy+(top-f)).to(x.dtype)
                    ex=torch.linspace(0,right-left,nx+1,device=x.device,dtype=x.dtype);ey=torch.linspace(0,top-bottom,ny+1,device=x.device,dtype=x.dtype)
                    wx=axis_overlap(ex,px,px+gx);wy=axis_overlap(ey,py,py+gy);u=(px+gx/2)/(right-left);v=(py+gy/2)/(top-bottom);bases=[]
                    for begin in range(0,h,config.pool_rows):
                        end=min(h,begin+config.pool_rows);xx=u[None].expand(end-begin,w);yy=v[begin:end,None].expand(end-begin,w)
                        bases.append(torch.stack([torch.ones_like(xx),xx,yy,xx*xx,yy*yy,xx*yy,torch.sin(2*math.pi*xx),torch.cos(2*math.pi*xx),torch.sin(2*math.pi*yy),torch.cos(2*math.pi*yy)]))
                    cache[key]=(wx,wy,bases)
                wx,wy,bases=cache[key];area=x.new_zeros(ny,nx);moments=x.new_zeros(10,ny,nx)
                for basis,begin in zip(bases,range(0,h,config.pool_rows)):
                    end=min(h,begin+config.pool_rows);q=keep[begin:end];value=torch.where(q,x[begin:end],0)
                    moments= moments+(wy[:,begin:end]@(basis*value))@wx.T
                    area=area+(wy[:,begin:end]@q.to(x.dtype))@wx.T
                moments=(moments/area.clamp_min(1e-12)).permute(1,2,0).reshape(ny*nx,10)
                return moments,area.flatten()>0,area.flatten()/((right-left)*(top-bottom)/(ny*nx))
        batched.pool_native=pooled
        try:return super().encode_views(images)
        finally:batched.pool_native=original

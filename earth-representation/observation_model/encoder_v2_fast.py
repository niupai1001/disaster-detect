"""Equivalent per-forward SRF reuse and larger execution chunks; same parameters."""
from dataclasses import replace
from types import MethodType
import math

import torch
from torch.nn import functional as F

from .encoder_v2 import EncoderConfig,ObservationEncoderV2,NativeFeatureBank
from .encoder_v2 import position_features
from .encoder_v2_tasks import run_task as original_run_task,sample_target_indices


def native_bank(self,band,grid):
    gx,gy=band.support;left,bottom,right,top=grid['bounds'];h,w=band.values.shape
    _,_,cx,_,_,cy=band.transform;device=band.values.device
    x=(torch.arange(w,device=device,dtype=torch.float64)*gx+(cx-left)+gx/2).float()
    y=(torch.arange(h,device=device,dtype=torch.float64)*gy+(top-cy)+gy/2).float()
    xx=x[None].expand(h,w)/(right-left);yy=y[:,None].expand(h,w)/(top-bottom)
    keep=band.keep();value=torch.where(keep,band.values,0)
    descriptor=(self.local_descriptor(torch.cat([band.srf,torch.log(band.srf.new_tensor([gx,gy])/10)]))
        if self.local_projection.condition_mode!='off' else band.values.new_zeros(self.config.local_width))
    # Reuse only within this forward: parameters may change at the next update.
    weight=self.local_projection.generated_weight(band.srf,band.support)
    features=[];coordinates=[]
    for begin in range(0,h,self.config.pool_rows):
        end=min(h,begin+self.config.pool_rows);u,v=xx[begin:end],yy[begin:end];q=keep[begin:end]
        basis=torch.stack([torch.ones_like(u),u,v,u*u,v*v,u*v,
            torch.sin(2*math.pi*u),torch.cos(2*math.pi*u),torch.sin(2*math.pi*v),torch.cos(2*math.pi*v)],-1)
        basis=(basis*value[begin:end,:,None])[q]
        features.append(F.gelu(F.linear(basis,weight,self.local_projection.shared.bias))+descriptor)
        xy=torch.stack([x[None].expand(end-begin,w),y[begin:end,None].expand(end-begin,w)],-1)
        coordinates.append(xy[q])
    index_map=torch.full(keep.shape,-1,device=device,dtype=torch.long)
    index_map[keep]=torch.arange(int(keep.sum()),device=device)
    return NativeFeatureBank(torch.cat(features),torch.cat(coordinates),band.support,band.transform,band.crs,band.band_key,index_map)


class FastObservationEncoderV2(ObservationEncoderV2):
    def __init__(self,config=None,condition_mode='learned'):
        config=replace(config or EncoderConfig(),pool_rows=256)
        super().__init__(config,condition_mode)
        self.encoder.native_bank=MethodType(native_bank,self.encoder)


def decode_many(decoder,representation,targets,indices):
    grid=representation.grids[0];ny,nx=grid['shape'];left,bottom,right,top=grid['bounds']
    tokens=representation.features[0,:ny*nx];valid=representation.valid[0,:ny*nx];context=tokens[valid].mean(0)
    coordinates=[];conditions=[];supports=[];lengths=[]
    for target,ids in zip(targets,indices):
        a,b,c,d,e,f=target.transform;assert not b and not d and target.crs==grid['crs']
        h,w=target.shape
        px=(ids%w).double()*a+(c-left)+a/2;py=(ids//w).double()*(-e)+(top-f)-e/2
        coordinates.append(torch.stack([px,py],-1));lengths.append(len(ids))
        condition=decoder.condition(torch.cat([target.srf.to(tokens.device),torch.log(tokens.new_tensor([a,-e])/10)]))
        conditions.append(condition.expand(len(ids),-1));supports.append(tokens.new_tensor([a,-e]).expand(len(ids),-1))
    coordinates=torch.cat(coordinates);conditions=torch.cat(conditions);supports=torch.cat(supports)
    xy=(coordinates/coordinates.new_tensor([right-left,top-bottom])).to(tokens.dtype)
    center_x=torch.floor(xy[:,0]*nx).long();center_y=torch.floor(xy[:,1]*ny).long()
    offsets=torch.tensor([(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]],device=tokens.device)
    outputs=[];coverage=[]
    for begin in range(0,len(xy),8192):
        end=min(begin+8192,len(xy));yy=center_y[begin:end,None]+offsets[:,0];xx=center_x[begin:end,None]+offsets[:,1]
        inside=(yy>=0)&(yy<ny)&(xx>=0)&(xx<nx);keys=yy.clamp(0,ny-1)*nx+xx.clamp(0,nx-1)
        keep=inside&valid[keys];k=tokens[keys]*keep[...,None];empty=~keep.any(1)
        k=k.clone();k[:,0]=torch.where(empty[:,None],context,k[:,0]);keep=keep.clone();keep[:,0]|=empty
        query_coordinates=coordinates[begin:end].float();scale=supports[begin:end].float();local=[]
        for bank in representation.optional_local_features[0]:
            count=min(decoder.config.local_neighbors,len(bank.features))
            with torch.no_grad():best=decoder.nearest_native(bank,query_coordinates,grid['bounds'])
            relative=bank.relative_xy_m[best]-query_coordinates[:,None]
            relative=torch.cat([torch.tanh(relative/(4*scale[:,None])),
                torch.log(query_coordinates.new_tensor(bank.native_pixel_support_m)/scale)[:,None].expand(-1,count,-1)],-1)
            local.append(decoder.local_key(bank.features[best])+decoder.local_relative(relative))
        local=torch.cat(local,1);k=torch.cat([k,local],1);keep=torch.cat([keep,torch.ones(local.shape[:2],device=tokens.device,dtype=torch.bool)],1)
        q=(conditions[begin:end]+decoder.position(position_features(xy[begin:end])))[:,None]
        for block in decoder.blocks:q=block(q,k,keep)
        observed=keep.any(1)&(xy[begin:end]>=0).all(1)&(xy[begin:end]<=1).all(1)
        outputs.append(decoder.output(q[:,0])[:,0]*observed);coverage.append(observed)
    output=torch.cat(outputs);covered=torch.cat(coverage)
    return list(zip(output.split(lengths),covered.split(lengths)))


def fast_run_task(model,task,observation,statistics,save_predictions=False,target_sample_pixels=None,logical_update=0):
    if len(task.targets)==1 or target_sample_pixels is None:
        return original_run_task(model,task,observation,statistics,save_predictions,target_sample_pixels,logical_update)
    indices=[sample_target_indices(t,observation['id'],task.kind,target_sample_pixels,logical_update) for t in task.targets]
    encode=model.encode;forward=model.decoder.forward;predictions=[];cursor=0
    def encode_with_queries(image):
        representation=encode(image)
        predictions.extend(decode_many(model.decoder,representation,[t.grid for t in task.targets],indices))
        return representation
    def cached_forward(representation,target,index=0,pixel_indices=None):
        nonlocal cursor
        assert target is task.targets[cursor].grid and index==0
        result=predictions[cursor];cursor+=1;return result
    # Each training process owns one model and executes tasks serially. Keep the
    # original loss/mask/resource implementation and replace only query execution.
    model.encode=encode_with_queries;model.decoder.forward=cached_forward
    try:return original_run_task(model,task,observation,statistics,save_predictions,target_sample_pixels,logical_update)
    finally:model.encode=encode;model.decoder.forward=forward

"""G2 optical encoder: physical conditioning, area-supported fusion, local attention.

No training entry point. Native values never become an interpolated image.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import AttentionBlock, CrossBlock


@dataclass(frozen=True)
class EncoderConfig:
    width: int = 128
    heads: int = 4
    ffn: int = 512
    spatial_blocks: int = 4
    window: int = 8
    srf_bins: int = 64
    rank: int = 8
    reference_tokens: int = 400
    reference_area_m2: float = 1200.**2
    reference_gsd_m: float = 10.
    alpha: float = 1.
    max_tokens: int = 8192
    pool_rows: int = 64
    query_chunk: int = 256


@dataclass
class OpticalBand:
    values: torch.Tensor
    valid: torch.Tensor
    transform: tuple
    crs: str
    srf: torch.Tensor
    band_key: str                    # audit/filter only; never an embedding ID
    visible: torch.Tensor | None = None

    @property
    def support(self):
        a,b,_,d,e,_=self.transform
        if b or d or a<=0 or e>=0:raise ValueError('G2 requires projected north-up native grids')
        return a,-e

    @property
    def bounds(self):
        a,_,c,_,e,f=self.transform;h,w=self.values.shape
        return c,f+e*h,c+a*w,f

    def keep(self):
        return self.valid if self.visible is None else self.valid & self.visible


@dataclass
class OpticalImage:
    bands: list[OpticalBand]
    product_id: str = ''


@dataclass
class TargetGrid:
    shape: tuple
    transform: tuple
    crs: str
    srf: torch.Tensor
    band_key: str = ''


@dataclass
class EncoderRepresentation:
    features: torch.Tensor           # B,N,D; padded N, real lengths below
    footprints: torch.Tensor         # B,N,4 projected bounds, float64
    valid: torch.Tensor
    scale_m: torch.Tensor             # sqrt(cell area), not source resolution
    source_support: list
    grids: list
    metadata: list = field(default_factory=list)


def srf_descriptor(wavelength, response, bins=64):
    """Fixed spectral-bin integrals; does not reconstruct/resample the source SRF."""
    w,r=np.asarray(wavelength,float),np.asarray(response,float)
    if not ((np.diff(w)>0).all() and (r>=0).all()):raise ValueError('Unsorted or negative SRF')
    increments=np.diff(w)*(r[:-1]+r[1:])/2
    cumulative=np.r_[0.,np.cumsum(increments)]
    if abs(cumulative[-1]-1)>1e-6:raise ValueError('G1 normalized SRF required')
    edges=np.linspace(300.,2600.,bins+1)
    # Integral of the piecewise-linear response, including partial edge intervals.
    indices=np.searchsorted(w,edges,side='right')-1
    k=np.clip(indices,0,len(w)-2);delta=np.clip(edges-w[k],0,w[k+1]-w[k])
    c=cumulative[k]+r[k]*delta+(r[k+1]-r[k])*delta**2/(2*(w[k+1]-w[k]))
    c[edges<=w[0]]=0;c[edges>=w[-1]]=cumulative[-1]
    values=np.diff(c)
    if abs(values.sum()-1)>1e-6:raise ValueError('SRF outside frozen 300..2600nm basis')
    return torch.tensor(values,dtype=torch.float32)


def image_from_g1(item, device='cpu'):
    bands=[]
    for band in item['bands']:
        with np.load(Path(band['spectral_response_ref']['path']),allow_pickle=False) as curve:
            descriptor=srf_descriptor(curve['spectral_wavelength_grid_nm'],curve['normalized_response'])
        bands.append(OpticalBand(torch.as_tensor(band['reflectance'],device=device),
            torch.as_tensor(band['pixel_valid'],device=device),tuple(band['transform']),band['crs'],
            descriptor.to(device),band['band_key']))
    return OpticalImage(bands,item['product_id'])


def target_from_band(band):
    return TargetGrid(tuple(band.values.shape),band.transform,band.crs,band.srf,band.band_key)


def token_grid(bounds, gsd, config):
    left,bottom,right,top=bounds;w,h=right-left,top-bottom
    raw=max(1,round(config.reference_tokens*w*h/config.reference_area_m2*(config.reference_gsd_m/gsd)**config.alpha))
    target=min(raw,config.max_tokens)
    ny=max(1,min(config.max_tokens,round(math.sqrt(target*h/w))))
    nx=max(1,min(config.max_tokens//ny,round(target/ny)))
    return dict(shape=(ny,nx),bounds=tuple(bounds),raw_tokens=raw,actual_tokens=ny*nx,
        rounding_delta=ny*nx-raw,token_budget_capped=raw>config.max_tokens,finest_visible_support_m=gsd)


class PhysicalProjection(nn.Module):
    """W(c)=W_shared+U diag(h(c)) V, operating on local spectral-spatial moments."""
    def __init__(self,config,condition_mode='learned'):
        super().__init__();d,r=config.width,config.rank
        if condition_mode not in ['learned','frozen','off']:raise ValueError(condition_mode)
        self.condition_mode=condition_mode
        self.shared=nn.Linear(10,d)
        self.u=nn.Parameter(torch.randn(d,r)*.02);self.v=nn.Parameter(torch.randn(r,10)*.02)
        self.hyper=nn.Sequential(nn.Linear(config.srf_bins+2,64),nn.GELU(),nn.Linear(64,r))
        if condition_mode=='frozen':self.hyper.requires_grad_(False)

    def coefficients(self,srf,support):
        c=torch.cat([srf,torch.log(srf.new_tensor(support)/10)])
        return self.hyper(c) if self.condition_mode!='off' else c.new_zeros(self.u.shape[1])

    def generated_weight(self,srf,support):
        return self.shared.weight+(self.u*self.coefficients(srf,support))@self.v

    def forward(self,moments,srf,support):
        return F.gelu(F.linear(moments,self.generated_weight(srf,support),self.shared.bias))


def axis_overlap(cell_edges,pixel_lo,pixel_hi):
    return (torch.minimum(cell_edges[1:,None],pixel_hi[None])-torch.maximum(cell_edges[:-1,None],pixel_lo[None])).clamp_min(0)


def pool_native(band,grid,config):
    """Area-integrate projected native local moments, bounded by source row chunks.

    Linear projection commutes with this integration; no D-wide native image is
    materialized. Spatial moments keep the within-cell position in the input path.
    """
    x=band.values;keep=band.keep();device=x.device;h,w=x.shape
    gx,gy=band.support;left,bottom,right,top=grid['bounds'];ny,nx=grid['shape']
    _,_,c,_,_,f=band.transform
    # Subtract the world origin in float64 before using float32 local coordinates.
    px=(torch.arange(w,device=device,dtype=torch.float64)*gx+(c-left)).to(x.dtype)
    py=(torch.arange(h,device=device,dtype=torch.float64)*gy+(top-f)).to(x.dtype)
    ex=torch.linspace(0,right-left,nx+1,device=device,dtype=x.dtype)
    ey=torch.linspace(0,top-bottom,ny+1,device=device,dtype=x.dtype)
    wx=axis_overlap(ex,px,px+gx);wy=axis_overlap(ey,py,py+gy)
    u=(px+gx/2)/(right-left);v=(py+gy/2)/(top-bottom)
    area=x.new_zeros(ny,nx);pooled=x.new_zeros(10,ny,nx)
    for begin in range(0,h,config.pool_rows):
        end=min(begin+config.pool_rows,h);q=keep[begin:end]
        value=torch.where(q,x[begin:end],0)
        xx=u[None].expand(end-begin,w);yy=v[begin:end,None].expand(end-begin,w)
        basis=torch.stack([torch.ones_like(xx),xx,yy,xx*xx,yy*yy,xx*yy,
            torch.sin(2*math.pi*xx),torch.cos(2*math.pi*xx),torch.sin(2*math.pi*yy),torch.cos(2*math.pi*yy)])
        pooled=pooled+(wy[:,begin:end]@(basis*value))@wx.T
        area=area+(wy[:,begin:end]@q.to(x.dtype))@wx.T
    moments=(pooled/area.clamp_min(1e-12)).permute(1,2,0).reshape(ny*nx,10)
    valid=area.flatten()>0
    return moments,valid,area.flatten()/((right-left)*(top-bottom)/(ny*nx))


def position_features(xy):
    return torch.cat([xy,torch.sin(2*math.pi*xy),torch.cos(2*math.pi*xy)],-1)


class OpticalEncoderV2(nn.Module):
    def __init__(self,config=None,condition_mode='learned'):
        super().__init__();self.config=config or EncoderConfig();c=self.config
        self.projection=PhysicalProjection(c,condition_mode)
        self.position=nn.Linear(6,c.width);self.query=nn.Parameter(torch.zeros(c.width))
        self.fusion=CrossBlock(c.width,c.heads)
        self.blocks=nn.ModuleList(AttentionBlock(c.width,c.heads,c.ffn) for _ in range(c.spatial_blocks))

    def spatial(self,x,keep,ny,nx):
        d=self.config.width;window=min(self.config.window,ny,nx)
        for index,block in enumerate(self.blocks):
            shift=window//2 if index%2 else 0
            bottom=(-(ny+shift))%window;right=(-(nx+shift))%window
            z=F.pad(x.reshape(ny,nx,d),(0,0,shift,right,shift,bottom))
            q=F.pad(keep.reshape(ny,nx),(shift,right,shift,bottom))
            hh,ww=q.shape
            z=z.reshape(hh//window,window,ww//window,window,d).permute(0,2,1,3,4).reshape(-1,window*window,d)
            q=q.reshape(hh//window,window,ww//window,window).permute(0,2,1,3).reshape(-1,window*window)
            z=block(z,q).reshape(hh//window,ww//window,window,window,d).permute(0,2,1,3,4).reshape(hh,ww,d)
            x=z[shift:shift+ny,shift:shift+nx].reshape(ny*nx,d)
        return x

    def encode_one(self,image,excluded_bands=()):
        # Filter whole targets before any metadata pooling, GSD selection or values.
        bands=[b for b in image.bands if b.band_key not in excluded_bands and bool(b.keep().any())]
        if not bands:raise ValueError('No actual visible optical observation')
        if len({b.crs for b in bands})!=1:raise ValueError('Different CRSs need an explicit geometry transform before fusion')
        if any(b.values.ndim!=2 or b.values.shape!=b.valid.shape for b in bands):raise ValueError('Native values/QA shape mismatch')
        bounds=(min(b.bounds[0] for b in bands),min(b.bounds[1] for b in bands),max(b.bounds[2] for b in bands),max(b.bounds[3] for b in bands))
        grid=token_grid(bounds,min(min(b.support) for b in bands),self.config);grid['crs']=bands[0].crs
        features=[];keeps=[];support=[]
        for band in bands:
            moments,valid,coverage=pool_native(band,grid,self.config)
            features.append(self.projection(moments,band.srf,band.support)*valid[:,None]);keeps.append(valid)
            support.append(dict(band_key=band.band_key,native_pixel_support_m=band.support,
                native_shape=tuple(band.values.shape),native_transform=band.transform,
                source_coverage_fraction=coverage.detach()))
        keys=torch.stack(features,1);mask=torch.stack(keeps,1);keep=mask.any(1)
        ny,nx=grid['shape'];left,bottom,right,top=bounds
        device=keys.device;dtype=keys.dtype
        yy,xx=torch.meshgrid(torch.arange(ny,device=device),torch.arange(nx,device=device),indexing='ij')
        xy=torch.stack([(xx.flatten()+.5)/nx,(yy.flatten()+.5)/ny],-1).to(dtype)
        x=self.fusion((self.query+self.position(position_features(xy)))[:,None],keys,mask)[:,0]*keep[:,None]
        x=self.spatial(x,keep,ny,nx)
        ex=torch.linspace(left,right,nx+1,device=device,dtype=torch.float64)
        ey=torch.linspace(top,bottom,ny+1,device=device,dtype=torch.float64)
        footprint=torch.stack([ex[xx.flatten()],ey[yy.flatten()+1],ex[xx.flatten()+1],ey[yy.flatten()]],-1)
        return x,keep,footprint,support,grid

    def encode(self,images,excluded_bands=()):
        if isinstance(images,OpticalImage):images=[images]
        encoded=[self.encode_one(image,excluded_bands) for image in images]
        count=max(len(row[0]) for row in encoded)
        features=torch.stack([F.pad(x,(0,0,0,count-len(x))) for x,_,_,_,_ in encoded])
        valid=torch.stack([F.pad(q,(0,count-len(q))) for _,q,_,_,_ in encoded])
        footprints=torch.stack([F.pad(p,(0,0,0,count-len(p))) for _,_,p,_,_ in encoded])
        scale=((footprints[...,2]-footprints[...,0])*(footprints[...,3]-footprints[...,1])).clamp_min(0).sqrt()
        return EncoderRepresentation(features,footprints,valid,scale,[r[3] for r in encoded],[r[4] for r in encoded])

    forward=encode


class NativeGridDecoder(nn.Module):
    """Read H and target physical metadata only; target arrays are not arguments."""
    def __init__(self,config=None):
        super().__init__();self.config=config or EncoderConfig();c=self.config
        self.condition=nn.Sequential(nn.Linear(c.srf_bins+2,c.width),nn.GELU(),nn.Linear(c.width,c.width))
        self.position=nn.Linear(6,c.width)
        self.blocks=nn.ModuleList(CrossBlock(c.width,c.heads) for _ in range(2))
        self.output=nn.Linear(c.width,1)

    def forward(self,representation,target,index=0):
        grid=representation.grids[index]
        if target.crs!=grid['crs']:raise ValueError('Target CRS differs; explicit coordinate transformation required')
        a,b,c,d,e,f=target.transform
        if b or d or a<=0 or e>=0:raise ValueError('North-up physical target grid required')
        h,w=target.shape;ny,nx=grid['shape'];left,bottom,right,top=grid['bounds']
        tokens=representation.features[index,:ny*nx];valid=representation.valid[index,:ny*nx]
        context=tokens[valid].mean(0)
        device=tokens.device;dtype=tokens.dtype
        ids=torch.arange(h*w,device=device);px=(ids%w).to(torch.float64)*a+(c-left)+a/2
        py=(ids//w).to(torch.float64)*(-e)+(top-f)-e/2
        xy=torch.stack([px/(right-left),py/(top-bottom)],-1).to(dtype)
        center_x=torch.floor(xy[:,0]*nx).long();center_y=torch.floor(xy[:,1]*ny).long()
        offsets=torch.tensor([(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]],device=device)
        condition=self.condition(torch.cat([target.srf.to(device),torch.log(tokens.new_tensor([a,-e])/10)]))
        outputs=[];coverage=[]
        for begin in range(0,len(ids),self.config.query_chunk):
            end=min(begin+self.config.query_chunk,len(ids))
            yy=center_y[begin:end,None]+offsets[:,0];xx=center_x[begin:end,None]+offsets[:,1]
            inside=(yy>=0)&(yy<ny)&(xx>=0)&(xx<nx)
            keys=yy.clamp(0,ny-1)*nx+xx.clamp(0,nx-1);keep=inside&valid[keys]
            k=tokens[keys]*keep[...,None]
            # A large hidden region still has a prediction, using only existing H.
            # This bounded global summary never reads hidden target values/QA.
            empty=~keep.any(1)
            k=k.clone();k[:,0]=torch.where(empty[:,None],context,k[:,0])
            keep=keep.clone();keep[:,0] |= empty
            q=(condition+self.position(position_features(xy[begin:end])))[:,None]
            for block in self.blocks:q=block(q,k,keep)
            observed=keep.any(1)&(xy[begin:end]>=0).all(1)&(xy[begin:end]<=1).all(1)
            outputs.append(self.output(q[:,0])[:,0]*observed);coverage.append(observed)
        return torch.cat(outputs).reshape(h,w),torch.cat(coverage).reshape(h,w)


class ObservationEncoderV2(nn.Module):
    def __init__(self,config=None,condition_mode='learned'):
        super().__init__();self.config=config or EncoderConfig()
        self.encoder=OpticalEncoderV2(self.config,condition_mode);self.decoder=NativeGridDecoder(self.config)

    def encode(self,images,excluded_bands=()):return self.encoder.encode(images,excluded_bands)

    def predict(self,image,target,excluded_bands=()):
        return self.decoder(self.encode(image,excluded_bands),target)

"""Shared spectral/spatial tokens and a single local target-query decoder."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BandRaster:
    name: str
    value: torch.Tensor                 # B,H,W; physical SR on actual native grid
    valid: torch.Tensor                 # B,H,W
    wavelength_nm: float
    bandwidth_nm: float | None
    native_gsd_m: int
    phase_xy_m: tuple = (0.,0.)          # first pixel center relative to footprint


@dataclass
class Representation:
    tokens: torch.Tensor                # B,C,subtile_y,subtile_x,D
    valid: torch.Tensor
    footprint_m: int
    band_names: tuple


@dataclass
class TargetQuery:
    sensor: str
    band: str
    wavelength_nm: float
    bandwidth_nm: float | None
    native_gsd_m: int
    grid_gsd_m: int
    phase_xy_m: tuple | None = None


def band_descriptor(wavelength,bandwidth,native_gsd,device):
    return torch.tensor([wavelength/2500.,(bandwidth or 0.)/500.,float(bandwidth is not None),native_gsd/60.],device=device)


class AttentionBlock(nn.Module):
    def __init__(self,width=128,heads=4,ffn=512):
        super().__init__()
        self.norm=nn.LayerNorm(width);self.attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norm2=nn.LayerNorm(width);self.ffn=nn.Sequential(nn.Linear(width,ffn),nn.GELU(),nn.Linear(ffn,width))

    def forward(self,x,valid):
        # All-invalid windows still need a finite dummy key; discard its output.
        mask=~valid.clone();empty=mask.all(1);mask[empty,0]=False
        q=self.norm(x)
        x=x+self.attn(q,q,q,key_padding_mask=mask,need_weights=False)[0]
        return (x+self.ffn(self.norm2(x)))*valid[...,None]


class NativeBandTokenizer(nn.Module):
    def __init__(self,width=128,tile=60):
        super().__init__();self.width=width;self.tile=tile
        self.projections=nn.ModuleDict({str(g):nn.Linear(2*(tile//g)**2,4*width) for g in [10,20,30]})
        self.descriptor=nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.Linear(width,width))
        self.position=nn.Linear(6,width)

    def forward(self,band,footprint):
        g=band.native_gsd_m;p=self.tile//g
        if g not in [10,20,30] or band.value.shape[-2:]!=(footprint//g,)*2:
            raise ValueError("native band grid must match declared 10/20/30m support")
        b=band.value.shape[0];n=footprint//self.tile
        value=torch.where(band.valid,band.value,0)
        values=F.unfold(value[:,None],p,stride=p).transpose(1,2)
        valid=F.unfold(band.valid[:,None].float(),p,stride=p).transpose(1,2)
        tokens=self.projections[str(g)](torch.cat([values,valid],-1)).reshape(b,n,n,2,2,self.width)
        tokens=tokens.permute(0,1,3,2,4,5).reshape(b,2*n,2*n,self.width)
        coverage=F.interpolate(band.valid[:,None].float(),size=(2*n,2*n),mode="area")[:,0]
        token_valid=coverage>0
        xy=torch.stack(torch.meshgrid(torch.arange(2*n,device=value.device),torch.arange(2*n,device=value.device),indexing="ij"),-1).flip(-1).float()
        xy=(xy+.5)/(2*n)
        phase=torch.as_tensor(band.phase_xy_m,device=value.device,dtype=value.dtype)/self.tile
        if phase.ndim==1:phase=phase.expand(b,2)
        pos=torch.cat([xy.expand(b,-1,-1,-1),xy.sin().expand(b,-1,-1,-1),phase[:,None,None].expand(b,2*n,2*n,2)],-1)
        desc=self.descriptor(band_descriptor(band.wavelength_nm,band.bandwidth_nm,g,value.device))
        return (tokens+desc+self.position(pos))*token_valid[...,None],token_valid


class SharedObservationTransformer(nn.Module):
    def __init__(self,width=128,heads=4,ffn=512):
        super().__init__();self.width=width
        self.spectral=nn.ModuleList(AttentionBlock(width,heads,ffn) for _ in range(2))
        self.spatial=nn.ModuleList(AttentionBlock(width,heads,ffn) for _ in range(4))

    def spatial_forward(self,x,valid,block,shift):
        b,c,h,w,d=x.shape;window=min(8,h,w)
        left=window//2 if shift else 0
        bottom=(-(h+left))%window;right=(-(w+left))%window
        x=F.pad(x,(0,0,left,right,left,bottom))
        keep=F.pad(valid,(left,right,left,bottom))
        hh,ww=x.shape[2:4]
        windows=x.reshape(b*c,hh//window,window,ww//window,window,d).permute(0,1,3,2,4,5).reshape(-1,window*window,d)
        masks=keep.reshape(b*c,hh//window,window,ww//window,window).permute(0,1,3,2,4).reshape(-1,window*window)
        out=block(windows,masks).reshape(b*c,hh//window,ww//window,window,window,d).permute(0,1,3,2,4,5).reshape(b,c,hh,ww,d)
        return out[:,:,left:left+h,left:left+w]

    def forward(self,x,valid):
        b,c,h,w,d=x.shape
        for i,block in enumerate(self.spatial):
            if i<2:
                seq=x.permute(0,2,3,1,4).reshape(b*h*w,c,d)
                keep=valid.permute(0,2,3,1).reshape(b*h*w,c)
                x=self.spectral[i](seq,keep).reshape(b,h,w,c,d).permute(0,3,1,2,4)
            x=self.spatial_forward(x,valid,block,i%2==1)
        return x


class CrossBlock(nn.Module):
    def __init__(self,width=128,heads=4):
        super().__init__();self.qnorm=nn.LayerNorm(width);self.knorm=nn.LayerNorm(width)
        self.attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norm=nn.LayerNorm(width);self.ffn=nn.Sequential(nn.Linear(width,512),nn.GELU(),nn.Linear(512,width))

    def forward(self,q,k,valid):
        mask=~valid.clone();empty=mask.all(1);mask[empty,0]=False
        q=q+self.attn(self.qnorm(q),self.knorm(k),self.knorm(k),key_padding_mask=mask,need_weights=False)[0]
        return q+self.ffn(self.norm(q))


class TargetQueryDecoder(nn.Module):
    def __init__(self,width=128,heads=4,coordinate_mode="legacy_local"):
        super().__init__()
        if coordinate_mode not in ["legacy_local","explicit_global"]:raise ValueError(coordinate_mode)
        self.coordinate_mode=coordinate_mode
        self.descriptor=nn.Sequential(nn.Linear(6,width),nn.GELU(),nn.Linear(width,width))
        self.position=nn.Sequential(nn.Linear(8 if coordinate_mode=="explicit_global" else 6,width),nn.GELU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList(CrossBlock(width,heads) for _ in range(2))
        self.output=nn.Linear(width,1)

    def forward(self,representation,query,tile_indices=None):
        x,valid=representation.tokens,representation.valid
        b,c,h,w,d=x.shape;grid=query.grid_gsd_m;tile=60
        if tile%grid:raise ValueError("v1 target grid must divide the60m physical tile")
        n=representation.footprint_m//tile;p=tile//grid
        # Group four subtokens per physical tile, then unfold3x3 tile neighborhoods.
        tiles=x.reshape(b,c,n,2,n,2,d).permute(0,2,4,1,3,5,6).reshape(b,n,n,c*4,d)
        keep=valid.reshape(b,c,n,2,n,2).permute(0,2,4,1,3,5).reshape(b,n,n,c*4)
        padded=F.pad(tiles.permute(0,3,4,1,2),(1,1,1,1))
        vp=F.pad(keep.permute(0,3,1,2),(1,1,1,1))
        indices=torch.arange(n*n,device=x.device) if tile_indices is None else tile_indices
        count=len(indices);yy=indices//n;xx=indices%n
        keys=torch.stack([padded[:,:,:,yy+dy,xx+dx] for dy in range(3) for dx in range(3)],3).permute(0,4,3,1,2).reshape(b*count,9*c*4,d)
        mask=torch.stack([vp[:,:,yy+dy,xx+dx] for dy in range(3) for dx in range(3)],2).permute(0,3,2,1).reshape(b*count,9*c*4)
        phase=(grid/2,grid/2) if query.phase_xy_m is None else query.phase_xy_m
        phase=torch.as_tensor(phase,device=x.device,dtype=x.dtype)
        if phase.ndim==1:phase=phase.expand(b,2)
        local=torch.stack(torch.meshgrid(torch.arange(p,device=x.device),torch.arange(p,device=x.device),indexing="ij"),-1).flip(-1).reshape(-1,2)*grid
        local=(local[None]+phase[:,None])/tile
        pos=torch.cat([local,torch.sin(math.pi*local),torch.cos(math.pi*local)],-1)
        desc=torch.cat([band_descriptor(query.wavelength_nm,query.bandwidth_nm,query.native_gsd_m,x.device),torch.tensor([grid/60.,float(query.sensor.startswith("S2"))],device=x.device)])
        if self.coordinate_mode=="explicit_global":
            # Local pixel coordinates retain fine detail; explicit window coordinates identify the requested tile.
            local_pos=pos[:,None].expand(b,count,p*p,6)
            origin=torch.stack([xx,yy],-1).to(local.dtype)*tile
            global_xy=(origin[None,:,None]+local[:,None]*tile)/representation.footprint_m
            q=(self.position(torch.cat([local_pos,global_xy],-1))+self.descriptor(desc)).reshape(b*count,p*p,d)
        else:q=(self.position(pos)+self.descriptor(desc))[:,None].expand(b,count,-1,-1).reshape(b*count,p*p,d)
        # Chunk query neighborhoods to keep full1200m prediction within GPU memory.
        out=[]
        for begin in range(0,len(q),64):
            z=q[begin:begin+64]
            for block in self.blocks:z=block(z,keys[begin:begin+64],mask[begin:begin+64])
            out.append(self.output(z).squeeze(-1))
        prediction=torch.cat(out).reshape(b,count,p*p)
        coverage=mask.any(1).reshape(b,count,1).expand(b,count,p*p)
        if tile_indices is None:
            prediction=prediction.reshape(b,n,n,p,p).permute(0,1,3,2,4).reshape(b,n*p,n*p)
            coverage=coverage.reshape(b,n,n,p,p).permute(0,1,3,2,4).reshape(b,n*p,n*p)
        return prediction,coverage


class ObservationModel(nn.Module):
    def __init__(self,footprint_m=1200,width=128,coordinate_mode="legacy_local"):
        super().__init__();self.footprint_m=footprint_m
        self.tokenizer=NativeBandTokenizer(width);self.encoder=SharedObservationTransformer(width)
        self.decoder=TargetQueryDecoder(width,coordinate_mode=coordinate_mode)

    def encode(self,bands):
        if not bands:raise ValueError("at least one actually visible band is required")
        values,masks=zip(*(self.tokenizer(b,self.footprint_m) for b in bands))
        x=torch.stack(values,1);valid=torch.stack(masks,1)
        return Representation(self.encoder(x,valid),valid,self.footprint_m,tuple(b.name for b in bands))

    def predict_bands(self,bands,query):
        # Target identity is removed before any numerical input path.
        visible=[band for band in bands if band.name!=query.band]
        return self.decoder(self.encode(visible),query)


class CNNBlock(nn.Module):
    def __init__(self,width,hidden):
        super().__init__()
        self.net=nn.Sequential(nn.GroupNorm(1,width),nn.Conv2d(width,hidden,1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,width,1))
    def forward(self,x):return x+self.net(x)


class ControlledCNN(nn.Module):
    def __init__(self,width,hidden):
        super().__init__();self.blocks=nn.Sequential(*(CNNBlock(width,hidden) for _ in range(4)))
    def forward(self,x,valid):
        weights=valid[...,None].to(x.dtype)
        fused=(x*weights).sum(1)/weights.sum(1).clamp_min(1)
        value=self.blocks(fused.permute(0,3,1,2)).permute(0,2,3,1)
        # Repeat only the fused representation, making its early aggregation explicit.
        return value[:,None].expand_as(x)*valid[...,None]


def checkpoint_coordinate_mode(state):
    weights=state.get("model",state)
    weight=weights["decoder.position.0.weight"]
    return {6:"legacy_local",8:"explicit_global"}[weight.shape[1]]


def make_model(variant="transformer",footprint_m=1200,coordinate_mode="legacy_local"):
    torch.manual_seed(0)
    model=ObservationModel(footprint_m,coordinate_mode=coordinate_mode)
    if variant=="cnn":
        target=sum(p.numel() for p in model.encoder.parameters())
        # Capacity calibration only, not an architecture-performance sweep.
        def parameter_count(hidden):return 4*(2*128+128*hidden+hidden+9*hidden*hidden+hidden+hidden*128+128)
        hidden=min(range(8,513),key=lambda h:abs(parameter_count(h)-target))
        model.encoder=ControlledCNN(128,hidden)
        actual=sum(p.numel() for p in model.encoder.parameters())
        if abs(actual/target-1)>.1:raise ValueError("CNN capacity mismatch")
        model.capacity_match=dict(transformer_encoder=target,cnn_encoder=actual,hidden=hidden)
    elif variant=="random":
        model.tokenizer.requires_grad_(False);model.encoder.requires_grad_(False)
    elif variant!="transformer":raise ValueError(variant)
    return model

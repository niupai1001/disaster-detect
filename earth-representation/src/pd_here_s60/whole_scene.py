"""Native whole-observation 60m spatial ViT with a shared transformer."""
from dataclasses import dataclass
from math import ceil, isclose
from functools import lru_cache

import torch
from torch import nn
from rasterio.crs import CRS
from torch.nn.attention import SDPBackend, sdpa_kernel

from pd_here_geotoken_dualvit_r3.model import Block, grid_position


@lru_cache(maxsize=256)
def metric_crs(text):
    crs = CRS.from_user_input(text)
    if not crs.is_projected or not isclose(crs.linear_units_factor[1],1.):
        raise ValueError(f'Expected projected metres, got {text}')
    return crs


def native_geometry(bands, patch_m=60):
    """Union of the supplied native rasters and fixed-size native-pixel slots."""
    crs = metric_crs(bands[0][3])
    for _,shape,affine,projection in bands:
        dx,r,_,s,dy,_ = affine
        if r or s or dx<=0 or dy>=0 or metric_crs(projection)!=crs or min(shape)<1:
            raise ValueError('Expected common north-up native raster grids')
    left = min(g[2][2] for g in bands)
    top = max(g[2][5] for g in bands)
    right = max(a[2]+a[0]*w for _,(_,w),a,_ in bands)
    bottom = min(a[5]+a[4]*h for _,(h,_),a,_ in bands)
    grid = (ceil((top-bottom)/patch_m-1e-8), ceil((right-left)/patch_m-1e-8))
    layout = tuple((key,ceil(patch_m/-a[4]-1e-8),ceil(patch_m/a[0]-1e-8)) for key,_,a,_ in bands)
    return (left,top),grid,layout


def native_axis(length, spacing, shift, capacity, patch_m=60):
    # The local slot is the pixel's order in its physical 60m cell.
    centers = shift+(torch.arange(length,dtype=torch.float64)+.5)*spacing
    cell = ((centers+1e-7)/patch_m).floor().long()
    i = torch.arange(length)
    starts = torch.cat((torch.ones(1,dtype=torch.bool),cell[1:]!=cell[:-1]))
    slot = i-torch.where(starts,i,0).cummax(0).values
    if int(slot.max())>=capacity:
        raise ValueError('Native pixel slots are too narrow')
    return cell,slot


@dataclass
class Scene:
    grid: tuple
    layout: tuple
    values: torch.Tensor
    qa: torch.Tensor
    origin: tuple
    valid_pixels: int
    patch_m: int = 60
    subcells: torch.Tensor | None = None
    metadata: object | None = None

    def to(self, device):
        return Scene(self.grid,self.layout,self.values.to(device),self.qa.to(device),self.origin,self.valid_pixels,
                     self.patch_m,None if self.subcells is None else self.subcells.to(device),
                     None if self.metadata is None else self.metadata.to(device))


def pack_scene(observation, scale, patch_m=60):
    bands = observation.bands
    meta = [(b.key,tuple(b.grid.shape),tuple(b.grid.affine),b.grid.crs) for b in bands]
    (left,top),(gh,gw),layout = native_geometry(meta,patch_m)
    widths = [h*w for _,h,w in layout]
    values = torch.zeros(gh*gw,sum(widths))
    qa = torch.zeros_like(values,dtype=torch.bool)
    offset = 0
    for b,(_,ph,pw),length in zip(bands,layout,widths,strict=True):
        dx,_,x,_,dy,y = b.grid.affine
        h,w = b.grid.shape
        iy,sy = native_axis(h,-dy,top-y,ph,patch_m)
        ix,sx = native_axis(w,dx,x-left,pw,patch_m)
        patch = iy[:,None]*gw+ix
        slot = offset+sy[:,None]*pw+sx
        if int(iy.max())>=gh or int(ix.max())>=gw:
            raise ValueError('Native pixel falls outside scene')
        values[patch,slot] = torch.where(b.valid,b.values,0.)/scale
        qa[patch,slot] = b.valid
        offset += length
    valid_pixels = sum(int(b.valid.sum()) for b in bands)
    if int(qa.sum())!=valid_pixels:
        raise ValueError('Native QA support changed during packing')
    subcells = None
    if patch_m==120 and int(qa.any(-1).sum())==1:
        subcells = torch.zeros_like(qa,dtype=torch.uint8)
        offset = 0
        for b,(_,ph,pw),length in zip(bands,layout,widths,strict=True):
            dx,_,x,_,dy,y = b.grid.affine
            h,w = b.grid.shape
            iy,sy = native_axis(h,-dy,top-y,ph,120)
            ix,sx = native_axis(w,dx,x-left,pw,120)
            jy,_ = native_axis(h,-dy,top-y,ph,60)
            jx,_ = native_axis(w,dx,x-left,pw,60)
            subcells[iy[:,None]*gw+ix,offset+sy[:,None]*pw+sx] = ((jy[:,None]%2)*2+jx%2).byte()
            offset += length
    return Scene((gh,gw),layout,values,qa,(left,top),valid_pixels,patch_m,subcells)


def tile_bounds(scene, limit=512):
    gh,gw = scene.grid
    h_step = limit//max(h for _,h,_ in scene.layout)
    w_step = limit//max(w for _,_,w in scene.layout)
    if min(h_step,w_step)<1:
        raise ValueError('One physical patch already exceeds the native output limit')
    return [(r,min(r+h_step,gh),c,min(c+w_step,gw))
            for r in range(0,gh,h_step) for c in range(0,gw,w_step)]


def tiles(scene, keep):
    bounds = tile_bounds(scene)
    if len(bounds)==1:
        yield scene,keep,slice(None)
        return
    gh,gw = scene.grid
    ids = torch.arange(gh*gw,device=keep.device).reshape(gh,gw)
    for r0,r1,c0,c1 in bounds:
        index = ids[r0:r1,c0:c1].flatten()
        metadata = None if scene.metadata is None else scene.metadata.crop(r0,r1,c0,c1)
        yield Scene((r1-r0,c1-c0),scene.layout,scene.values[index],scene.qa[index],
                    (scene.origin[0]+scene.patch_m*c0,scene.origin[1]-scene.patch_m*r0),0,
                    scene.patch_m,metadata=metadata),keep[index],index


class LayoutProjection(nn.Module):
    def __init__(self, layout, means, *, joint_input=True, decoder_width=128):
        super().__init__()
        widths = tuple(h*w for _,h,w in layout)
        self.widths = widths
        if joint_input:
            self.input_values = nn.Linear(sum(widths),128)
            self.input_qa = nn.Linear(sum(widths),128,bias=False)
            self.availability = nn.Linear(len(widths),128,bias=False)
        self.decoder_availability = nn.Linear(len(widths),decoder_width,bias=False)
        self.output = nn.Linear(decoder_width,sum(widths))
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.copy_(torch.repeat_interleave(torch.tensor(means),torch.tensor(widths)))
        self.register_buffer('band_slots',torch.repeat_interleave(torch.arange(len(widths)),torch.tensor(widths)))


class WholeSceneViT(nn.Module):
    projection_type = LayoutProjection

    def __init__(self, layouts, *, decoder_width=128):
        super().__init__()
        self.decoder_width = decoder_width
        self.layout_indices = {key:str(i) for i,key in enumerate(layouts)}
        decoder_options = {'decoder_width': decoder_width} if decoder_width != 128 else {}
        self.projections = nn.ModuleDict({self.layout_indices[k]:self.projection_type(k,v,**decoder_options)
                                          for k,v in layouts.items()})
        self.spatial_block = Block(128)
        self.encoder_norm = nn.LayerNorm(128,eps=1e-6)
        self.decoder_embed = nn.Linear(128,decoder_width)
        self.mask_token = nn.Parameter(torch.empty(1,decoder_width))
        nn.init.normal_(self.mask_token,std=.02)
        self.decoder_blocks = nn.ModuleList([Block(decoder_width),Block(decoder_width)])
        self.decoder_norm = nn.LayerNorm(decoder_width,eps=1e-6)

    def forward(self, scene, keep, bands):
        a = self.projections[self.layout_indices[scene.layout]]
        structural = scene.qa.any(-1)
        position = grid_position(scene.grid,self.decoder_width).to(scene.values.device)[structural]
        qa = scene.qa[structural]
        evidence = qa & bands[a.band_slots]
        selected = keep[structural] & evidence.any(-1)
        input_qa = evidence[selected]
        x = torch.where(input_qa,scene.values[structural][selected],0.)
        x = a.input_values(x)+a.input_qa(input_qa.float())+a.availability(bands.float())+position[selected]
        # Efficient SDPA evaluates exact global attention without storing L x L matrices.
        if x.is_cuda:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                x = self.transform(x,selected,position,a.decoder_availability(bands.float()))
        else:
            x = self.transform(x,selected,position,a.decoder_availability(bands.float()))
        return a.output(x)

    def transform(self, tokens, selected, position, availability):
        encoded = self.encoder_norm(self.spatial_block(tokens[None])[0]) if len(tokens) else tokens
        return self.decode_latent(encoded,selected,position,availability)

    def decode_latent(self, encoded, selected, position, availability):
        restore = self.mask_token.expand(len(position),-1).clone()
        if len(encoded):
            restore[selected] = self.decoder_embed(encoded).to(restore.dtype)
        decoded = (restore+position+availability)[None]
        for block in self.decoder_blocks:
            decoded = block(decoded)
        return self.decoder_norm(decoded[0])

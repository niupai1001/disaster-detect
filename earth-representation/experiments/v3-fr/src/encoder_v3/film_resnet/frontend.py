"""Native ordered values, shared content lift, and masked stride-one ResNet."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from encoder_v3.latent_response.types import NativeGrid


@dataclass(frozen=True)
class VisibleBand:
    """Cleaned [H,W] float and boolean mask, with complete native grid."""
    clean: Tensor
    visible: Tensor
    grid: NativeGrid

    @classmethod
    def from_native(cls, values, qa, visible, grid):
        if values.shape != grid.shape or not values.is_floating_point():
            raise ValueError("Floating native grid values required")
        if any(m.shape != values.shape or m.dtype != torch.bool or m.device != values.device
               for m in (qa, visible)):
            raise ValueError("QA and visible masks must match the native tensor")
        if bool((visible & ~qa).any()):
            raise ValueError("Visibility must be a subset of QA")
        clean = torch.where(visible, values, 0.)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("Nonfinite visible input")
        return cls(clean, visible, grid)


@dataclass(frozen=True)
class PackedBand:
    """[16, Hp, Wp] values and masks; phases are a_y*r+a_x in original north-up rows."""
    values: Tensor
    visible: Tensor
    structural: Tensor
    grid: NativeGrid
    factor: int

    @property
    def observed(self):
        return self.visible.any(dim=0, keepdim=True)


def factor_for(grid, metres=40.):
    if not math.isfinite(metres) or metres <= 0:
        raise ValueError("A positive pack threshold in metres is required")
    return max((r for r in (1, 2, 4) if r*max(grid.gsd_xy) <= metres), default=1)


def pack(band, metres=40.):
    r = factor_for(band.grid, metres)
    h, w = band.grid.shape
    # Pixel-unshuffle has channel, phase-y, phase-x ordering.
    def phases(x):
        x = F.pad(x, (0, (-w)%r, 0, (-h)%r))
        return F.pad(x.reshape(h//r + bool(h%r), r, w//r + bool(w%r), r)
                     .permute(1,3,0,2).flatten(0,1), (0,0,0,0,0,16-r*r))

    return PackedBand(phases(band.clean), phases(band.visible),
                      phases(torch.ones_like(band.visible)), band.grid, r)


def unpack(phases, packed):
    r = packed.factor
    h,w = packed.grid.shape
    _,hp,wp = phases.shape
    return phases[:r*r].reshape(r,r,hp,wp).permute(2,0,3,1).reshape(hp*r,wp*r)[:h,:w]


class ContentLift(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = nn.Conv2d(32, 16, 1)
        self.residual = nn.Sequential(nn.Conv2d(48, 32, 1), nn.GELU(), nn.Conv2d(32, 32, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, values, visible, structural):
        mask = torch.cat((visible,structural),1).to(values.dtype)
        initial = torch.cat((values,self.mask(mask)),1)
        features = initial + .1*self.residual(torch.cat((values,mask),1))
        return torch.where(visible.any(1,keepdim=True),features,0.)


def channel_norm(norm, x):
    return norm(x.movedim(1,-1)).movedim(-1,1)


class ResidualBlock(nn.Module):
    """Masked 3x3 convolutions each preserve original observed parent support."""
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(32), nn.LayerNorm(32)
        self.film = nn.Linear(32,64)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.conv1 = nn.Conv2d(32,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,32,3,padding=1)
        with torch.no_grad():
            self.conv2.weight.mul_(.1)
            self.conv2.bias.zero_()
        self.scale = nn.Parameter(torch.tensor(.1))

    def forward(self, x, observed, denominator, affine):
        def masked(layer, values):
            y = F.conv2d(torch.where(observed, values, 0.),layer.weight,padding=1)
            return torch.where(observed, y/denominator+layer.bias[None,:,None,None], 0.)

        z = channel_norm(self.norm1, x)
        if affine is not None:
            gamma,beta = affine.chunk(2,dim=1)
            z = (1+gamma[...,None,None])*z+beta[...,None,None]
        z = masked(self.conv1,F.gelu(z))
        z = masked(self.conv2,F.gelu(channel_norm(self.norm2,z)))
        return torch.where(observed,x+self.scale*z,0.)


class Frontend(nn.Module):
    """One set of residual weights shared by all bands and all legal geometries."""
    def __init__(self, blocks=2, film=True):
        super().__init__()
        if type(blocks) is not int or blocks not in (0,2) or type(film) is not bool:
            raise ValueError("Registered frontend requires zero or two blocks and an explicit FiLM switch")
        self.lift = ContentLift()
        self.blocks = nn.ModuleList(ResidualBlock() for _ in range(blocks))
        self.film = film
        if not film:
            for block in self.blocks:
                block.film.requires_grad_(False)

    @property
    def halo(self):
        return 2*len(self.blocks)

    def forward(self, packed, descriptor, tile=None):
        if not packed or descriptor.shape != (len(packed),32):
            raise ValueError("Each packed band needs one 32-dimensional descriptor")
        values, visible, structural = (torch.stack([getattr(p,k) for p in packed])
                                       for k in ("values","visible","structural"))
        affine = [block.film(descriptor) if self.film else None for block in self.blocks]
        def run(u,v,s):
            observed = v.any(1,keepdim=True)
            x = self.lift(u,v,s)
            denominator = F.avg_pool2d(observed.to(x.dtype),3,stride=1,padding=1).clamp_min(1/9)
            for block,cond in zip(self.blocks,affine,strict=True):
                x = block(x,observed,denominator,cond)
            return x
        if tile is None:
            return run(values,visible,structural)
        th,tw = tile
        if min(th,tw) < 1:
            raise ValueError("Positive tile dimensions required")
        h,w = values.shape[-2:]
        halo = self.halo
        rows = []
        for y in range(0,h,th):
            y0,y1 = max(0,y-halo), min(h,y+th+halo)
            columns = []
            for x in range(0,w,tw):
                x0,x1 = max(0,x-halo), min(w,x+tw+halo)
                result = run(*(t[...,y0:y1,x0:x1] for t in (values,visible,structural)))
                columns.append(result[...,y-y0:min(y+th,h)-y0,x-x0:min(x+tw,w)-x0])
            rows.append(torch.cat(columns,dim=-1))
        return torch.cat(rows,dim=-2)

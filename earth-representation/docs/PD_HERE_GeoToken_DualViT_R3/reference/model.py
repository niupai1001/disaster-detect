"""CPU reference for GeoToken-DualViT R3. Not a geospatial loader or trainer.

Input values/QA: one [batch, geographic_patches, H_b, W_b] tensor per
canonical band. Native patch shapes and band ordering are fixed by Layout.
A deletes whole geographic patches before both branches. B1 deletes bands
from both branches, then scatters only computed features into fixed zero
slots before the linear reducers. One joint token per geographic patch.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Layout:
    band_keys: tuple[str, ...]
    native_shapes: tuple[tuple[int, int], ...]
    patch_grid: tuple[int, int]
    subpatch: int = 2

    def __post_init__(self):
        if not self.band_keys or len(set(self.band_keys)) != len(self.band_keys):
            raise ValueError('Canonical bands must be nonempty and unique')
        if len(self.band_keys) != len(self.native_shapes):
            raise ValueError('One native shape per band is required')
        if min(self.patch_grid) <= 0 or self.subpatch <= 0:
            raise ValueError('Positive grid and subpatch sizes required')
        if any(min(s) <= 0 or s[0] % self.subpatch or s[1] % self.subpatch
               for s in self.native_shapes):
            raise ValueError('This reference expects divisible, prevalidated native shapes')

    @property
    def patches(self):
        return math.prod(self.patch_grid)

    @property
    def pixels_per_patch(self):
        return sum(math.prod(s) for s in self.native_shapes)

    @property
    def spatial_tokens_per_patch(self):
        return sum(math.prod(s) // self.subpatch ** 2 for s in self.native_shapes)


def sincos_1d(position: Tensor, width: int) -> Tensor:
    if width % 2:
        raise ValueError('Position width must be even')
    frequency = torch.exp(-math.log(10000.) * torch.arange(
        width // 2, device=position.device, dtype=torch.float32) / (width // 2))
    phase = position.float()[..., None] * frequency
    return torch.cat((phase.sin(), phase.cos()), -1)


def grid_position(shape: tuple[int, int], width: int) -> Tensor:
    if width % 4:
        raise ValueError('2D positional width must be divisible by four')
    yy, xx = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
    return torch.cat((sincos_1d(xx.flatten(), width // 2),
                      sincos_1d(yy.flatten(), width // 2)), -1)


def to_subpatch(x: Tensor, side: int = 2) -> Tensor:
    """[K,H,W] -> [K,M,side**2], row-major blocks and row-major pixels."""
    k, h, w = x.shape
    if h % side or w % side:
        raise ValueError('Input must be padded to divisible dimensions before tokenization')
    return x.reshape(k, h//side, side, w//side, side).permute(0,1,3,2,4).reshape(k,-1,side*side)


class Block(nn.Module):
    def __init__(self, width: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        safe = None
        if valid is not None:
            safe = valid.clone()
            empty = ~safe.any(-1)
            safe[empty, 0] = True  # numerical dummy; result is zeroed below
        h = self.norm1(x)
        a = self.attention(h, h, h, key_padding_mask=None if safe is None else ~safe,
                           need_weights=False)[0]
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x if valid is None else torch.where(valid[..., None], x, 0.)


class GeoTokenDualViT(nn.Module):
    def __init__(self, layout: Layout, width: int = 128, decoder_width: int = 64,
                 heads: int = 4):
        super().__init__()
        self.layout, self.width, self.decoder_width = layout, width, decoder_width
        n, d = len(layout.band_keys), width
        self.spectral_values = nn.ModuleDict()
        self.spectral_qa = nn.ModuleDict()
        for h,w in set(layout.native_shapes):
            key = f'{h}x{w}'
            self.spectral_values[key] = nn.Linear(h*w, d)
            self.spectral_qa[key] = nn.Linear(h*w, d, bias=False)
        self.spatial_values = nn.Linear(layout.subpatch**2, d)
        self.spatial_qa = nn.Linear(layout.subpatch**2, d, bias=False)
        self.spectral_block = Block(d, heads)
        self.spatial_block = Block(d, heads)
        self.spectral_reduce = nn.Linear(n*d, d)
        self.spatial_reduce = nn.Linear(layout.spatial_tokens_per_patch*d, d)
        self.fuse = nn.Linear(2*d, d)
        self.availability = nn.Linear(n, d, bias=False)
        self.global_block = Block(d, heads)
        self.encoder_norm = nn.LayerNorm(d, eps=1e-6)
        self.decoder_embed = nn.Linear(d, decoder_width)
        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_width))
        nn.init.normal_(self.mask_token, std=.02)
        self.decoder_availability = nn.Linear(n, decoder_width, bias=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_width, heads) for _ in range(2)])
        self.decoder_norm = nn.LayerNorm(decoder_width, eps=1e-6)
        self.prediction = nn.Linear(decoder_width, layout.pixels_per_patch)
        self.register_buffer('band_position', sincos_1d(torch.arange(n), d))
        self.register_buffer('geo_position', grid_position(layout.patch_grid, d))
        self.register_buffer('decoder_position', grid_position(layout.patch_grid, decoder_width))
        for j,(h,w) in enumerate(layout.native_shapes):
            self.register_buffer(f'local_position_{j}', grid_position((h//layout.subpatch,w//layout.subpatch),d))

    def forward(self, values: tuple[Tensor,...], qa: tuple[Tensor,...],
                keep_geo: Tensor, band_visible: Tensor) -> dict:
        layout, d = self.layout, self.width
        n, p = len(layout.band_keys), layout.patches
        if len(values) != n or len(qa) != n:
            raise ValueError('Use the exact frozen band order')
        batch = values[0].shape[0]
        if keep_geo.shape != (batch,p) or band_visible.shape != (batch,n):
            raise ValueError('Visibility shapes do not match layout')
        if keep_geo.dtype != torch.bool or band_visible.dtype != torch.bool:
            raise TypeError('Visibility masks must be boolean')
        kp, kb = keep_geo.sum(-1), band_visible.sum(-1)
        if (kp==0).any() or not (kp==kp[0]).all() or (kb==0).any() or not (kb==kb[0]).all():
            raise ValueError('Reference batches need a fixed nonzero visible-patch and visible-band count')
        pv, nv = int(kp[0]), int(kb[0])
        ids_keep = keep_geo.nonzero(as_tuple=False)[:,1].reshape(batch,pv)
        band_ids = band_visible.nonzero(as_tuple=False)[:,1].reshape(batch,nv)
        spectral, spatial, supports = [], [], []
        computed_pairs = 0
        for j,(x,q,(h,w)) in enumerate(zip(values,qa,layout.native_shapes,strict=True)):
            if x.shape != (batch,p,h,w) or q.shape != x.shape or q.dtype != torch.bool:
                raise ValueError('Native shapes/QA mismatch')
            active = keep_geo & band_visible[:,j,None]
            pairs = active.nonzero(as_tuple=False)
            ns = h*w//layout.subpatch**2
            e = x.new_zeros(batch,p,d)
            v = x.new_zeros(batch,p,ns,d)
            support = torch.zeros(batch,p,dtype=torch.bool,device=x.device)
            if len(pairs):
                bi,pi = pairs.unbind(-1)
                x0, q0 = x[bi,pi], q[bi,pi]
                safe = torch.where(q0,x0,torch.zeros_like(x0))
                if not torch.isfinite(safe).all():
                    raise ValueError('Nonfinite visible observations')
                key = f'{h}x{w}'
                ej = self.spectral_values[key](safe.flatten(1)) + self.spectral_qa[key](q0.flatten(1).to(x.dtype))
                present = q0.flatten(1).any(-1)
                ej = torch.where(present[:,None],ej,0.)
                e[bi,pi] = ej
                sub, subqa = to_subpatch(safe,layout.subpatch),to_subpatch(q0,layout.subpatch)
                vj = self.spatial_values(sub) + self.spatial_qa(subqa.to(x.dtype))
                vj = vj + getattr(self,f'local_position_{j}')[None] + self.band_position[j]
                vj = self.spatial_block(vj,subqa.any(-1))
                v[bi,pi], support[bi,pi] = vj,present
                computed_pairs += len(pairs)
            spectral.append(e)
            spatial.append(v)
            supports.append(support)
        spectral = torch.stack(spectral,2)       # [B,P,N,d], missing values are zero scratch slots
        support = torch.stack(supports,2)         # [B,P,N]
        spatial = torch.cat(spatial,2)            # [B,P,sum M,d]
        batch_ids = torch.arange(batch,device=values[0].device)[:,None]
        e = spectral[batch_ids,ids_keep]
        supported = support[batch_ids,ids_keep]
        v = spatial[batch_ids,ids_keep]
        gather_b = band_ids[:,None,:,None].expand(batch,pv,nv,d)
        evis = torch.gather(e,2,gather_b) + self.band_position[band_ids][:,None]
        valid_b = torch.gather(supported,2,band_ids[:,None].expand(batch,pv,nv))
        if not valid_b.any(-1).all():
            raise ValueError('Each retained geographic patch needs some observed support')
        evis = self.spectral_block(evis.reshape(batch*pv,nv,d),valid_b.reshape(batch*pv,nv)).reshape(batch,pv,nv,d)
        fixed = torch.zeros_like(e).scatter(2,gather_b,evis)
        s = self.spectral_reduce(fixed.flatten(2))
        t = self.spatial_reduce(v.flatten(2))
        geo = self.fuse(torch.cat((t,s),-1)) + self.availability(band_visible.float())[:,None]
        encoded = self.encoder_norm(self.global_block(geo+self.geo_position[ids_keep]))
        full = self.mask_token.expand(batch,p,-1).clone()
        full[batch_ids,ids_keep] = self.decoder_embed(encoded)
        decoder_input = full + self.decoder_position[None] + self.decoder_availability(band_visible.float())[:,None]
        out = decoder_input
        for block in self.decoder_blocks:
            out = block(out)
        prediction = self.prediction(self.decoder_norm(out))
        return dict(prediction=prediction,encoded=encoded,ids_keep=ids_keep,
                    pre_position_decoder=full,decoder_input=decoder_input,
                    trace=dict(geographic_encoder_length=pv,decoder_length=p,
                               spectral_attention_length=nv,computed_band_patches=computed_pairs))


def masked_reconstruction_loss(prediction: Tensor, truth: tuple[Tensor,...],
                               qa: tuple[Tensor,...], keep_geo: Tensor,
                               band_visible: Tensor, task: str) -> Tensor:
    """Pixel -> target band -> parent, then mean over parents. No patch mean weights."""
    if task not in ('A','B1'):
        raise ValueError('Expected A or B1')
    batch,p,_ = prediction.shape
    band_losses, band_used = [], []
    offset = 0
    for j,(y,q) in enumerate(zip(truth,qa,strict=True)):
        width = y.shape[-2]*y.shape[-1]
        pred = prediction[...,offset:offset+width].reshape_as(y).float()
        offset += width
        mask = (~keep_geo)[:,:,None,None] if task=='A' else (~band_visible[:,j])[:,None,None,None]
        mask = mask & q
        safe_y = torch.where(mask,y.float(),0.)
        delta = torch.where(mask,pred-safe_y,0.)
        count = mask.flatten(1).sum(-1)
        loss = delta.square().flatten(1).sum(-1)/count.clamp_min(1)
        band_losses.append(loss)
        band_used.append(count>0)
    if offset != prediction.shape[-1]:
        raise ValueError('Prediction width is not the native layout width')
    losses, used = torch.stack(band_losses,-1), torch.stack(band_used,-1)
    if not used.any(-1).all():
        raise ValueError('A parent has no valid hidden supervision')
    return ((losses*used).sum(-1)/used.sum(-1)).mean()

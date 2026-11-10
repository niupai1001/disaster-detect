"""Native frontend and permutation-invariant channel fusion, without decoder skips."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .dynamic_conv import masked_native_conv
from .native_patch import aggregate_microcells
from .geometry import native_cells, fourier2d


class LocalEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 128)
        y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
        self.register_buffer('position', fourier2d(torch.stack((x.flatten(), y.flatten()), -1).float()/4+.125-.5).flatten())

    def forward(self, u, support):
        clean = torch.where(support[..., None]>0, u, 0)
        position = self.position.expand(*support.shape[:-1], 256)
        return self.linear(torch.cat((clean.flatten(-2), support, position), -1))


def channel_front(values, keep, kernel, bias, channel, geometry, embed, tile_rows):
    patches = geometry.shape[0]*geometry.shape[1]
    total = values.new_zeros(patches, 16, 32)
    support = values.new_zeros(patches, 16)
    height = values.shape[0]
    for begin in range(0, height, tile_rows):
        end = min(begin+tile_rows, height)
        if not bool(keep[begin:end].any()):
            continue
        low, high = max(0, begin-1), min(height, end+1)
        feature = masked_native_conv(values[None, low:high], keep[None, low:high], kernel, bias)
        feature = feature[0, :, begin-low:end-low].permute(1, 2, 0).reshape(-1, 32)
        result = aggregate_microcells(feature, native_cells(channel.grid, values.device, begin, end),
                                     keep[begin:end].flatten(), origin_xy=geometry.origin, patch_size=geometry.patch_size)
        index = result['patch'][:, 1]*geometry.shape[1]+result['patch'][:, 0]
        total = total.index_add(0, index, result['U']*result['support'][..., None])
        support = support.index_add(0, index, result['support'])
    u = total/support.clamp_min(torch.finfo(values.dtype).tiny)[..., None]
    return embed(u, support), support


def grouped_channel_front(values,keep,kernels,bias,plans,geometry,embed,tile_rows):
    """Same exact area integral as channel_front, batched over a common native grid."""
    group,height,width=values.shape;patches=geometry.shape[0]*geometry.shape[1]
    total=values.new_zeros(group,patches*16,32);support=values.new_zeros(group,patches*16)
    finite=torch.ones((),device=values.device,dtype=torch.bool)
    for begin,plan in zip(range(0,height,tile_rows),plans):
        end=min(begin+tile_rows,height);low,high=max(0,begin-1),min(height,end+1)
        feature=masked_native_conv(values[:,low:high],keep[:,low:high],kernels,bias)
        feature=feature[:,:,begin-low:end-low].permute(0,2,3,1).reshape(group,-1,32)
        visible=keep[:,begin:end].reshape(group,-1)
        clean=torch.where(visible[...,None],feature,0)
        finite=finite & torch.isfinite(clean).all()
        source,destination=plan['source'],plan['destination']
        area=plan['area'].to(values.dtype)[None]*visible[:,source]
        total=total.index_add(1,destination,clean[:,source]*area[...,None])
        support=support.index_add(1,destination,area)
    if not bool(finite):raise ValueError('Nonfinite visible native feature')
    u=(total/support.clamp_min(torch.finfo(values.dtype).tiny)[...,None]).reshape(group,patches,16,32)
    support=support.reshape(group,patches,16)/(geometry.patch_size/4)**2
    return embed(u,support),support


class ChannelFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(128+32+16, 128)
        self.value = nn.Linear(128+32+16, 128)
        self.summary_query = nn.Parameter(torch.randn(128)*.02)
        self.latent_query = nn.Linear(6, 128)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.)

    def forward(self, h, support, condition):
        # h[P,C,128], support[P,C,16]; canonical channel order gives reproducible reductions.
        valid = (support>0).any(-1)
        content = valid.any(-1)
        ids = content.nonzero().flatten()
        if not len(ids):
            raise ValueError('NO_VISIBLE_CONTEXT: no synthetic blank content is permitted')
        inputs = torch.cat((h[ids], condition[None].expand(len(ids), -1, -1), support[ids]), -1)
        counts = valid[ids].sum(-1)
        r = counts.float().sqrt().ceil().long().clamp(max=16)
        slots = torch.arange(int(r.max()), device=h.device)
        u = (slots[None]+.5)/r[:, None]
        phase = 2*math.pi*u[..., None]*h.new_tensor([1., 2., 4.])
        latent_q = self.latent_query(torch.cat((phase.sin(), phase.cos()), -1))
        query = torch.cat((self.summary_query[None, None].expand(len(ids), 1, -1), latent_q), 1)
        result, _ = self.attention(query, self.key(inputs), self.value(inputs), key_padding_mask=~valid[ids], need_weights=False)
        latent_valid = slots[None]<r[:, None]
        latent = torch.where(latent_valid[..., None], result[:, 1:], 0)
        return ids, result[:, 0], latent, latent_valid, counts

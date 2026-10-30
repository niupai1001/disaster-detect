"""Conservative shared physical mask: intersecting source pixels are hidden everywhere."""
import copy
import torch

from .physical import area_grid


def mask_asset(asset, hidden, grid):
    matrix,shape,_,_=area_grid(asset,grid)
    if tuple(hidden.shape)!=shape: raise ValueError('Shared mask shape mismatch')
    touched=torch.sparse.mm(matrix.transpose(0,1),hidden.float().flatten()[:,None]).squeeze(1)>0
    result=copy.copy(asset)
    result.pixel_valid=asset.pixel_valid & ~touched.reshape(asset.image.shape[-2:]).to(asset.image.device)[None]
    result.image=torch.where(result.pixel_valid,asset.image,0)
    return result,touched.reshape(asset.image.shape[-2:])


@torch.no_grad()
def update_ema(teacher,student,momentum):
    if not 0<=momentum<=1: raise ValueError('Invalid EMA momentum')
    for (_,target),(_,source) in zip(teacher.named_parameters(),student.named_parameters(),strict=True):
        target.lerp_(source.detach(),1-momentum)
    for (_,target),(_,source) in zip(teacher.named_buffers(),student.named_buffers(),strict=True):
        target.copy_(source)
    teacher.requires_grad_(False).eval()

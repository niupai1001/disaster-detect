"""Feature grids carry their CRS; label mapping never assumes equal array extent."""
import numpy as np
import torch
from pyproj import Transformer
from torch.nn import functional as F


def to_label_grid(level,transform,crs,shape):
    if level.crs is None:raise ValueError('Feature CRS required')
    a,b,c,d,e,f=transform
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    px,py=Transformer.from_crs(crs,level.crs,always_xy=True).transform(a*(xx+.5)+b*(yy+.5)+c,d*(xx+.5)+e*(yy+.5)+f)
    a,b,c,d,e,f=level.transform
    inverse=np.linalg.inv(np.array([[a,b],[d,e]]))
    uv=inverse @ np.stack((px-c,py-f)).reshape(2,-1)
    h,w=level.tensor.shape[-2:]
    grid=np.stack((2*uv[0]/w-1,2*uv[1]/h-1),-1).reshape(*shape,2)
    grid=torch.as_tensor(grid,dtype=level.tensor.dtype,device=level.tensor.device)[None]
    values=F.grid_sample(level.tensor[None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]
    valid=F.grid_sample(level.valid_mask[None,None].float(),grid,mode='nearest',padding_mode='zeros',align_corners=False)[0,0]>.5
    return values,valid

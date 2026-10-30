"""Area-supported local path. Native semantic anchors are a separate P0 gate."""
import math

import numpy as np
import torch
from torch import nn
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box

from . import FeatureLevel, USEOutput


def area_grid(asset, grid=None):
    """Intersect each real affine pixel polygon with fixed 30m cells in local AEQD."""
    asset.validate()
    _, h, w = asset.image.shape
    a,b,c,d,e,f = asset.sample_grid['transform']
    crs = CRS(asset.sample_grid['crs'])
    lon,lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(a*w/2+b*h/2+c,d*w/2+e*h/2+f)
    local = CRS(grid['crs']) if grid else CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    project = Transformer.from_crs(crs, local, always_xy=True)
    yy,xx = np.mgrid[:h+1,:w+1]
    gx,gy = project.transform(a*xx+b*yy+c,d*xx+e*yy+f)
    x0,y0 = math.floor(gx.min()/120)*120, math.floor(gy.min()/120)*120
    nw,nh = math.ceil((gx.max()-x0)/120)*4, math.ceil((gy.max()-y0)/120)*4
    if grid:
        x0,y0=grid['origin']; nh,nw=grid['shape']
        if nh%4 or nw%4: raise ValueError('Shared grid must contain complete 120m cells')
    ri,ci,areas = [],[],[]
    for y in range(h):
        for x in range(w):
            p = Polygon([(gx[Y,X],gy[Y,X]) for Y,X in ((y,x),(y,x+1),(y+1,x+1),(y+1,x))])
            left,bottom,right,top = p.bounds
            for v in range(max(0,math.floor((bottom-y0)/30)),min(nh,math.ceil((top-y0)/30))):
                for u in range(max(0,math.floor((left-x0)/30)),min(nw,math.ceil((right-x0)/30))):
                    area = p.intersection(box(x0+u*30,y0+v*30,x0+(u+1)*30,y0+(v+1)*30)).area
                    if area > 1e-8:
                        ri.append(v*nw+u); ci.append(y*w+x); areas.append(area)
    matrix = torch.sparse_coo_tensor(torch.tensor([ri,ci],dtype=torch.long),torch.tensor(areas,dtype=torch.float32),
                                     (nh*nw,h*w),check_invariants=True).coalesce()
    return matrix,(nh,nw),(30.,0.,x0,0.,30.,y0),local.to_wkt()


class PhysicalLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.optical = nn.Linear(1,192)
        self.sar = nn.Linear(1,192)
        self.polarization=nn.Embedding(2,192)
        self.channel = nn.Linear(6,192)
        self.position = nn.Linear(5,192)
        self.aggregate = nn.TransformerEncoder(nn.TransformerEncoderLayer(192,6,768,0.,batch_first=True),2,
                                               enable_nested_tensor=False)

    def forward(self, asset, hidden_cells=None, grid=None):
        if asset.modality == 'hr':
            raise ValueError('HR is read-only in this release')
        matrix, (h,w), transform, crs = area_grid(asset,grid)
        matrix = matrix.to(asset.image.device)
        x = torch.where(asset.pixel_valid,asset.image,0).flatten(1)
        valid = asset.pixel_valid.flatten(1).float()
        support = torch.sparse.mm(matrix,valid.T).T
        means = torch.sparse.mm(matrix,x.T).T / support.clamp_min(1e-8)
        visible = support > 0
        if hidden_cells is not None:
            if hidden_cells.shape != (h,w):
                raise ValueError('Mask must use the declared metric grid')
            visible = visible & ~hidden_cells.flatten()[None,:]
        projection = self.sar if asset.modality == 's1' else self.optical
        features = projection(means[:,:,None])
        if asset.modality=='s1':
            if any(name.lower() not in ('vv','vh') for name in asset.channel_names):raise ValueError('Unknown SAR polarization')
            ids=torch.tensor([('vv','vh').index(name.lower()) for name in asset.channel_names],device=x.device)
            features=features+self.polarization(ids)[:,None,:]
        if asset.modality != 's1':
            metadata = torch.cat((torch.where(asset.metadata_known,asset.channel_metadata,0),
                                  asset.metadata_known.float()),-1)
            features = features+self.channel(metadata)[:,None,:]
        features = (features*visible[:,:,None]).sum(0)/visible.sum(0).clamp_min(1)[:,None]
        yy,xx = torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
        a,b,_,d,e,_ = asset.sample_grid['transform']
        gsd = math.sqrt(abs(a*e-b*d))
        coords = torch.stack((xx*30+15,yy*30+15,torch.full_like(xx,gsd,dtype=torch.float32),
                              torch.full_like(xx,30),torch.full_like(xx,120)), -1).float().reshape(-1,5)
        features = features+self.position(coords/120)
        mask = visible.any(0).reshape(h,w)
        patches = features.reshape(h//4,4,w//4,4,192).permute(0,2,1,3,4).reshape(-1,16,192)
        pmask = mask.reshape(h//4,4,w//4,4).permute(0,2,1,3).reshape(-1,16)
        nonempty = pmask.any(1)
        encoded = torch.zeros_like(patches)
        if nonempty.any():
            encoded[nonempty] = self.aggregate(patches[nonempty],src_key_padding_mask=~pmask[nonempty])
        area_support=support.max(0).values.reshape(h,w)*mask
        patch_area=area_support.reshape(h//4,4,w//4,4).permute(0,2,1,3).reshape(-1,16)
        pooled = (encoded*patch_area[:,:,None]).sum(1)/patch_area.sum(1).clamp_min(1e-8)[:,None]
        tensor = pooled.reshape(h//4,w//4,192).permute(2,0,1)
        local_tensor=encoded.reshape(h//4,w//4,4,4,192).permute(0,2,1,3,4).reshape(h,w,192).permute(2,0,1)
        local_level=FeatureLevel(local_tensor,mask,transform,30.,120.,asset.modality,
                                 asset.native_resolution,True,crs)
        level = FeatureLevel(tensor,nonempty.reshape(h//4,w//4),
                             (120.,0.,transform[2],0.,120.,transform[5]),120.,120.,
                             asset.modality,asset.native_resolution,True,crs)
        global_embedding = pooled[nonempty].mean(0) if nonempty.any() else None
        count=asset.pixel_valid.sum(0).clamp_min(1)
        fine=projection((torch.where(asset.pixel_valid,asset.image,0).sum(0)/count)[:,:,None])
        if asset.modality!='s1':
            channel=self.channel(metadata)
        else:channel=self.polarization(ids)
        fine=fine+(channel[:,:,None,None]*asset.pixel_valid[:,None]).sum(0).permute(1,2,0)/count[:,:,None]
        fine=fine.permute(2,0,1)*asset.pixel_valid.any(0)
        detail=FeatureLevel(fine,asset.pixel_valid.any(0),tuple(asset.sample_grid['transform']),gsd,gsd,
                            asset.modality,asset.native_resolution,
                            asset.native_resolution is None or any(v is None or abs(v-gsd)>.1 for v in asset.native_resolution),
                            asset.sample_grid['crs'])
        return USEOutput(native_detail=detail,metric_local=local_level,metric_patch=level,shared_metric_tokens=level,global_embedding=global_embedding)

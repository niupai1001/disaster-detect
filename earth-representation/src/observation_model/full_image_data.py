"""Read complete local native rasters; right/bottom invalid padding, never crop or resize."""
import json
import math
import hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject,transform_bounds
from rasterio.enums import Resampling
import torch
from .native_data import BAD,aerosol_valid


def raster(path):
    with rasterio.open(path) as f:
        x=f.read(1);valid=np.isfinite(x)
        if f.nodata is not None:valid&=x!=f.nodata
        return x,valid,f.transform,f.crs


def read_full(row):
    parent=Path(row['bands'][0]['source']).parent
    if row['sensor']=='S2':
        qa,qv,qt,qc=raster(parent/'SCL.tif');qv&=~np.isin(qa,BAD)
    else:
        qa,qv,qt,qc=raster(parent/'qa_pixel.tif');qv&=(qa.astype(np.uint16)&63)==0
        sat,sv,st,sc=raster(parent/'qa_radsat.tif');a,av,at,ac=raster(parent/'qa_aerosol.tif')
        if st!=qt or at!=qt or sc!=qc or ac!=qc:raise ValueError('full original QA grid mismatch')
        qv&=sv&(sat==0)&av&aerosol_valid(a)
    arrays=[];meta=[]
    # Use actual raw-file origin; cached1200m transforms are not original extents.
    raw=[]
    for band in row['bands']:
        x,v,t,crs=raster(band['source']);g=band['native_gsd_m']
        if t.a!=g or t.e!=-g or t.b or t.d:raise ValueError('not a native north-up source')
        raw.append((band,x,v,t,crs))
    west=min(x[3].c for x in raw);north=max(x[3].f for x in raw)
    side=math.ceil(max(max(t.c-west+x.shape[1]*b['native_gsd_m'],north-t.f+x.shape[0]*b['native_gsd_m']) for b,x,_,t,_ in raw)/60)*60
    for band,x,v,t,crs in raw:
        g=band['native_gsd_m'];v&=x!=0;q=np.zeros(x.shape,np.uint8)
        reproject(qv.astype(np.uint8),q,src_transform=qt,src_crs=qc,dst_transform=t,dst_crs=crs,resampling=Resampling.nearest)
        v&=q.astype(bool);value=np.where(v,x.astype(np.float32)*band['scale']+band['offset'],0)
        count=side//g
        if x.shape[0]>count or x.shape[1]>count:raise ValueError('source exceeds padding support')
        padded=np.zeros((count,count),np.float32);valid=np.zeros((count,count),bool)
        padded[:x.shape[0],:x.shape[1]]=value;valid[:x.shape[0],:x.shape[1]]=v
        arrays.append((torch.from_numpy(padded),torch.from_numpy(valid)))
        meta.append(dict(**{k:v for k,v in band.items() if k not in ['transform','crs','phase_xy_m','valid_fraction']},transform=list(t)[:6],crs=str(crs),phase_xy_m=[t.c+g/2-west,north-t.f+g/2],source_shape=list(x.shape),padded_shape=[count,count],valid_fraction=float(v.mean()),input_sha256=hashlib.sha256(padded.tobytes()).hexdigest(),valid_sha256=hashlib.sha256(valid.tobytes()).hexdigest()))
    bounds=[]
    for band,x,_,t,crs in raw:
        bounds.append(transform_bounds(crs,'EPSG:4326',t.c,t.f-x.shape[0]*band['native_gsd_m'],t.c+x.shape[1]*band['native_gsd_m'],t.f))
    record=dict(row,full_bands=meta,full_footprint_m=side,source_extent='complete local original files; no crop; invalid right/bottom padding only',full_geographic_bbox=[min(b[0] for b in bounds),min(b[1] for b in bounds),max(b[2] for b in bounds),max(b[3] for b in bounds)])
    return arrays,record

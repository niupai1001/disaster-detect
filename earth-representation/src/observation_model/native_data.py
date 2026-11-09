"""Read true native L2 windows; interpolated legacy arrays are not accepted here."""
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from pyproj import Transformer,CRS

S2=[("B02",492.4,10),("B03",559.8,10),("B04",664.6,10),("B08",832.8,10),("B05",704.1,20),("B06",740.5,20),("B07",782.8,20),("B8A",864.7,20),("B11",1613.7,20),("B12",2202.4,20)]
LANDSAT=[("coastal",442.96,30),("blue",481.59,30),("green",561.25,30),("red",654.59,30),("nir08",864.67,30),("swir16",1608.86,30),("swir22",2200.73,30)]
LS_NAME={name:f"SR_B{i+1}" for i,(name,_,_) in enumerate(LANDSAT)}
BAD=(0,1,3,8,9,10,11,255)


def aerosol_valid(qa):
    qa=qa.astype(np.uint16)
    high=((qa>>6)&3)==3
    return ((qa&1)==0)&~(high&(((qa&2)==0)|((qa&32)!=0)))


def native_window(path,bbox,gsd,opener=None,bbox_crs=None):
    left,bottom,right,top=bbox;side=round((right-left)/gsd)
    with rasterio.open(path,opener=opener) as src:
        if src.res!=(gsd,gsd) or src.transform.b or src.transform.d:
            raise ValueError(f"not an original north-up{gsd}m raster: {path}")
        output_crs=str(src.crs);dx=dy=0.
        if bbox_crs and CRS.from_user_input(bbox_crs)!=CRS.from_user_input(src.crs):
            a=CRS.from_user_input(bbox_crs).to_epsg();b=CRS.from_user_input(src.crs).to_epsg()
            if not (a and b and a//100 in [326,327] and b//100 in [326,327] and a%100==b%100):
                raise ValueError("different native UTM zones need an explicit geometry operator; no hidden image warp")
            transform=Transformer.from_crs(bbox_crs,src.crs,always_xy=True)
            x,y=transform.transform(left,top);dx=x-left;dy=y-top
            left+=dx;right+=dx;top+=dy;bottom+=dy;output_crs=str(bbox_crs)
        # Keep pixel centers inside the requested support, retaining the source phase.
        col=int(math.ceil((left-src.transform.c)/gsd-.5-1e-7))
        row=int(math.ceil((src.transform.f-top)/gsd-.5-1e-7))
        if min(col,row)<0 or col+side>src.width or row+side>src.height:raise ValueError("native source does not cover1200m window")
        window=Window(col,row,side,side);value=src.read(1,window=window)
        transform=Affine.translation(-dx,-dy)*src.window_transform(window);crs=output_crs
        valid=np.isfinite(value)
        if src.nodata is not None:valid &= value!=src.nodata
        return value,valid,transform,crs


def load_product_record(paths):
    record=next(iter(paths.values()))
    path=Path(record["file"])
    return path.parent/"product.json"


def optical_from_four_view(base,site,sensor,bbox):
    paths=site["files"]["s2_l2a" if sensor=="S2" else "landsat_l2"]
    bands=S2 if sensor=="S2" else LANDSAT
    product=json.loads((base/load_product_record(paths)).read_text())
    if sensor=="S2":
        baseline=float(product["properties"]["s2:processing_baseline"])
        if baseline>=4:
            raise ValueError("PB>=04 native source requires exact BOA metadata audit; not blindly applying cached offsets")
        scale,offset=.0001,0.
        qkey="SCL";qa_gsd=20
    else:scale,offset=.0000275,-.2;qkey="qa_pixel";qa_gsd=30
    qa,qa_valid,qt,qc=native_window(base/paths[qkey]["file"],bbox,qa_gsd,bbox_crs=site.get("crs"))
    qa_valid &= ~np.isin(qa,BAD) if sensor=="S2" else (qa.astype(np.uint16)&63)==0
    if sensor!="S2":
        sat,_,st,sc=native_window(base/paths["qa_radsat"]["file"],bbox,30,bbox_crs=site.get("crs"))
        if st!=qt or sc!=qc:raise ValueError("QA grid mismatch")
        qa_valid &= sat==0
        aerosol_path=base/paths["qa_aerosol"]["file"] if "qa_aerosol" in paths else (base/paths["qa_pixel"]["file"]).parent/"qa_aerosol.tif"
        aerosol,av,at,ac=native_window(aerosol_path,bbox,30,bbox_crs=site.get("crs"))
        if at!=qt or ac!=qc:raise ValueError("aerosol QA grid mismatch")
        qa_valid &= av&aerosol_valid(aerosol)
    arrays={};meta=[]
    for key,wavelength,gsd in bands:
        path=base/paths[key]["file"]
        dn,valid,transform,crs=native_window(path,bbox,gsd,bbox_crs=site.get("crs"))
        valid &= dn!=0
        q=np.zeros_like(valid,dtype=np.uint8)
        reproject(qa_valid.astype(np.uint8),q,src_transform=qt,src_crs=qc,dst_transform=transform,dst_crs=crs,resampling=Resampling.nearest)
        valid &= q.astype(bool)
        name=("S2:"+key) if sensor=="S2" else ("L8:"+LS_NAME[key])
        arrays[name]=np.where(valid,dn.astype(np.float32)*scale+offset,0)
        arrays[name+"_valid"]=valid
        phase=(transform.c+gsd/2-bbox[0],bbox[3]-(transform.f-gsd/2))
        meta.append(dict(name=name,native_gsd_m=gsd,wavelength_nm=wavelength,bandwidth_nm=None,transform=list(transform)[:6],crs=crs,phase_xy_m=phase,source=str(path),scale=scale,offset=offset,valid_fraction=float(valid.mean())))
    date=product["properties"]["datetime"]
    return arrays,dict(sensor=sensor,product_id=product["id"],date=date,processing_domain="L2_SR",radiometry="official prePB04 quantification10000" if sensor=="S2" else "USGS_C2_L2 scale0.0000275 offset-0.2",bbox=bbox,crs=meta[0]["crs"],bands=meta,source_site=site["site_id"],footprint_m=1200)


def read_cached(path,device="cpu"):
    import torch
    from .model import BandRaster
    path=Path(path);meta=json.loads(path.with_suffix(".json").read_text())
    with np.load(path) as z:
        bands=[BandRaster(m["name"],torch.from_numpy(z[m["name"]].copy())[None].to(device),torch.from_numpy(z[m["name"]+"_valid"].copy())[None].to(device),m["wavelength_nm"],m["bandwidth_nm"],m["native_gsd_m"],tuple(m["phase_xy_m"])) for m in meta["bands"]]
    return bands,meta

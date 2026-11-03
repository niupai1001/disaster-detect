"""Versioned optical data expansion, preserving real band grids and role isolation."""
from collections import defaultdict
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import io
from urllib.parse import urlsplit

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds, reproject

from .metered_http import DownloadBudget, HTTPRangeFile
from .native_data import aerosol_valid, BAD

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'observation_model/outputs/encoder_v2/data_expansion'
PRE=ROOT/'observation_model/outputs/encoder_v2/pre_G1_data_contract'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(t) for t in Path(p).read_text(encoding='utf-8-sig').splitlines() if t.strip()]
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def append(p,d):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')


class Network:
    def __init__(self):
        parent=read(ROOT/'observation_model/outputs/phase0/download_budget.json')['received_bytes']
        srf=read(ROOT/'data/physics/srf/acquisition_manifest.json')['received_bytes']
        self.cap=10*1024**3-parent-srf
        self.budget=ParallelBudget(OUT/'network.json',self.cap);self.tokens={};self.token_lock=threading.Lock();self.range_lock=threading.Lock();self.ranges=OrderedDict()
    def sign(self,href):
        if '.blob.core.windows.net' not in href:return href
        p=urlsplit(href);account=p.hostname.split('.')[0];container=p.path.split('/')[1];k=(account,container)
        with self.token_lock:
            if k not in self.tokens:
                self.tokens[k]=self.budget.json_request('GET',f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}')['token']
        return href+'?'+self.tokens[k]
    def crop(self,href,crs,transform,shape,path,qa=True):
        path=Path(path)
        if path.exists():return
        address=self.sign(href)
        def opener(name,mode='r'):
            if str(name)!='asset.tif':raise FileNotFoundError(name)
            with self.range_lock:
                shared=self.ranges.get(href)
                if shared is None:
                    file=self.budget.open(address);file.block_size=65536
                    self.ranges[href]=(file.size,file.blocks)
                    while len(self.ranges)>32:self.ranges.popitem(last=False)
                    return file
                self.ranges.move_to_end(href)
                file=HTTPRangeFile.__new__(HTTPRangeFile);io.RawIOBase.__init__(file)
                file.url=address;file.budget=self.budget;file.pos=0;file.size=shared[0];file.blocks=shared[1];file.block_size=65536
                return file
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with rasterio.open('asset.tif',opener=opener) as src:
                if qa:
                    with WarpedVRT(src,crs=crs,transform=Affine(*transform),width=shape[1],height=shape[0],resampling=Resampling.nearest) as vrt:
                        data=vrt.read(1,masked=True);t=Affine(*transform);c=crs
                else:
                    bounds=rasterio.windows.bounds(Window(0,0,shape[1],shape[0]),Affine(*transform))
                    bounds=transform_bounds(crs,src.crs,*bounds)
                    w=from_bounds(*bounds,transform=src.transform)
                    x0,y0=int(np.floor(w.col_off)),int(np.floor(w.row_off));x1,y1=int(np.ceil(w.col_off+w.width)),int(np.ceil(w.row_off+w.height))
                    if x0<0 or y0<0 or x1>src.width or y1>src.height:raise ValueError('native source does not cover requested support')
                    window=Window(x0,y0,x1-x0,y1-y0);data=src.read(1,window=window,masked=True);t=src.window_transform(window);c=src.crs
                profile=dict(driver='GTiff',width=data.shape[1],height=data.shape[0],count=1,dtype=str(data.dtype),crs=c,transform=t,compress='deflate')
                path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.tif')
                with rasterio.open(temp,'w',**profile) as dst:
                    dst.write(data.filled(0),1);dst.write_mask((~np.ma.getmaskarray(data)).astype('uint8')*255)
                temp.replace(path)
    def json(self,method,url,**kwargs):return self.budget.json_request(method,url,**kwargs)


class ParallelBudget(DownloadBudget):
    """Reserve response bytes before concurrent I/O, then charge only bytes read."""
    def __init__(self,path,cap):
        super().__init__(path,cap);self.reserved=0;self.sessions=threading.local()
    def reserve(self,size):
        with self.lock:
            if size>self.cap-self.received-self.reserved:raise RuntimeError('DOWNLOAD_BUDGET_EXHAUSTED')
            self.reserved+=size
    def charge(self,reservation,received):
        with self.lock:self.reserved-=reservation;self.received+=received;self.save()
    def session_for_thread(self):
        import requests
        if not hasattr(self.sessions,'value'):
            self.sessions.value=requests.Session();self.sessions.value.headers['Accept-Encoding']='identity'
        return self.sessions.value
    def bytes_request(self,method,url,**kwargs):
        limit=min(8*1024**2,self.cap-self.received-self.reserved)
        if limit<=0:raise RuntimeError('DOWNLOAD_BUDGET_EXHAUSTED')
        self.reserve(limit);data=b''
        try:
            with self.session_for_thread().request(method,url,stream=True,timeout=(20,45),**kwargs) as response:
                response.raise_for_status();data=response.raw.read(limit);return data
        finally:self.charge(limit,len(data))
    def get(self,url,start,end):
        n=end-start+1;self.reserve(n);data=b''
        try:
            with self.session_for_thread().get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,45)) as response:
                response.raise_for_status()
                if response.status_code!=206:raise ValueError('bounded range request not honored')
                data=response.raw.read(n);return data,int(response.headers['Content-Range'].split('/')[-1])
        finally:self.charge(n,len(data))


def write_observation(identifier,arrays,metadata):
    cache=OUT/'cache'/(identifier+'.npz');side=cache.with_suffix('.json')
    if side.exists():return read(side)
    if any(float(valid.mean())<.5 for _,valid in arrays.values()):raise ValueError('per-band valid coverage below50percent')
    values={}
    for b,(x,q) in arrays.items():
        if not np.isfinite(x[q]).all():raise ValueError('nonfinite valid reflectance')
        values[b]=np.where(q,x,0).astype('float32');values[b+'__valid']=q.astype(bool)
    cache.parent.mkdir(parents=True,exist_ok=True);temp=cache.with_suffix('.tmp.npz');np.savez_compressed(temp,**values);temp.replace(cache)
    row=dict(metadata,id=identifier,cache=str(cache),cache_sha256=digest(cache),band_count=len(arrays),
             valid_fraction={b:float(q.mean()) for b,(_,q) in arrays.items()},
             training_authorized=False,srf_scientific_ingestion='pending_G1',units='surface_reflectance',created_utc=datetime.now(timezone.utc).isoformat())
    save(side,row);return row


def existing_native(r):
    arrays={};bands=[];sensor=r['sensor']
    qa=[]
    for p in r['quality_paths']:
        with rasterio.open(p) as f:qa.append((Path(p).name,f.read(1),f.dataset_mask()>0,f.transform,f.crs))
    for descriptor in r['native_band_contract']:
        p=descriptor['source'];cal=descriptor['calibration'];b=descriptor['band_key']
        with rasterio.open(p) as f:
            dn=f.read(1);q=f.dataset_mask()>0;t=f.transform;c=f.crs
        q &= np.isfinite(dn)&(dn!=0)
        for name,a,v,qt,qc in qa:
            if sensor=='S2':v=v&~np.isin(a,BAD)
            elif 'qa_pixel' in name:v=v&((a.astype('uint16')&63)==0)
            elif 'qa_radsat' in name:v=v&(a==0)
            elif 'qa_aerosol' in name:v=v&aerosol_valid(a)
            mask=np.zeros(dn.shape,'uint8');reproject(v.astype('uint8'),mask,src_transform=qt,src_crs=qc,dst_transform=t,dst_crs=c,resampling=Resampling.nearest);q &= mask.astype(bool)
        x=dn.astype('float32')*cal['scale']+cal['offset'];arrays[b]=(x,q)
        bands.append(dict(band_key=b,crs=str(c),transform=list(t)[:6],shape=list(x.shape),native_pixel_support_m=[abs(t.a),abs(t.e)],source=p,scale=cal['scale'],offset=cal['offset']))
    return write_observation(r['id'],arrays,dict(sensor=sensor,instrument=r['instrument'],product=r['product'],date=r['date'],role=r['effective_partition'],isolation_group=r['isolation_group'],bbox=r['bbox'],bands=bands,processing_domain='original_L2_SR',provenance='existing_native_TIF_with_QA',acquisition_key=r['acquisition_key']))


def cached_native(r,original,network=None):
    sensor=r['sensor'];source=Path(original['cache_uri'])
    if digest(source)!=original['sha256']:raise ValueError('existing cache SHA differs from frozen source manifest')
    with np.load(source,allow_pickle=False) as z:
        x=z['x'].copy();valid=z['pixel_valid'].copy();meta=z['metadata'].copy()
    if x.dtype!='float32':raise ValueError('expected calibrated float32 cache')
    t=original['transform'];crs=original['crs'];aliases={'blue':'B02','green':'B03','red':'B04','nir':'B08'}
    selected=[i for i,b in enumerate(original['band_ids']) if b in aliases] if sensor=='S2' else list(range(len(x)))
    expected=10 if sensor=='S2' else 30
    if abs(t[0]-expected)>1e-6 or abs(t[4]+expected)>1e-6 or t[1] or t[3]:raise ValueError('cache does not retain a supported native grid')
    if not all(abs(float(meta[i,2])-expected)<1e-5 for i in selected):raise ValueError('selected band original support differs from cache grid')
    if sensor!='S2':
        source_t=original['properties']['proj:transform']
        if any(abs((t[k]-source_t[k])/30-round((t[k]-source_t[k])/30))>1e-6 for k in [2,5]):raise ValueError('Landsat cache phase is not native')
        if network is None:raise ValueError('aerosol QA required')
        first=original['physical_transforms'][0]['href'];href=re.sub(r'_SR_B\d+\.TIF$', '_SR_QA_AEROSOL.TIF',first)
        path=OUT/'raw_QA'/(r['id']+'.tif');network.crop(href,crs,t,x.shape[-2:],path)
        with rasterio.open(path) as f:valid &= (aerosol_valid(f.read(1))&(f.dataset_mask()>0))[None]
    names_s2={'blue':'B02','green':'B03','red':'B04','nir':'B08'}
    names_ls=dict(zip(['blue','green','red','nir08','swir16','swir22'],['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']))
    arrays={};bands=[]
    for i in selected:
        old=original['band_ids'][i];b=(names_s2 if sensor=='S2' else names_ls)[old];arrays[b]=(x[i],valid[i])
        physical=original['physical_transforms'][i]
        dn=np.rint((x[i][valid[i]]-physical['offset'])/physical['scale'])
        error=float(np.max(np.abs(dn*physical['scale']+physical['offset']-x[i][valid[i]]))) if len(dn) else float('inf')
        if error>max(abs(physical['scale'])*.51,1e-6):raise ValueError('calibration round trip inconsistent with original integer grid')
        bands.append(dict(band_key=b,crs=crs,transform=t,shape=list(x[i].shape),native_pixel_support_m=[expected,expected],source_cache_channel=i,source_asset=physical['href'],cached_already_scaled=True,scale=physical['scale'],offset=physical['offset'],arithmetic_roundtrip_max_abs=error,wavelength_nm=float(meta[i,0])*1000))
    return write_observation(r['id'],arrays,dict(sensor=sensor,instrument=r['instrument'],product=r['product'],date=r['date'],role=r['effective_partition'],bbox=r['bbox'],isolation_group=r['isolation_group'],bands=bands,processing_domain='original_L2_SR',provenance='native_lattice_cache_channels_only',source_cache=str(source),source_cache_sha256=original['sha256'],excluded_derived_channels=[b for i,b in enumerate(original['band_ids']) if i not in selected],QA_provenance='frozen_source_SCL_mask' if sensor=='S2' else 'frozen_QA_PIXEL_RADSAT_plus_new_native_aerosol',acquisition_key=r['acquisition_key']))

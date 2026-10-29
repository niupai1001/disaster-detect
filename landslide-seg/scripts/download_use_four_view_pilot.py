"""Bounded four-sensor pilot. New data only; never resumes the old downloader."""
import argparse
import hashlib
import io
import json
import os
import struct
import threading
import time
import zipfile
import sys
import subprocess
from datetime import datetime, timezone
import secrets
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/use_data_pairing_plan/four_view_10gb_pilot'
DATA = ROOT / 'data/use_multisensor/four_view_pilot_v1'
ZIP_URL = 'https://zenodo.org/records/15382551/files/hr_dataset_raw.zip?download=1'
ZIP_SIZE = 11387407107
PC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
EARTH = 'https://earth-search.aws.element84.com/v1'
_CACHE_LOCKS=[threading.Lock() for _ in range(128)]


def cache_lock(path):
    return _CACHE_LOCKS[int(hashlib.sha256(str(path).encode()).hexdigest()[:8],16)%len(_CACHE_LOCKS)]


def replace_retry(source,destination):
    for attempt in range(5):
        try:
            os.replace(source,destination)
            return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.05*2**attempt)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    replace_retry(tmp, path)


class BudgetExceeded(RuntimeError):
    pass


class Network:
    def __init__(self, limit=9_700_000_000):
        self.path = OUT / 'network.json'
        self.lock = threading.Lock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'received_bytes': 0, 'requests': 0,
            'unmetered_bootstrap_attempts': 'Initial manual range probes interrupted; bytes unknown, reserve 300MB outside the 9.7GB metered cap.',
        }
        self.limit = min(limit,self.state.get('authorized_meter_cap_bytes',limit))
        self.reserved=0
        self.last_flush=0
        self.session=requests.Session()
        adapter=requests.adapters.HTTPAdapter(pool_connections=20,pool_maxsize=20,max_retries=0)
        self.session.mount('https://',adapter)

    def flush(self,force=False):
        with self.lock:
            if force or time.monotonic()-self.last_flush>=1:
                self.state['inflight_reserved_bytes']=self.reserved
                save(self.path,self.state)
                self.last_flush=time.monotonic()

    def add(self, n):
        with self.lock:
            self.reserved-=n
            self.state['received_bytes'] += n
            if self.state['received_bytes'] > self.limit:
                save(self.path, self.state)
                raise BudgetExceeded('10GB pilot network budget reached')

    def get(self, url, *, start=None, end=None, max_bytes=50_000_000, params=None, method='GET'):
        if self.state.get('cooldown_until',0)>time.time():
            raise RuntimeError('Source rate-limit cooldown active')
        if start is not None:
            assert end >= start
            max_bytes = end - start + 1
        # Reserve response bounds across all simultaneous positions, including one overflow chunk.
        allowance=max_bytes+65536
        with self.lock:
            if self.state['received_bytes']+self.reserved+allowance>self.limit:
                raise BudgetExceeded('Insufficient remaining network budget for bounded request')
            self.reserved+=allowance
            self.state['requests']+=1
        headers = {'Range': f'bytes={start}-{end}'} if start is not None else {}
        if start is not None and 'zenodo.org' in url:
            url += ('&' if '?' in url else '?') + f'pilot_range={start}_{end}'
        response = None
        consumed=0
        try:
            response = self.session.request(method, url, params=params, headers=headers, stream=True, timeout=(10, 25))
            if response.status_code == 429:
                retry=response.headers.get('Retry-After','600')
                delay=float(retry) if retry.isdigit() else 600
                self.state['cooldown_until']=time.time()+max(60,delay)
                raise RuntimeError('HTTP429 Retry-After=' + response.headers.get('Retry-After', 'unknown'))
            response.raise_for_status()
            length=response.headers.get('Content-Length')
            if length and int(length)>max_bytes:
                raise RuntimeError('Declared response exceeds byte bound')
            if start is not None:
                expected = f'bytes {start}-{end}/'
                if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                    raise RuntimeError('Server did not honor the exact bounded Range')
            data = bytearray()
            checkpoint = time.monotonic()
            for chunk in response.iter_content(65536):
                self.add(len(chunk))
                consumed+=len(chunk)
                self.flush()
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise RuntimeError('Response exceeds declared byte bound')
                if time.monotonic() - checkpoint > 10:
                    self.flush()
                    print('network_MB', round(self.state['received_bytes'] / 1e6, 1), flush=True)
                    checkpoint = time.monotonic()
            if start is not None and len(data) != max_bytes:
                raise RuntimeError('Truncated range response')
            return bytes(data), dict(response.headers)
        finally:
            if response is not None:
                response.close()
            with self.lock:
                self.reserved-=allowance-consumed
            self.flush()

    def json(self, url, params=None):
        return json.loads(self.get(url, params=params, max_bytes=30_000_000)[0])


class RemoteZip(io.RawIOBase):
    def __init__(self, net, shared_cache=None):
        self.net, self.pos = net, 0
        if shared_cache is not None:
            self.cache=list(shared_cache)
            return
        self.cache = []
        tail_path = OUT / 'zip_tail.bin'
        if tail_path.exists():
            tail = tail_path.read_bytes()
        else:
            tail = net.get(ZIP_URL, start=ZIP_SIZE - 65536, end=ZIP_SIZE - 1)[0]
            tail_path.write_bytes(tail)
        self.cache.append((ZIP_SIZE - len(tail), tail))
        z64 = struct.unpack_from('<4sQ2H2L4Q', tail, tail.rfind(b'PK\x06\x06'))
        size, offset = z64[8:10]
        central_path = OUT / 'zip_central_correct.bin'
        if central_path.exists() and central_path.stat().st_size == size:
            central = central_path.read_bytes()
        else:
            central = net.get(ZIP_URL, start=offset, end=offset + size - 1)[0]
            central_path.write_bytes(central)
        self.cache.append((offset, central))

    def seekable(self):
        return True

    def readable(self):
        return True

    def prefetch(self, member):
        start=member.header_offset
        end=min(ZIP_SIZE-1,start+member.compress_size+2047)
        if any(offset<=start and start+member.compress_size+512<=offset+len(data) for offset,data in self.cache):
            return
        body=self.net.get(ZIP_URL,start=start,end=end)[0]
        self.cache=self.cache[:2]+[(start,body)]

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else ZIP_SIZE + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = ZIP_SIZE - self.pos
        size = min(size, ZIP_SIZE - self.pos)
        if size <= 0:
            return b''
        start = self.pos
        for offset, data in self.cache:
            if offset <= start and start + size <= offset + len(data):
                self.pos += size
                return data[start-offset:start-offset+size]
        # ZipFile reads local headers and compressed member bodies separately.
        data = self.net.get(ZIP_URL, start=start, end=start + size - 1)[0]
        self.pos += size
        return data


def bootstrap(net):
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ['metadata.csv', 'LICENSE.txt', 'stratified_train_val_test_split.csv']:
        path = DATA / 'sources' / name
        if not path.exists():
            body = net.get('https://zenodo.org/records/15382551/files/' + name + '?download=1', max_bytes=20_000_000)[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        print('source', name, path.stat().st_size, flush=True)
    archive = zipfile.ZipFile(RemoteZip(net))
    files = [{'name':x.filename, 'offset':x.header_offset, 'compressed':x.compress_size,
              'size':x.file_size, 'crc':x.CRC} for x in archive.infolist() if x.filename.lower().endswith(('.tif','.tiff','.jp2'))]
    save(OUT/'hr_members.json', files)
    print('HR raster members', len(files), files[:5], flush=True)
    import pandas as pd
    frame = pd.read_csv(DATA/'sources/metadata.csv')
    print('metadata rows', len(frame), 'columns', list(frame.columns), flush=True)
    print(frame.head(1).to_json(orient='records')[:1800], flush=True)
    return archive, frame


def search(net, collection, bounds, year, earth=False):
    endpoint = EARTH if earth else PC
    params = {'collections': collection, 'bbox': ','.join(map(str,bounds)),
              'datetime': f'{max(2016,year-1)}-01-01T00:00:00Z/{min(2026,year+1)}-12-31T23:59:59Z', 'limit': 60}
    result = net.json(endpoint + '/search', params)
    return result.get('features', [])


def cached_json(net, name, url, params=None):
    path=OUT/'catalog'/(hashlib.sha256((url+json.dumps(params,sort_keys=True)).encode()).hexdigest()+'.json')
    with cache_lock(path):
        if path.exists():
            return json.loads(path.read_text())
        value=net.json(url,params);save(path,value)
        return value


def gcs_list(net, bucket, prefix, delimiter=None):
    params={'prefix':prefix,'maxResults':1000}
    if delimiter:
        params['delimiter']=delimiter
    result=cached_json(net,'gcs',f'https://storage.googleapis.com/storage/v1/b/{bucket}/o',params)
    if result.get('nextPageToken'):
        raise ValueError('GCS prefix must be narrowed before pagination')
    return result


def candidates(net, collection, bounds, year):
    params={'collections':collection,'bbox':','.join(map(str,bounds)),
            'datetime':f'{max(2016,year-1)}-01-01T00:00:00Z/{min(2021,year+1)}-12-31T23:59:59Z','limit':100}
    if collection in ['landsat-c2-l2','sentinel-2-l2a']:
        params['sortby']='+eo:cloud_cover'
    items=cached_json(net,'search',PC+'/search',params).get('features',[])
    if collection=='landsat-c2-l2':
        items=[x for x in items if x['properties'].get('platform')=='landsat-8' and 'coastal' in x['assets'] and 'lwir11' in x['assets']]
    if collection=='sentinel-1-rtc':
        items=[x for x in items if {'vv','vh'}<=set(x['assets'])]
    return sorted(items,key=lambda x:(x['properties'].get('eo:cloud_cover',0),abs(int(x['properties']['datetime'][:4])-year),x['id']))


def landsat_l1_assets(net, item):
    props=item['properties'];date=item['id'].split('_')[3]
    prefix=f"LC08/01/{int(props['landsat:wrs_path']):03}/{int(props['landsat:wrs_row']):03}/"
    dirs=gcs_list(net,'gcp-public-data-landsat',prefix,'/').get('prefixes',[])
    dirs=[x for x in dirs if '_'+date+'_' in x and '_L1TP_' in x and '_T1/' in x]
    if not dirs:
        raise ValueError('No public Landsat8 L1 C1 counterpart for this acquisition')
    folder=sorted(dirs)[-1]
    files=gcs_list(net,'gcp-public-data-landsat',folder).get('items',[])
    assets={}
    for f in files:
        name=f['name'];base=Path(name).name
        match=re.search(r'_B(\d+)\.TIF$',base)
        key='B'+match.group(1) if match else 'BQA' if base.endswith('_BQA.TIF') else None
        if key:
            assets[key]={'href':'https://storage.googleapis.com/gcp-public-data-landsat/'+name,
                         'raster:bands':[{'nodata':0,'spatial_resolution':15 if key=='B8' else 30}]}
        elif base.endswith('_MTL.txt'):
            assets['mtl']={'href':'https://storage.googleapis.com/gcp-public-data-landsat/'+name}
    if not {f'B{i}' for i in range(1,12)}<=set(assets):
        raise ValueError('Incomplete Landsat L1 bands')
    return {'id':folder.rstrip('/').split('/')[-1], 'collection':'Landsat8 Collection1 Level1',
            'properties':{'datetime':item['properties']['datetime']},'assets':assets}


def s2_l1_assets(net,item):
    props=item['properties'];tile=props['s2:mgrs_tile']
    product=props.get('s2:product_uri',item['id'])
    sat,_,acquisition=product.split('_')[:3]
    prefix=f'tiles/{tile[:2]}/{tile[2]}/{tile[3:]}/{sat}_MSIL1C_{acquisition}'
    dirs=gcs_list(net,'gcp-public-data-sentinel-2',prefix,'/').get('prefixes',[])
    if not dirs:
        raise ValueError('No public S2 L1C counterpart for this acquisition')
    folder=sorted(dirs)[-1]
    files=gcs_list(net,'gcp-public-data-sentinel-2',folder).get('items',[])
    assets={}
    for f in files:
        name=f['name'];match=re.search(r'_(B(?:0[1-9]|1[012]|8A))\.jp2$',name)
        if match:
            key=match.group(1)
            gsd=10 if key in ['B02','B03','B04','B08'] else 60 if key in ['B01','B09','B10'] else 20
            physical={'nodata':0,'scale':0.0001,'spatial_resolution':gsd}
            baseline=re.search(r'_N(\d{4})_',folder)
            if baseline and int(baseline.group(1))<400:
                physical['offset']=0
            assets[key]={'href':'https://storage.googleapis.com/gcp-public-data-sentinel-2/'+name,
                         'raster:bands':[physical], 'conversion_status':'Read source XML for any additive offset not explicitly recorded.'}
        elif Path(name).name in ['MTD_MSIL1C.xml','MTD_TL.xml']:
            assets[Path(name).name]={'href':'https://storage.googleapis.com/gcp-public-data-sentinel-2/'+name}
    if len([x for x in assets if re.fullmatch(r'B(?:0[1-9]|1[012]|8A)',x)])!=13:
        raise ValueError('Incomplete S2 L1C bands')
    return {'id':folder.split('/')[-2], 'properties':{'datetime':props['datetime']},'assets':assets}


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def hr_files(archive, root):
    import rasterio
    if root in ['.','..'] or '/' in root or '\\' in root or ':' in root:
        raise ValueError('Unsafe archive root name')
    members=[x for x in archive.infolist() if x.filename.startswith('hr_dataset_raw/'+root+'/') and x.filename.endswith('.JP2')]
    root_members=[x for x in archive.infolist() if x.filename.startswith('hr_dataset_raw/'+root+'/')]
    if not (DATA/'sources/hr'/root/'metadata.json').exists():
        start=min(x.header_offset for x in root_members)
        last=max(root_members,key=lambda x:x.header_offset)
        end=min(ZIP_SIZE-1,last.header_offset+last.compress_size+1023)
        path=DATA/'sources/hr'/root/'original_product_range.bin'
        if path.exists() and path.stat().st_size==end-start+1:
            body=path.read_bytes()
        else:
            body=archive.fp.net.get(ZIP_URL,start=start,end=end)[0]
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
        archive.fp.cache=archive.fp.cache[:2]+[(start,body)]
        save(path.with_suffix('.json'),{'source':ZIP_URL,'range':[start,end],'sha256':hash_file(path)})
    result={}
    for kind, marker in [('ms','_MS_'),('pan','_P_')]:
        matched=[x for x in members if marker in Path(x.filename).name]
        if len(matched)!=1:
            raise ValueError('Expected exactly one original HR '+kind)
        member=matched[0]
        path=DATA/'sources/hr'/root/(kind+'.jp2')
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            archive.fp.prefetch(member)
            path.write_bytes(archive.read(member))
        with rasterio.open(path) as src:
            if src.crs is None or src.count!=(4 if kind=='ms' else 1):
                raise ValueError('Invalid HR georeferencing/channel count')
        result[kind]=(path,member.filename)
    # DIMAP calibration and band-order metadata are kept alongside original imagery.
    meta_path=DATA/'sources/hr'/root/'metadata.json'
    if not meta_path.exists():
        records=[]
        for member in archive.infolist():
            if member.filename.startswith('hr_dataset_raw/'+root+'/') and member.filename.lower().endswith('.xml'):
                archive.fp.prefetch(member)
                body=archive.read(member)
                path=DATA/'sources/hr'/root/'xml'/Path(member.filename).name
                # Same basenames in MS/PAN directories must not overwrite one another.
                path=path.with_name(hashlib.sha256(member.filename.encode()).hexdigest()[:10]+'_'+path.name)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
                records.append({'archive_member':member.filename,'file':str(path.relative_to(DATA)),'sha256':hash_file(path)})
        save(meta_path,records)
    return result


def crop_raster(url,path,crs,bounds,proxy=None,require_valid=True):
    import rasterio
    from rasterio.windows import from_bounds,Window
    from rasterio.warp import transform_bounds
    if path.exists():
        try:
            with rasterio.open(path) as src:
                valid=float((src.dataset_mask()>0).mean())
                src.read(1)
                return {'file':str(path.relative_to(DATA)),'sha256':hash_file(path),'shape':[src.count,src.height,src.width],
                        'crs':str(src.crs),'transform':list(src.transform)[:6],'valid_ratio':valid,'dtype':src.dtypes[0]}
        except rasterio.errors.RasterioIOError:
            if not path.resolve().is_relative_to((DATA/'staging').resolve()):raise
            replace_retry(path,path.with_name(path.name+f'.corrupt.{time.time_ns()}'))
    if proxy is not None:
        # GDAL open can hold the GIL; a separate reader keeps the local HTTP server responsive.
        return proxy.reader.submit(crop_raster,proxy.register(url),path,crs,bounds,None,require_valid).result(timeout=240)
    address=str(url)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.TIF,.tiff,.jp2,.JP2',
                      GDAL_HTTP_TIMEOUT='35',GDAL_HTTP_MAX_RETRY='0',GDAL_TIFF_INTERNAL_MASK=True,VSI_CACHE=False):
        with rasterio.open(address) as src:
            if src.crs is None:
                raise ValueError('Missing CRS')
            projected=transform_bounds(crs,src.crs,*bounds,densify_pts=21)
            win=from_bounds(*projected,transform=src.transform)
            left,top=int(np.floor(win.col_off)),int(np.floor(win.row_off))
            right,bottom=int(np.ceil(win.col_off+win.width)),int(np.ceil(win.row_off+win.height))
            if left<0 or top<0 or right>src.width or bottom>src.height:
                raise ValueError('Common footprint is not fully covered')
            win=Window(left,top,right-left,bottom-top)
            arr=src.read(window=win,masked=True)
            valid=(~np.ma.getmaskarray(arr)&np.isfinite(arr.data)).all(0)
            if require_valid and valid.mean()<.8:
                raise ValueError('Less than 80% finite source coverage')
            profile=src.profile.copy()
            profile.update(driver='GTiff',width=arr.shape[2],height=arr.shape[1],transform=src.window_transform(win),
                           compress='deflate',tiled=False)
            for key in ['blockxsize','blockysize','photometric']:
                profile.pop(key,None)
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.stem+'.writing.tif')
            with rasterio.open(temporary,'w',**profile) as dst:
                dst.write(arr.filled(src.nodata if src.nodata is not None else 0))
                dst.write_mask(valid.astype('uint8')*255)
                dst.scales=src.scales;dst.offsets=src.offsets
                dst.update_tags(**src.tags())
                for i,desc in enumerate(src.descriptions,1):
                    if desc:dst.set_band_description(i,desc)
            replace_retry(temporary,path)
    return crop_raster(url,path,crs,bounds,proxy,require_valid)


def export_product(net,proxy,item,folder,crs,bounds,keys=None):
    assets=item['assets']
    if keys is None:
        keys=[k for k,v in assets.items() if v.get('type','').startswith('image/tiff')]
    records={}
    for key in keys:
        asset=assets[key];url=asset['href']
        print('band',folder.name,key,flush=True)
        record=crop_raster(url,folder/(key+'.tif'),crs,bounds,proxy,
                           require_valid=key not in ['qa','cloud_qa','qa_pixel','qa_radsat','BQA'])
        record['asset_metadata']=asset
        records[key]=record
    for key,asset in assets.items():
        if key.lower().endswith(('.xml','.txt','.json')) or key=='mtl':
            path=folder/(re.sub(r'[^a-zA-Z0-9_.-]','_',key)+'.metadata')
            if not path.exists():
                body=net.get(proxy.sign(asset['href']),max_bytes=5_000_000)[0]
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
    save(folder/'product.json',{'id':item['id'],'properties':item.get('properties',{}),'assets':records})
    return records


def aligned(path,crs,bounds,size=144,band=1,nearest=False):
    import rasterio
    from rasterio.warp import reproject,Resampling
    from rasterio.transform import from_bounds
    dst=np.full((size,size),np.nan,'float32')
    with rasterio.open(path) as src:
        arr=src.read(band).astype('float32')
        arr[src.read_masks(band)==0]=np.nan
        reproject(arr,dst,src_transform=src.transform,src_crs=src.crs,src_nodata=np.nan,
                  dst_transform=from_bounds(*bounds,size,size),dst_crs=crs,dst_nodata=np.nan,
                  resampling=Resampling.nearest if nearest else Resampling.average)
    return dst


def quality(path,crs,bounds,kind):
    qa=aligned(path,crs,bounds,nearest=True)
    if kind=='scl':
        ok=np.isfinite(qa)&~np.isin(qa,[0,1,3,8,9,10])
    elif kind=='landsat':
        qa=np.nan_to_num(qa).astype('uint16')
        ok=(qa&63)==0
    else:
        ok=np.isfinite(qa)&(qa==0)
    if float(ok.mean())<.8:
        raise ValueError('Pixel quality below 80% '+kind)
    return float(ok.mean())


def process_site(net,proxy,archive,row,roots):
    import ast
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform
    import prepare_use_external as old
    tile=str(row['tile']);root=tile
    if root not in roots:
        root=tile.rsplit('-',2)[0]
    if root not in roots:
        raise ValueError('HR archive location not found')
    geographic=ast.literal_eval(row['bounds'])
    if old.overlaps(box(*geographic),process_site.forbidden):
        raise ValueError('Held-out event exclusion')
    site_id=re.sub(r'[^a-zA-Z0-9_-]','_',tile)
    destination=DATA/'sites'/site_id
    done=destination/'site.json'
    if done.exists():
        return json.loads(done.read_text(encoding='utf-8'))
    site=DATA/'staging'/site_id
    if destination.exists() and not site.exists():
        assert destination.resolve().is_relative_to(DATA.resolve()) and site.resolve().is_relative_to(DATA.resolve())
        site.parent.mkdir(parents=True,exist_ok=True)
        os.replace(destination,site)
    year=max(2017,min(2020,int(str(row['highres_date'])[:4])))
    # Check all three satellite catalogs before transferring HR image members.
    s2s=candidates(net,'sentinel-2-l2a',geographic,year)
    lss=candidates(net,'landsat-c2-l2',geographic,year)
    s1s=candidates(net,'sentinel-1-rtc',geographic,year)
    if not s2s or not lss or not s1s:
        raise ValueError(f'Missing catalog candidate S2={len(s2s)} Landsat8={len(lss)} S1RTC={len(s1s)}')
    with cache_lock(DATA/'sources/hr'/root):
        hr=hr_files(archive,root)
    with rasterio.open(hr['ms'][0]) as src:
        crs=str(src.crs)
        transform_to=Transformer.from_crs(4326,src.crs,always_xy=True).transform
        cx,cy=transform_to((geographic[0]+geographic[2])/2,(geographic[1]+geographic[3])/2)
        cx,cy=round(cx/60)*60,round(cy/60)*60
        bounds=[cx-720,cy-720,cx+720,cy+720]
    geo=transform(Transformer.from_crs(crs,4326,always_xy=True).transform,box(*bounds))
    if not box(*geographic).buffer(.00005).covers(geo):
        raise ValueError('Snapped ROI exceeds the anchor metadata footprint')
    records={}
    records['hr']={kind:crop_raster(path,site/'hr'/(kind+'.tif'),crs,bounds) for kind,(path,_) in hr.items()}
    print('site',site_id,'HR ready; finding clear optical observations',flush=True)
    last=None
    for item in s2s[:5]:
        folder=site/'s2_l2a'/item['id']
        try:
            crop_raster(item['assets']['SCL']['href'],folder/'SCL.tif',crs,bounds,proxy,False)
            q=quality(folder/'SCL.tif',crs,bounds,'scl')
            s2l1=s2_l1_assets(net,item)
            s2=item;break
        except BudgetExceeded:
            raise
        except Exception as exc:
            last=exc
    else:
        raise ValueError('No usable S2 full L1/L2 pair: '+str(last).split('?')[0][:150])
    for item in lss[:8]:
        folder=site/'landsat_l2'/item['id']
        try:
            crop_raster(item['assets']['qa_pixel']['href'],folder/'qa_pixel.tif',crs,bounds,proxy,False)
            lq=quality(folder/'qa_pixel.tif',crs,bounds,'landsat')
            crop_raster(item['assets']['qa_radsat']['href'],folder/'qa_radsat.tif',crs,bounds,proxy,False)
            quality(folder/'qa_radsat.tif',crs,bounds,'saturation')
            ls1=landsat_l1_assets(net,item)
            ls2=item;break
        except BudgetExceeded:
            raise
        except Exception as exc:
            last=exc
    else:
        raise ValueError('No usable Landsat full L1/L2 pair: '+str(last).split('?')[0][:150])
    s2_keys=[k for k in s2['assets'] if re.fullmatch(r'B(?:0[1-9]|1[12]|8A)',k)]+['SCL','AOT','WVP']
    if len([k for k in s2_keys if k.startswith('B')])!=12:
        raise ValueError('S2 L2A expected 12 bands')
    jobs={
        's1':(s1s[0],site/'s1',['vv','vh']),
        's2_l1c':(s2l1,site/'s2_l1c',[f'B{i:02}' for i in range(1,13)]+['B8A']),
        's2_l2a':(s2,site/'s2_l2a'/s2['id'],s2_keys),
        'landsat_l1':(ls1,site/'landsat_l1_c1',[f'B{i}' for i in range(1,12)]+['BQA']),
        'landsat_l2':(ls2,site/'landsat_l2'/ls2['id'],None),
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={key:pool.submit(export_product,net,proxy,item,folder,crs,bounds,keys)
                 for key,(item,folder,keys) in jobs.items()}
        for key,future in futures.items():
            records[key]=future.result()
    dates={'hr':str(row['highres_date']),'s2':s2['properties']['datetime'],
           'landsat':ls2['properties']['datetime'],'s1':s1s[0]['properties']['datetime']}
    # Every native-grid crop must contain the entire requested polygon.
    for group,files in records.items():
        for key,rec in files.items():
            arr=aligned(DATA/rec['file'],crs,bounds,size=48)
            if np.isfinite(arr).mean()<.8:
                raise ValueError('Common-footprint valid fraction failed '+group+'/'+key)
    result={'site_id':site_id,'anchor_tile':tile,'anchor_root':root,'status':'complete_four_sensor_georeferenced',
            'role':'UNASSIGNED_NOT_USED_FOR_TRAINING','crs':crs,'bbox_projected':bounds,'footprint_wgs84':geo.__geo_interface__,
            'datetime':dates,'quality':{'s2_clear_fraction':q,'landsat_clear_fraction':lq},'files':records,
            'limitations':['No same-day constraint; source dates differ.',
                          'Landsat L1 is public Collection1; same-acquisition L2 is Collection2, stored separately.',
                          'Native source georeferencing and common polygon verified; subpixel SAR-optical registration not certified.',
                          'No new statistics fitted or training started.'],
            'license':'WorldStrat HR CC-BY-NC4.0; source LICENSE retained.'}
    make_preview(site,result)
    def committed_paths(value):
        if isinstance(value,dict):return {k:committed_paths(v) for k,v in value.items()}
        if isinstance(value,list):return [committed_paths(v) for v in value]
        if isinstance(value,str):
            for sep in ['/',chr(92)]:
                prefix='staging'+sep+site_id+sep
                if value.startswith(prefix):return 'sites/'+site_id+'/'+value[len(prefix):].replace(chr(92),'/')
        return value
    result=committed_paths(result)
    for product in site.rglob('product.json'):
        save(product,committed_paths(json.loads(product.read_text(encoding='utf-8'))))
    save(site/'site.json',result)
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert site.resolve().is_relative_to(DATA.resolve()) and destination.resolve().is_relative_to(DATA.resolve())
    replace_retry(site,destination)
    return result


def make_preview(site,record):
    from PIL import Image,ImageDraw
    crs,bounds=record['crs'],record['bbox_projected']
    groups=record['files']
    def stretch(arr):
        out=[]
        for a in arr:
            lo,hi=np.nanpercentile(a,[2,98]);a=np.nan_to_num((a-lo)/max(hi-lo,1e-8))
            out.append(np.clip(a*255,0,255).astype('uint8'))
        return np.stack(out,-1)
    hr=DATA/groups['hr']['ms']['file']
    # SPOT MS channel identity is checked against the retained DIMAP metadata.
    rgb=stretch([aligned(hr,crs,bounds,288,b) for b in [1,2,3]])
    s2=stretch([aligned(DATA/groups['s2_l2a'][k]['file'],crs,bounds,288) for k in ['B04','B03','B02']])
    ls=stretch([aligned(DATA/groups['landsat_l2'][k]['file'],crs,bounds,288) for k in ['red','green','blue']])
    sar=aligned(DATA/groups['s1']['vv']['file'],crs,bounds,288)
    sar=stretch([np.log10(np.maximum(sar,1e-8))]*3)
    pan=stretch([aligned(DATA/groups['hr']['pan']['file'],crs,bounds,288)]*3)
    canvas=Image.new('RGB',(288*5,330),'white');draw=ImageDraw.Draw(canvas)
    for i,(arr,label) in enumerate(zip([rgb,pan,s2,ls,sar],['HR MS channels 1/2/3','HR PAN','S2 RGB','Landsat RGB','S1 VV log display'])):
        canvas.paste(Image.fromarray(arr),(i*288,40));draw.text((i*288+4,7),label,fill='black')
    canvas.save(site/'paired_preview.jpg',quality=92)


def prepare_parallel_resume():
    marker=OUT/'parallel_resume_prepared.json'
    if marker.exists():return
    trusted={};moved=[]
    staging=DATA/'staging'
    for product in staging.glob('**/product.json'):
        for record in json.loads(product.read_text(encoding='utf-8'))['assets'].values():
            trusted[product.parent/Path(record['file']).name]=record['sha256']
    for path in staging.glob('**/*.tif'):
        if path in trusted and hash_file(path)==trusted[path]:continue
        for item in [path,path.with_name(path.name+'.msk'),path.with_name(path.name+'.aux.xml')]:
            if not item.exists():continue
            target=item.with_name(item.name+'.unverified_serial')
            assert item.resolve().is_relative_to(staging.resolve()) and target.resolve().is_relative_to(staging.resolve())
            replace_retry(item,target);moved.append(str(item.relative_to(DATA)))
    save(marker,{'utc':datetime.now(timezone.utc).isoformat(),'quarantined_uncommitted_files':moved,
                 'reason':'The old serial process was stopped; only group-manifest hashes prove those non-atomic crops completed.'})


def run(net,archive,frame,max_sites,first_site=None,site_workers=3,reader_workers=6):
    import prepare_use_external as old
    proxy=AssetProxy(net,reader_workers=reader_workers)
    prepare_parallel_resume()
    process_site.forbidden=old.exclusion()
    roots={x.filename.split('/')[1] for x in archive.infolist() if x.filename.startswith('hr_dataset_raw/') and x.filename.endswith('.JP2')}
    rows=frame.drop_duplicates('tile').to_dict('records')
    rows.sort(key=lambda x:hashlib.sha256(('1234:'+x['tile']).encode()).hexdigest())
    if first_site:
        rows=[x for x in rows if x['tile']==first_site]
    completed=[]
    failure_path=OUT/'failures.json'
    failures=json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else []
    for path in (DATA/'sites').glob('*/site.json') if (DATA/'sites').exists() else []:
        completed.append(json.loads(path.read_text(encoding='utf-8')))
    known={x['site_id'] for x in completed}
    # Old interrupted attempts are never presented in the accepted sites directory.
    for partial in (DATA/'sites').iterdir() if (DATA/'sites').exists() else []:
        if partial.is_dir() and not (partial/'site.json').exists():
            target=DATA/'staging'/partial.name
            assert partial.resolve().is_relative_to(DATA.resolve()) and target.resolve().is_relative_to(DATA.resolve())
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True);replace_retry(partial,target)
    attempted={re.sub(r'[^a-zA-Z0-9_-]','_',f['tile']) for f in failures}
    local=threading.local()
    archives=[]
    archives_lock=threading.Lock()
    def process(row):
        if not hasattr(local,'archive'):
            local.archive=zipfile.ZipFile(RemoteZip(net,shared_cache=archive.fp.cache[:2]))
            with archives_lock:archives.append(local.archive)
        return process_site(net,proxy,local.archive,row,roots)
    pending={};cursor=iter(rows);exhausted=False;stopping=False
    started=time.monotonic();last_status=0
    status={'status':'running','site_workers':site_workers,'reader_workers':reader_workers}
    def progress(force=False):
        nonlocal last_status
        failed_keys={re.sub(r'[^a-zA-Z0-9_-]','_',f['tile']) for f in failures}
        status.update(completed=len(completed),failed=len(failed_keys-known),failed_attempts=len(failures),
                      network_bytes=net.state['received_bytes'],
                      network_cap_bytes=net.limit,
                      active_sites=[re.sub(r'[^a-zA-Z0-9_-]','_',r['tile']) for r,_ in pending.values()],
                      elapsed_seconds=round(time.monotonic()-started,1))
        if force or time.monotonic()-last_status>=5:
            save(OUT/'status.json',status);last_status=time.monotonic()
    progress(True)
    try:
        with ThreadPoolExecutor(max_workers=site_workers) as pool:
            while pending or not exhausted:
                cooling=net.state.get('cooldown_until',0)>time.time()
                if net.state['received_bytes']>=9_500_000_000:
                    stopping=True;status['status']='budget_reached'
                if cooling:status['status']='rate_limit_cooldown'
                elif not stopping:status['status']='running'
                while not stopping and not cooling and not exhausted and len(pending)<site_workers and len(completed)+len(pending)<max_sites:
                    try:row=next(cursor)
                    except StopIteration:exhausted=True;break
                    key=re.sub(r'[^a-zA-Z0-9_-]','_',row['tile'])
                    if key in known or key in attempted:continue
                    attempted.add(key)
                    pending[pool.submit(process,row)]=(row,time.monotonic())
                    print('START',key,'active',len(pending),'completed',len(completed),flush=True)
                if not pending:
                    if stopping or exhausted or len(completed)>=max_sites:break
                    progress();time.sleep(1);continue
                finished,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in finished:
                    row,begin=pending.pop(future)
                    key=re.sub(r'[^a-zA-Z0-9_-]','_',row['tile'])
                    try:
                        result=future.result()
                        completed.append(result);known.add(result['site_id'])
                        print('ACCEPTED',key,len(completed),flush=True)
                    except BudgetExceeded:
                        stopping=True;status['status']='budget_reached'
                    except Exception as exc:
                        msg=type(exc).__name__+': '+str(exc).split('?')[0][:400]
                        failures.append({'tile':row['tile'],'error':msg})
                        print('REJECTED',key,msg,flush=True)
                    event={'utc':datetime.now(timezone.utc).isoformat(),'tile':row['tile'],
                           'accepted':key in known,'seconds':round(time.monotonic()-begin,2),
                           'completed':len(completed),'network_bytes_total':net.state['received_bytes'],
                           'site_workers':site_workers,'reader_workers':reader_workers}
                    with (OUT/'parallel_events.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(event,ensure_ascii=False)+'\n')
                if finished:
                    save(OUT/'failures.json',failures)
                    save(DATA/'manifest.json',sorted(completed,key=lambda x:x['site_id']))
                progress(bool(finished))
    finally:
        # Drain readers before the final byte counter and checksum audit are frozen.
        proxy.reader.shutdown(wait=True,cancel_futures=True)
        proxy.server.shutdown()
        for item in archives:item.close()
        net.flush(force=True)
        if status['status']=='running':status['status']='bounded_run_finished'
        progress(True)
        save(OUT/'failures.json',failures);save(DATA/'manifest.json',sorted(completed,key=lambda x:x['site_id']))


def audit():
    import rasterio
    from rasterio.warp import transform_bounds
    accepted=[];errors=[];total_bytes=0;files=0
    for meta in sorted((DATA/'sites').glob('*/site.json')):
        site=json.loads(meta.read_text(encoding='utf-8'))
        try:
            groups=site['files']
            assert len(groups['s2_l1c'])==13
            assert sum(k.startswith('B') for k in groups['s2_l2a'])==12
            assert {f'B{i}' for i in range(1,12)}<=set(groups['landsat_l1'])
            assert {'coastal','blue','green','red','nir08','swir16','swir22','lwir11'}<=set(groups['landsat_l2'])
            assert {'vv','vh'}<=set(groups['s1']) and set(groups['hr'])=={'ms','pan'}
            for group,records in groups.items():
                for key,record in records.items():
                    path=DATA/record['file']
                    assert hash_file(path)==record['sha256'],str(path)
                    with rasterio.open(path) as src:
                        assert src.crs is not None and src.count==record['shape'][0]
                        assert (src.dataset_mask()>0).mean()>=.8,group+'/'+key
                        b=transform_bounds(src.crs,site['crs'],*src.bounds,densify_pts=21)
                        roi=site['bbox_projected']
                        assert b[0]<=roi[0]+.1 and b[1]<=roi[1]+.1 and b[2]>=roi[2]-.1 and b[3]>=roi[3]-.1,group+'/'+key
                    total_bytes+=path.stat().st_size;files+=1
            accepted.append(site['site_id'])
        except Exception as exc:
            errors.append({'site':site['site_id'],'error':str(exc)[:250]})
    value={'ok':not errors and bool(accepted),'accepted_sites':len(accepted),'verified_raster_files':files,
           'verified_raster_bytes':total_bytes,'errors':errors,'sites':accepted,
           'network':json.loads((OUT/'network.json').read_text()),
           'scope':'Four complete sensor groups, all planned primary bands, readable native-grid crops, hashes and common footprint. No same-day or subpixel registration certification. No training.',
           'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    save(OUT/'acceptance.json',value)
    print('AUDIT',json.dumps({k:value[k] for k in ['ok','accepted_sites','verified_raster_files','verified_raster_bytes','errors']}),flush=True)
    return value


def probe(net, frame):
    import ast
    row = frame.iloc[0]
    bounds = ast.literal_eval(row['bounds'])
    year = int(row['highres_date'][:4])
    for key, collection, earth in [('ls1','landsat-c2-l1',False),('ls2','landsat-c2-l2',False),
                                    ('s1','sentinel-1-rtc',False),('s2','sentinel-2-l2a',False),
                                    ('s2l1','sentinel-2-l1c',True)]:
        items = search(net, collection, bounds, year, earth)
        save(OUT / (key+'_candidates.json'), items)
        print('catalog',key,'count',len(items),flush=True)
        if items:
            print(json.dumps({k:{'href':v.get('href'),'roles':v.get('roles'),
                                 'bands':v.get('raster:bands'),'eo':v.get('eo:bands')}
                              for k,v in items[0]['assets'].items()})[:7000],flush=True)


def probe_hr(archive):
    import rasterio
    members=[x for x in archive.infolist() if x.filename.startswith('hr_dataset_raw/Amnesty POI-10/') and x.filename.endswith('.JP2')]
    for member in members:
        path=OUT/('probe_ms.jp2' if '_MS_' in member.filename else 'probe_pan.jp2')
        if not path.exists():
            path.write_bytes(archive.read(member))
        with rasterio.open(path) as src:
            print('HR_PROBE',src.count,src.shape,str(src.crs),list(src.transform)[:6],src.bounds,src.dtypes,flush=True)


class AssetProxy:
    """Account for every GDAL range body; cache source blocks, never signed URLs."""
    def __init__(self, net, reader_workers=6):
        self.net, self.assets, self.tokens = net, {}, {}
        self.token_lock=threading.Lock()
        self.reader=ProcessPoolExecutor(max_workers=reader_workers)
        self.nonce = secrets.token_hex(12)
        outer = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self, *_):
                pass
            def do_HEAD(self):
                self.transfer(True)
            def do_GET(self):
                self.transfer(False)
            def transfer(self, head):
                url = outer.assets.get(self.path)
                if url is None:
                    self.send_error(404); return
                try:
                    if head:
                        with outer.net.session.head(outer.sign(url),timeout=(10,20)) as r:
                            r.raise_for_status()
                            self.send_response(200)
                            self.send_header('Content-Length', r.headers['Content-Length'])
                            self.send_header('Accept-Ranges','bytes')
                            self.end_headers()
                        return
                    match = re.fullmatch(r'bytes=(\d+)-(\d+)', self.headers.get('Range',''))
                    if not match:
                        raise ValueError('Unbounded GDAL request refused')
                    start,end = map(int,match.groups())
                    key = hashlib.sha256(f'{url}:{start}:{end}'.encode()).hexdigest()
                    path = DATA/'source_blocks'/key[:2]/key
                    with cache_lock(path):
                        if path.exists() and path.stat().st_size==end-start+1:
                            body = path.read_bytes()
                        else:
                            body,_ = outer.net.get(outer.sign(url),start=start,end=end)
                            path.parent.mkdir(parents=True,exist_ok=True)
                            temporary=path.with_name(path.name+'.part')
                            temporary.write_bytes(body)
                            replace_retry(temporary,path)
                    self.send_response(206)
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Content-Range',f'bytes {start}-{end}/*')
                    self.send_header('Accept-Ranges','bytes')
                    self.end_headers();self.wfile.write(body)
                except (BrokenPipeError,ConnectionAbortedError,ConnectionResetError):
                    # GDAL may cancel speculative range reads after it has enough bytes.
                    return
                except Exception as exc:
                    # Never include a SAS query in diagnostics.
                    print('asset_error',type(exc).__name__,str(exc).split('?')[0][:180],flush=True)
                    try:
                        self.send_error(502)
                    except (BrokenPipeError,ConnectionAbortedError,ConnectionResetError):
                        pass
        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()

    def sign(self, url):
        host=urlparse(url).hostname
        if host and host.endswith('.blob.core.windows.net'):
            account=host.split('.')[0];container=urlparse(url).path.split('/')[1]
            key=(account,container)
            with self.token_lock:
                cached=self.tokens.get(key)
                if cached is None or cached[1]<time.time()+120:
                    token=self.net.json(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}',{'_ts':int(time.time())})['token']
                    expiry=parse_qs(token).get('se',[None])[0]
                    if not expiry:raise ValueError('SAS token lacks expiry')
                    expiry=datetime.fromisoformat(expiry.replace('Z','+00:00')).timestamp()
                    if expiry<time.time()+120:raise ValueError('Expired SAS token from provider')
                    self.tokens[key]=(token,expiry)
                token=self.tokens[key][0]
            return url+'?'+token
        return url

    def register(self, url):
        if urlparse(url).scheme != 'https':
            raise ValueError('HTTPS assets only')
        key='/'+self.nonce+'/'+hashlib.sha256(url.encode()).hexdigest()+Path(urlparse(url).path).suffix
        self.assets[key]=url
        return 'http://127.0.0.1:'+str(self.server.server_address[1])+key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--max-sites',type=int,default=10000)
    parser.add_argument('--site-workers',type=int,choices=range(1,5),default=3)
    parser.add_argument('--reader-workers',type=int,choices=range(1,9),default=6)
    parser.add_argument('--first-site')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--register-on-finish',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot=Path(__file__).read_bytes();source_sha=hashlib.sha256(snapshot).hexdigest()
    snap=OUT/'source_snapshots'/(source_sha+'.py')
    snap.parent.mkdir(parents=True,exist_ok=True)
    if not snap.exists():snap.write_bytes(snapshot)
    save(OUT/'last_launch.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source_sha,'argv':sys.argv[1:],'pid':os.getpid()})
    if args.audit:
        audit();return
    net = Network()
    archive,frame=bootstrap(net)
    if args.probe:
        probe_hr(archive)
        probe(net,frame)
    if args.run:
        run(net,archive,frame,args.max_sites,args.first_site,args.site_workers,args.reader_workers)
        runtime=json.loads((OUT/'status.json').read_text())
        runtime['status']='auditing';save(OUT/'status.json',runtime)
        checked=audit()
        runtime.update(status='complete' if checked['ok'] else 'audit_failed',audited_sites=checked['accepted_sites'],
                       verified_raster_files=checked['verified_raster_files'])
        save(OUT/'status.json',runtime)
        if args.register_on_finish and checked['ok']:
            note=f"四源试下载已结束：{checked['accepted_sites']}个完整同区包，{checked['verified_raster_files']}个主清单栅格哈希/覆盖复核通过；实际计量网络{checked['network']['received_bytes']}字节。Landsat L1 C1和L2 C2分开，无同日/亚像素保证，无训练；详见status与acceptance。"
            subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/manage_agent_work.py'),'finish','use-four-view-10gb-pilot',
                            '--owner','session-01a17d3f-5cb9-7c50-8674-7b22c4d7b5bb',
                            '--evidence','outputs/use_data_pairing_plan/four_view_10gb_pilot/acceptance.json',
                            '--evidence','data/use_multisensor/four_view_pilot_v1/manifest.json','--note',note],cwd=ROOT,check=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(OUT/'last_error.json',{'utc':datetime.now(timezone.utc).isoformat(),
             'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
             'error':type(exc).__name__+': '+str(exc).split('?')[0][:500]})
        raise

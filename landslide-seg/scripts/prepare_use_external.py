"""Bounded public optical COG windows with geographic exclusion and provenance."""
import argparse
import hashlib
import json
import sys
import time
import threading
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import numpy as np
import requests
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform
from rasterio.enums import Resampling
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/use/external'
S2=('blue','green','red','rededge1','rededge2','rededge3','nir','nir08','swir16','swir22')
LS=('blue','green','red','nir08','swir16','swir22')
CROP_OFFSETS=sorted([(x*16000,y*16000) for x in range(-2,3) for y in range(-2,3)],
    key=lambda xy:(xy[0]**2+xy[1]**2,hashlib.sha256(f'1234:{xy}'.encode()).hexdigest()))
SESSION=requests.Session()
TOKEN_LOCK=threading.Lock()
TOKEN_COOLDOWN=0
TOKEN_CACHE={}


class RemoteThrottled(RuntimeError):
    def __init__(self,message,retry_after=600):
        super().__init__(message);self.retry_after=retry_after


def safe_error(exc):
    return re.sub(r'(https://[^\s\x27\x22?]+)\?[^\s\x27\x22]+',r'\1?[redacted]',str(exc))


def get(url,**kwargs):
    r=SESSION.get(url,timeout=45,**kwargs)
    if r.status_code==429:
        value=r.headers.get('Retry-After','600')
        try:delay=max(60,float(value))
        except ValueError:delay=600
        raise RemoteThrottled('429; stop acquisition, retry after '+value+' seconds',delay)
    r.raise_for_status();return r.json()


def azure_token(account,container,epoch=None):
    key=(account,container);now=time.time();cached=TOKEN_CACHE.get(key)
    if cached and cached[1]>now+120:return cached[0]
    url=f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}'
    for attempt in range(2):
        response=get(url,headers={'Cache-Control':'no-cache'},**({'params':{'_ts':int(now)}} if attempt else {}))
        token=response['token'];expiry=response.get('msft:expiry') or parse_qs(token).get('se',[None])[0]
        if not expiry:raise ValueError('Token response has no verifiable expiry')
        expires=datetime.fromisoformat(expiry.replace('Z','+00:00')).timestamp()
        if expires>time.time()+120:
            TOKEN_CACHE[key]=(token,expires);return token
    raise ValueError('Token provider returned expired or imminently expiring access')


def signed(url):
    global TOKEN_COOLDOWN
    parts=urlparse(url)
    if parts.scheme!='https' or not parts.hostname or not parts.hostname.endswith(('.amazonaws.com','.blob.core.windows.net')):
        raise ValueError('Only declared public optical HTTPS asset hosts are allowed')
    if '.blob.core.windows.net' not in parts.netloc:return url
    account=parts.netloc.split('.')[0];container=parts.path.split('/')[1]
    with TOKEN_LOCK:
        if TOKEN_COOLDOWN>time.time():raise RemoteThrottled('Token provider cooldown active',TOKEN_COOLDOWN-time.time())
        try:token=azure_token(account,container,int(time.time()//600))
        except RemoteThrottled as exc:
            TOKEN_COOLDOWN=time.time()+exc.retry_after;raise
    return url+'?'+token


def exclusion(events=None):
    # All local event footprints, including withheld members; no labels are read.
    result=[]
    paths=[ROOT/'manifests/v4r1/pool_100.jsonl']
    protocol=json.loads((ROOT/'manifests/v4r1/protocol.json').read_text())
    rows=[json.loads(l) for l in paths[0].read_text().splitlines()]
    original=ROOT/'manifests/s2_10band_v4_p20.jsonl'
    if hashlib.sha256(original.read_bytes()).hexdigest()!=protocol['source_manifest_sha256']:
        raise ValueError('Original event footprint manifest hash differs from frozen protocol')
    rows.extend(json.loads(l) for l in original.read_text().splitlines() if l.strip())
    # Find the original manifest registered by the frozen protocol for withheld regions.
    for name in protocol['files']:
        path=Path(name)
        if path.suffix=='.jsonl' and path.exists():
            rows.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    required={m['event_id'] for m in protocol['members'].values()}
    grouped={}
    for row in rows:
        if not row.get('bbox') or not row.get('crs'):continue
        key=(row['event_id'],row['crs']);bounds=row['bbox']
        if key not in grouped:grouped[key]=list(bounds)
        else:
            old=grouped[key];grouped[key]=[min(old[0],bounds[0]),min(old[1],bounds[1]),max(old[2],bounds[2]),max(old[3],bounds[3])]
    by_event={}
    for (event,crs),bounds in grouped.items():
        geo=transform(Transformer.from_crs(crs,4326,always_xy=True).transform,box(*bounds))
        by_event.setdefault(event,[]).append(geo)
    for event,parts in by_event.items():
        if events is not None and event not in events:continue
        # Conservative event envelopes exclude gaps between local project tiles too.
        geo=unary_union(parts).envelope
        if geo.bounds[2]-geo.bounds[0]>180:raise ValueError('Ambiguous date-line event extent')
        # A local azimuthal projection gives a metric buffer, including near the date line.
        lon,lat=geo.centroid.x,geo.centroid.y
        local=f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m'
        buffered=transform(Transformer.from_crs(4326,local,always_xy=True).transform,geo).buffer(10000)
        result.append(transform(Transformer.from_crs(local,4326,always_xy=True).transform,buffered))
    if not required<=set(by_event):raise ValueError('Missing event footprints: '+str(required-set(by_event)))
    return result


def overlaps(geometry,others):
    if others is None:return False
    if isinstance(others,STRtree):return len(others.query(geometry,predicate='intersects'))>0
    return any(geometry.intersects(g) for g in others)


def crop_window(width,height,affine,slot):
    dx,dy=CROP_OFFSETS[slot]
    dc,dr=(~affine)*(affine.c+dx,affine.f+dy)
    return Window(width//2-64+round(dc),height//2-64+round(dr),128,128)


def read_window(item,sensor,forbidden,seen,cell=None,slot=0):
    keys=S2 if sensor=='s2' else LS
    assets=item['assets']
    blue=assets['blue'];props=item['properties']
    shape=blue.get('proj:shape',props.get('proj:shape'))
    affine=blue.get('proj:transform',props.get('proj:transform'))
    crs=blue.get('proj:code',props.get('proj:code',props.get('proj:epsg')))
    if shape and affine and crs:
        declared=rasterio.Affine(*affine[:6]);window=crop_window(shape[1],shape[0],declared,slot)
        geo=transform(Transformer.from_crs(crs,4326,always_xy=True).transform,box(*rasterio.windows.bounds(window,declared)))
        if cell is not None:
            lon,lat=cell
            if not box(lon,lat,lon+5,lat+5).covers(geo.centroid):return None
        if overlaps(geo,forbidden) or overlaps(geo,seen):return None
    with rasterio.open(signed(assets['blue']['href'])) as reference:
        if reference.crs is None:raise ValueError('Missing CRS')
        # Fixed spatial slots; reject invalid/cloudy windows without consulting labels.
        win=crop_window(reference.width,reference.height,reference.transform,slot)
        affine=window_transform(win,reference.transform)
        bounds=rasterio.windows.bounds(win,reference.transform)
        footprint=transform(Transformer.from_crs(reference.crs,4326,always_xy=True).transform,box(*bounds))
        if cell is not None:
            lon,lat=cell
            if not box(lon,lat,lon+5,lat+5).covers(footprint.centroid):return None
        if overlaps(footprint,forbidden) or overlaps(footprint,seen):return None
        qa_key='scl' if sensor=='s2' else 'qa_pixel'
        with rasterio.open(signed(assets[qa_key]['href'])) as src:
            with WarpedVRT(src,crs=reference.crs,transform=affine,width=128,height=128,resampling=Resampling.nearest) as vrt:
                qa=vrt.read(1)
        quality=np.isin(qa,[4,5,6,7]) if sensor=='s2' else (qa & 0b111111)==0
        if quality.mean()<.8:return None
        if sensor=='landsat':
            with rasterio.open(signed(assets['qa_radsat']['href'])) as src:
                with WarpedVRT(src,crs=reference.crs,transform=affine,width=128,height=128,resampling=Resampling.nearest) as vrt:
                    quality &= vrt.read(1)==0
            if quality.mean()<.8:return None
        target_crs=reference.crs
        def read_band(key):
            asset=assets[key];bands=asset.get('raster:bands',[])
            if not bands or 'scale' not in bands[0] or 'offset' not in bands[0]:
                raise ValueError('Missing physical scale/offset')
            props=bands[0];eo=asset.get('eo:bands',[{}])[0]
            if 'center_wavelength' not in eo:raise ValueError('Missing wavelength')
            with rasterio.open(signed(asset['href'])) as src:
                gsd=float(props.get('spatial_resolution',abs(src.transform.a)))
                with WarpedVRT(src,crs=target_crs,transform=affine,width=128,height=128,resampling=Resampling.bilinear) as vrt:
                    raw=vrt.read(1,masked=True)
            ok=~np.ma.getmaskarray(raw)&np.isfinite(raw.data)
            rho=raw.data.astype('float32')*props['scale']+props['offset']
            return (np.where(ok,rho,0),ok,[eo['center_wavelength'],eo.get('full_width_half_max',0),gsd],
                    [True,'full_width_half_max' in eo,True],dict(scale=props['scale'],offset=props['offset'],href=asset['href']))
        with ThreadPoolExecutor(max_workers=4) as executor:
            data,valid,metadata,known,transforms=map(list,zip(*executor.map(read_band,keys)))
        valid=np.asarray(valid)&quality[None]
        if valid.all(0).mean()<.8:return None
        provenance=dict(item_id=item['id'],sensor=sensor,datetime=item['properties']['datetime'],
                        crs=str(reference.crs),transform=list(affine)[:6],bbox=list(footprint.bounds),
                        band_ids=keys,physical_transforms=transforms,properties=item['properties'],crop_slot=slot)
        return np.asarray(data),valid,np.asarray(metadata,'float32'),np.asarray(known),provenance,footprint


def scan_cell(sensor,collection,cell,forbidden,seen,slot=0):
    lon,lat=cell
    endpoint='https://earth-search.aws.element84.com/v1/search' if sensor=='s2' else 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
    directory=OUT/'catalogue';directory.mkdir(parents=True,exist_ok=True)
    cache=directory/f'{sensor}_{lon}_{lat}.json'
    if cache.exists():found=json.loads(cache.read_text())
    else:
        found=get(endpoint,params={'collections':collection,'bbox':f'{lon},{lat},{lon+5},{lat+5}',
            'datetime':'2020-01-01T00:00:00Z/2024-12-31T23:59:59Z','limit':100,
            'query':json.dumps({'eo:cloud_cover':{'lt':20}})})['features']
        temp=cache.with_suffix('.tmp');temp.write_text(json.dumps(found));temp.replace(cache)
    if sensor=='landsat':found=[f for f in found if f['properties'].get('platform') in ('landsat-8','landsat-9')]
    found.sort(key=lambda f:hashlib.sha256(('1234'+f['id']).encode()).hexdigest())
    with rasterio.Env(GDAL_HTTP_TIMEOUT='20',GDAL_HTTP_CONNECTTIMEOUT='10',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        start=(slot*8)%len(found) if found else 0
        for item in found[start:start+8]:
            result=read_window(item,sensor,forbidden,seen,cell,slot)
            if result is not None:return item,result
    return None


def prefetched_groups(workers,sensor,collection,cells,start,stop,forbidden,seen):
    """Overlap remote IO for two groups, preserving the four-cursor commit order."""
    pending={}
    def submit(begin):
        snapshot=STRtree(seen) if seen else None
        for cursor in range(begin,min(begin+4,stop)):
            pending[cursor]=workers.submit(scan_cell,sensor,collection,cells[cursor%len(cells)],
                forbidden,snapshot,cursor//len(cells))
    submit(start);submit(start+4)
    for begin in range(start,stop,4):
        cursors=list(range(begin,min(begin+4,stop)))
        snapshot=STRtree(seen) if seen else None
        yield cursors,[pending.pop(c) for c in cursors],snapshot
        submit(begin+8)


def audit_pilot(full=False,current=False):
    pilot=OUT/('current_manifest.json' if current else 'manifest.json' if full else 'pilot_manifest.json')
    if not pilot.exists() and not full:pilot.write_bytes((OUT/'manifest.json').read_bytes())
    rows=json.loads(pilot.read_text())
    counts={s:sum(r['sensor']==s for r in rows) for s in ('s2','landsat')}
    expected=5000 if full else 32
    if current:
        frozen=json.loads((OUT/'acquisition_paused.json').read_text())
        if counts!=frozen['sensors'] or hashlib.sha256(pilot.read_bytes()).hexdigest()!=frozen['manifest_sha256']:
            raise ValueError('Frozen current dataset changed')
        if any(not 32<=n<=5000 for n in counts.values()) or sum(counts.values())>10000:
            raise ValueError('Current dataset outside accepted bounds')
    elif any(n!=expected for n in counts.values()):
        raise ValueError('External sample count does not match the declared stage')
    geos=[box(*r['bbox']) for r in rows];tree=STRtree(geos)
    forbidden=exclusion();seen=[];ranges={}
    for i,row in enumerate(rows):
        target=OUT/row['file']
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed external sample')
        geo=box(*row['bbox']);lon,lat=row['cell']
        if any(geo.intersects(p) for p in forbidden) or any(int(j)!=i for j in tree.query(geo)):
            raise ValueError('Overlapping or excluded external window')
        if not box(lon,lat,lon+5,lat+5).covers(geo.centroid):raise ValueError('Sample lies outside its registered geographic cell')
        seen.append(geo)
        with np.load(target) as z:
            x,valid,m,k=[z[key] for key in ('x','pixel_valid','metadata','known')]
            c=10 if row['sensor']=='s2' else 6
            if x.shape!=(c,128,128) or valid.shape!=x.shape or valid.dtype!=np.bool_:
                raise ValueError('External shape/quality mismatch')
            if valid.all(0).mean()<.8 or not np.isfinite(x[valid]).all():raise ValueError('Invalid external reflectance')
            if m.shape!=(c,3) or k.shape!=m.shape or not np.isfinite(m).all():raise ValueError('Invalid external metadata')
            if (m[~k]!=0).any() or (m[k]<=0).any() or ((m[:,0]<.1)|(m[:,0]>3)).any():raise ValueError('External metadata units or known mask mismatch')
            if abs(row['transform'][0])!=(10 if row['sensor']=='s2' else 30):raise ValueError('Unexpected output GSD')
        ranges[row['file']]=[float(x[valid].min()),float(x[valid].max())]
    contract=dict(ok=True,locations=len(rows),sensors=counts,
        exclusion='All project event envelopes plus 10km, checked independent of observation date',
        source_manifest_sha256=hashlib.sha256((ROOT/'manifests/s2_10band_v4_p20.jsonl').read_bytes()).hexdigest(),
        manifest=str(pilot),manifest_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest(),reflectance_ranges=ranges)
    previous=json.loads((OUT/'pilot_acceptance.json').read_text()) if (OUT/'pilot_acceptance.json').exists() else {}
    recipe=json.loads((OUT/'acquisition_contract.json').read_text()) if (OUT/'acquisition_contract.json').exists() else {}
    contract['pilot_grid_latitude_edges']=previous.get('pilot_grid_latitude_edges',recipe.get('latitude_edges',[-55,75]))
    contract['sampling_note']='Accepted pilot windows are retained; the expansion uses the full global 5-degree grid.'
    if not full and not current:
        contract['pilot_manifest']='outputs/use/external/pilot_manifest.json'
        contract['pilot_manifest_sha256']=contract['manifest_sha256']
    (OUT/('current_acceptance.json' if current else 'full_acceptance.json' if full else 'pilot_acceptance.json')).write_text(json.dumps(contract,indent=1),encoding='utf-8')


def main():
    if (OUT/'acquisition_paused.json').exists():
        raise RuntimeError('Downloads paused by user; explicit authorization required before removing pause marker')
    parser=argparse.ArgumentParser();parser.add_argument('--per-sensor',type=int,default=32)
    parser.add_argument('--cell-workers',type=int,choices=(4,8),default=8)
    args=parser.parse_args()
    if not 1<=args.per_sensor<=5000:raise ValueError('Bounded to 5000 per sensor')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.per_sensor>32:
        acceptance=json.loads((OUT/'pilot_acceptance.json').read_text())
        if acceptance.get('ok') is not True:raise ValueError('Pilot not accepted')
    forbidden=exclusion();seen=[];rows=[];errors=[];progress={};resume_after=0
    # Same frozen order for pilot and expansion; equal round-robin geographic coverage.
    cells=[(lon,lat) for lat in range(-90,90,5) for lon in range(-180,180,5)]
    cells.sort(key=lambda c:hashlib.sha256(f'1234:{c}'.encode()).hexdigest())
    status_path=OUT/'acquisition_status.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        resume_after=previous.get('resume_after_epoch',0)
        if resume_after>time.time():raise RemoteThrottled('Recorded provider cooldown still active',resume_after-time.time())
        rows=previous['samples'];errors=previous.get('errors',[]);progress=previous.get('progress',{})
        for error in errors:error['error']=safe_error(error['error'])
        if previous.get('grid')!='global5deg_full_v2':progress={'s2':0,'landsat':0}
        for row in rows:
            target=OUT/row['file'];geo=box(*row['bbox'])
            if hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256'] or any(geo.intersects(p) for p in forbidden):
                raise ValueError('Existing external sample failed identity or conservative exclusion audit')
            seen.append(geo)
        for sensor in ('s2','landsat'):
            existing=[r for r in rows if r['sensor']==sensor]
            if len(existing)>args.per_sensor:raise ValueError('Requested sample cap smaller than existing evidence')
            progress.setdefault(sensor,max([r.get('iteration',cells.index(tuple(r['cell'])))+1 for r in existing],default=0))
    last_save=0
    recipe=dict(seed=1234,years=[2020,2024],grid_degrees=5,latitude_edges=[-90,90],
        per_sensor_cap=args.per_sensor,scene_cloud_percent_lt=20,pixel_valid_ratio_min=.8,
        crop_offsets_metres=CROP_OFFSETS,window_shape=[128,128],candidate_limit=100,candidates_per_slot=8,
        cell_workers=args.cell_workers,commit_batch_size=4,band_workers=4,pilot_policy='Retain the accepted 32+32 pilot windows in the expansion',
        position_definition='Non-overlapping georeferenced windows; multiple windows may share a source scene',
        geographic_exclusion='Frozen source-manifest event envelopes plus 10km',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pilot_manifest_sha256=hashlib.sha256((OUT/'pilot_manifest.json').read_bytes()).hexdigest() if (OUT/'pilot_manifest.json').exists() else None)
    recipe_path=OUT/'acquisition_contract.json'
    if not recipe_path.exists():recipe_path.write_text(json.dumps(recipe,indent=1),encoding='utf-8')
    else:
        previous_recipe=json.loads(recipe_path.read_text())
        previous_recipe.setdefault('commit_batch_size',4)
        # Concurrency is execution tuning; membership, ordering and physics stay frozen.
        for key,value in recipe.items():
            if key not in ('code_sha256','per_sensor_cap','pilot_manifest_sha256','cell_workers') and previous_recipe.get(key)!=json.loads(json.dumps(value)):
                raise ValueError('Acquisition recipe changed: '+key)
        if previous_recipe.get('cell_workers')!=args.cell_workers:
            previous_recipe.setdefault('execution_tuning',[]).append(dict(at=time.time(),
                previous_cell_workers=previous_recipe.get('cell_workers'),cell_workers=args.cell_workers,
                commit_batch_size=4,lookahead_groups=2,reason='User requested Landsat download acceleration'))
        previous_recipe['cell_workers']=args.cell_workers
        if previous_recipe['per_sensor_cap']>args.per_sensor:raise ValueError('Do not reduce a started acquisition cap')
        previous_recipe['per_sensor_cap']=args.per_sensor
        old_pilot=previous_recipe.get('pilot_manifest_sha256')
        if old_pilot and recipe['pilot_manifest_sha256']!=old_pilot:raise ValueError('Pilot manifest changed')
        previous_recipe['pilot_manifest_sha256']=recipe['pilot_manifest_sha256']
        hashes=previous_recipe.setdefault('resume_code_sha256',[])
        if recipe['code_sha256'] not in hashes:hashes.append(recipe['code_sha256'])
        recipe_path.write_text(json.dumps(previous_recipe,indent=1),encoding='utf-8')
    def save(status='running'):
        nonlocal last_save
        if status=='running' and time.time()-last_save<30:return
        payload=dict(status=status,samples=rows,errors=errors,progress=progress,updated_at=time.time(),resume_after_epoch=resume_after,
                     grid='global5deg_full_v2')
        tmp=OUT/'acquisition_status.tmp';tmp.write_text(json.dumps(payload,indent=1),encoding='utf-8')
        for attempt in range(50):
            try:tmp.replace(status_path);break
            except PermissionError:
                if attempt==49:raise
                time.sleep(.1)
        last_save=time.time()
    with rasterio.Env(GDAL_HTTP_TIMEOUT='20',GDAL_HTTP_CONNECTTIMEOUT='10',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        for sensor,collection in [('s2','sentinel-2-l2a'),('landsat','landsat-c2-l2')]:
            count=sum(r['sensor']==sensor for r in rows);failures=0
            if count>=args.per_sensor:continue
            with ThreadPoolExecutor(max_workers=args.cell_workers) as workers:
              for cursors,futures,snapshot in prefetched_groups(workers,sensor,collection,cells,
                      progress.get(sensor,0),len(cells)*len(CROP_OFFSETS),forbidden,seen):
                if count>=args.per_sensor:break
                batch_seen=[]
                for cursor,future in zip(cursors,futures):
                  if count>=args.per_sensor:break
                  lon,lat=cells[cursor%len(cells)]
                  try:
                    outcome=future.result()
                    # Earlier commits may invalidate a prefetched candidate; rescan in the same hash order.
                    if outcome is not None and overlaps(outcome[1][-1],snapshot):
                        outcome=scan_cell(sensor,collection,cells[cursor%len(cells)],forbidden,snapshot,cursor//len(cells))
                    if outcome is not None:
                        item,result=outcome
                        x,valid,metadata,known,prov,footprint=result
                        geo=box(*prov['bbox'])
                        if overlaps(geo,forbidden) or overlaps(geo,snapshot) or overlaps(geo,batch_seen):
                            progress[sensor]=cursor+1;save();continue
                        if not re.fullmatch(r'[A-Za-z0-9_-]+',item['id']):raise ValueError('Unsafe external sample identity')
                        name=sensor+'_'+item['id']+f'_r{prov["crop_slot"]}_c{cursor%len(cells)}.npz';target=OUT/name
                        if not target.exists():
                            temporary=target.with_suffix('.tmp.npz')
                            np.savez_compressed(temporary,x=x,pixel_valid=valid,metadata=metadata,known=known)
                            temporary.replace(target)
                        else:
                            with np.load(target) as old:
                                if not all(np.array_equal(old[key],value) for key,value in [('x',x),('pixel_valid',valid),('metadata',metadata),('known',known)]):
                                    raise ValueError('Existing external window differs from current source')
                        row=dict(file=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),cell=[lon,lat],iteration=cursor,**prov)
                        rows.append(row);seen.append(geo);batch_seen.append(geo);count+=1
                        print(sensor,count,flush=True)
                    failures=0
                  except RemoteThrottled as exc:
                    errors.append(dict(sensor=sensor,cell=[lon,lat],error=safe_error(exc),error_type=type(exc).__name__))
                    resume_after=time.time()+exc.retry_after
                    progress[sensor]=cursor;save('rate_limited');raise
                  except Exception as exc:
                    errors.append(dict(sensor=sensor,cell=[lon,lat],error=safe_error(exc),error_type=type(exc).__name__))
                    failures+=1
                    if failures>=3:
                        progress[sensor]=cursor;save('failed');raise RuntimeError('Three consecutive acquisition errors; inspect sanitized evidence before retry') from None
                  progress[sensor]=cursor+1;save()
    manifest_tmp=OUT/'manifest.tmp'
    manifest_tmp.write_text(json.dumps(rows,indent=1),encoding='utf-8');manifest_tmp.replace(OUT/'manifest.json')
    if any(sum(r['sensor']==s for r in rows)!=args.per_sensor for s in ('s2','landsat')):
        raise RuntimeError('Insufficient accepted external samples; do not pretrain')
    save('complete')


if __name__=='__main__':
    if '--audit-pilot' in sys.argv:audit_pilot()
    elif '--audit-current' in sys.argv:audit_pilot(current=True)
    elif '--audit-full' in sys.argv:audit_pilot(full=True)
    else:main()

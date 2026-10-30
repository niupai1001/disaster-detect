"""Metered, resumable completion and authoritative replacement of uncalibrated local data."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import re
import sys
import time
import requests

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds
from shapely.geometry import box, shape
from shapely.ops import transform
from shapely.strtree import STRtree

import download_use_four_view_pilot as remote
from integrate_use_sources import (ROOT,OUT,PILOT,S2_BANDS,S2_WAVES,LS_BANDS,LS_WAVES,
    read,lines,append,identity,publish,materialize,read_grid,save)
from prepare_use_external import exclusion

# Reuse the existing bounded range proxy, with a completely separate ledger and cache.
remote.OUT=OUT/'completion'
remote.DATA=OUT
S2_ASSETS=dict(zip(S2_BANDS,('coastal','blue','green','red','rededge1','rededge2','rededge3','nir','nir08','nir09','swir16','swir22')))


class CompletionNetwork(remote.Network):
    def get(self, *args, **kwargs):
        for attempt in range(3):
            try:
                return super().get(*args, **kwargs)
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt==2:raise
                time.sleep(2**attempt)


def crop_worker(url,path,crs,bounds):
    remote.DATA=OUT
    return remote.crop_raster(url,Path(path),crs,bounds,require_valid=False)


def new_request(identifier, sensor, crs, bounds, date, role, **extra):
    geo=transform(Transformer.from_crs(crs,4326,always_xy=True).transform,box(*bounds))
    return dict(id=identifier,target_sensor=sensor,crs=crs,bounds=list(bounds),bbox=list(geo.bounds),
                date=date,role=role,**extra)


def prepare():
    current=lines(OUT/'local_release/manifest.jsonl')
    requests=[]
    # The 288 package counterparts are already present; only missing/failed optical views need repair.
    by_site=defaultdict(set)
    for row in current:
        if row.get('origin')=='four_view_pilot':by_site[row['site_id']].add(row['sensor'])
    for row in current:
        if row['role']=='GUARD' or row['sensor'] not in ('s2','landsat'):continue
        if row.get('origin')=='four_view_pilot' and {'s2','landsat'}<=by_site[row['site_id']]:continue
        a,b,c,d,e,f=row['transform'];_,h,w=row['shape'] if 'shape' in row else row['cache_audit']['shape']
        bounds=box(c,f+e*h,c+a*w,f).bounds
        target='s2' if row['sensor']=='landsat' else 'landsat'
        requests.append(new_request('counterpart_'+row['id'],target,row['crs'],bounds,row['date'],row['role'],
            anchor_id=row['id'],anchor_sha256=row['sha256'],origin='counterpart',priority=0 if row.get('origin')=='four_view_pilot' else 3 if target=='s2' else 4))
    # E-drive: deduplicate both roots by observation/band key; never infer historical scale from uint16.
    groups=defaultdict(dict)
    pattern=re.compile(r'^T_[A-Z0-9]+_(\w+)_(\d+)_EV(\d{8})_IM(\d{8})_F(\d{2})\.tif$',re.I)
    with (ROOT.parent/'.tmp/eo_E.csv').open(encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            path=Path(row['Path']);match=pattern.match(path.name)
            if match and match.group(5) in ('01','02','03','04','05','06','07','08','11','12'):
                key=match.groups()[:4];code=match.group(5)
                previous=groups[key].get(code)
                if previous is None or ('yaogan' in str(path) and 'yaogan' not in str(previous)):
                    groups[key][code]=path
    candidate=[]
    for key,bands in groups.items():
        if len(bands)!=10:continue
        path=bands['01']
        if not path.exists():continue
        with rasterio.open(path) as src:
            bounds=src.bounds;crs=str(src.crs)
        date=datetime.strptime(key[3],'%Y%m%d').isoformat()
        candidate.append(new_request('E_'+identity(str(key)),'s2',crs,bounds,date,'UNASSIGNED',
            origin='E_authoritative_replacement',local_source=str(path),local_group=key[0]+'_'+key[1],priority=2,
            replacement_reason='Historical radiometry/cloud provenance absent; fetch matching official L2A rather than guess DN scaling'))
    inventory=read(ROOT/'outputs/use_next/local_inventory_audit/audit.json')
    seen_fire=set()
    for header in inventory['firepaper']['headers']:
        path=Path(header['file'])
        if not path.exists():continue
        digest=remote.hash_file(path)
        if digest in seen_fire:continue
        seen_fire.add(digest)
        date_match=re.search(r'(20\d{6})',path.stem)
        if not date_match:continue
        with rasterio.open(path) as src:bounds=src.bounds;crs=str(src.crs)
        candidate.append(new_request('fire_'+digest[:24],'s2',crs,bounds,datetime.strptime(date_match[1],'%Y%m%d').isoformat(),'UNASSIGNED',
            origin='firepaper_authoritative_replacement',local_source=str(path),local_source_sha256=digest,
            local_group=path.stem.split('_Rank')[0].replace('Pre_','').replace('Post_',''),priority=1,
            replacement_reason='Composite/quality provenance unavailable; substitute explicit single acquisition, preserve original'))
    # All dates and duplicate geographic footprints form connected components before role assignment.
    all_geo=[box(*r['bbox']) for r in candidate];tree=STRtree(all_geo);parent=list(range(len(candidate)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    by_group={}
    for i,row in enumerate(candidate):
        group=row['origin']+':'+row['local_group']
        if group in by_group:parent[find(i)]=find(by_group[group])
        else:by_group[group]=i
        for j in tree.query(all_geo[i],predicate='intersects'):parent[find(int(j))]=find(i)
    protected=STRtree(exclusion(events={'hiroshima','thrissur','italy','newzealand'}))
    existing=[box(*r['bbox']) for r in current];old_tree=STRtree(existing)
    component_roles={}
    for i,row in enumerate(candidate):
        component=find(i);roles=component_roles.setdefault(component,set())
        for j in old_tree.query(all_geo[i],predicate='intersects'):roles.add(current[int(j)]['role'])
        if len(protected.query(all_geo[i],predicate='intersects')):roles.add('GUARD')
    for i,row in enumerate(candidate):
        roles=component_roles[find(i)]
        if 'GUARD' in roles or len(roles)>1:role='GUARD'
        elif roles:role=next(iter(roles))
        else:role='PRETRAIN_VAL' if int(identity(candidate[find(i)]['id']),16)%20==0 else 'TRAIN'
        row['role']=role
    held=STRtree([box(*r['bbox']) for r in current+candidate if r['role']!='TRAIN'])
    for row in candidate:
        geo=box(*row['bbox']);lon,lat=geo.centroid.x,geo.centroid.y
        local=f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m'
        buffered=transform(Transformer.from_crs(local,4326,always_xy=True).transform,
            transform(Transformer.from_crs(4326,local,always_xy=True).transform,geo).buffer(1280))
        if row['role']=='TRAIN' and len(held.query(buffered,predicate='intersects')):row['role']='GUARD'
    requests.extend(r for r in candidate if r['role']!='GUARD')
    directory=OUT/'completion';directory.mkdir(exist_ok=True)
    requests.sort(key=lambda r:(r['priority'],r['role']!='TRAIN',r['id']))
    (directory/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests),encoding='utf8')
    (directory/'local_candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in candidate),encoding='utf8')
    save(directory/'queue_summary.json',dict(requests=len(requests),by_origin=dict(Counter(r['origin'] for r in requests)),
        by_role=dict(Counter(r['role'] for r in requests)),local_candidates=dict(Counter(r['origin']+'/'+r['role'] for r in candidate)),
        budget='First metered batch capped at 9.7GB response bodies; no unlimited acquisition',site_workers=3,reader_workers=6))
    print('prepared',len(requests),flush=True)


def catalog(net, task):
    date=datetime.fromisoformat(task['date'].replace('Z','+00:00')).replace(tzinfo=timezone.utc)
    sensor=task['target_sensor'];collection='sentinel-2-l2a' if sensor=='s2' else 'landsat-c2-l2'
    endpoint=remote.EARTH if sensor=='s2' else remote.PC
    # Local replacement must stay near its actual date; counterpart searches may broaden.
    windows=(3,) if 'replacement' in task['origin'] else (30,180,730)
    for days in windows:
        key=identity(task['id']+str(days));cache=OUT/'completion/catalogue'/(key+'.json')
        if cache.exists():result=read(cache)
        else:
            result=net.json(endpoint+'/search',dict(collections=collection,bbox=','.join(map(str,task['bbox'])),
                datetime=(date-timedelta(days=days)).isoformat()+'/'+(date+timedelta(days=days)).isoformat(),
                limit=30,query=json.dumps({'eo:cloud_cover':{'lt':20}})))
            save(cache,result)
        found=result.get('features',[])
        if sensor=='landsat':found=[f for f in found if f['properties'].get('platform') in ('landsat-8','landsat-9')]
        if not found and sensor=='s2':
            pc_cache=cache.with_name(cache.stem+'_pc.json')
            if pc_cache.exists():pc_result=read(pc_cache)
            else:
                pc_result=net.json(remote.PC+'/search',dict(collections='sentinel-2-l2a',
                    bbox=','.join(map(str,task['bbox'])),
                    datetime=(date-timedelta(days=days)).isoformat()+'/'+(date+timedelta(days=days)).isoformat(),
                    limit=30,query=json.dumps({'eo:cloud_cover':{'lt':80}})))
                save(pc_cache,pc_result)
            found=pc_result.get('features',[])
        if found:
            found.sort(key=lambda f:abs((datetime.fromisoformat(f['properties']['datetime'].replace('Z','+00:00'))-date).total_seconds()))
            return found[:4]
    return []


def complete_one(net,proxy,task):
    items=catalog(net,task)
    sensor=task['target_sensor'];last_error='No suitable catalog acquisition'
    for item in items:
        try:
            assets=item['assets'];bands=S2_BANDS if sensor=='s2' else LS_BANDS
            keys=[b if b in assets else S2_ASSETS[b] for b in bands] if sensor=='s2' else list(bands)
            quality_keys=[('SCL' if 'SCL' in assets else 'scl')] if sensor=='s2' else ['qa_pixel','qa_radsat']
            records={}
            for key in quality_keys+keys:
                asset=assets[key];path=OUT/'raw'/task['id']/item['id']/(key+'.tif')
                if path.exists():
                    with rasterio.open(path) as src:src.read(1)
                else:
                    proxy.reader.submit(crop_worker,proxy.register(asset['href']),str(path),task['crs'],task['bounds']).result(timeout=180)
                records[key]=dict(path=str(path),sha256=remote.hash_file(path),asset_metadata=asset)
            affine=from_bounds(*task['bounds'],128,128)
            quality_data=[read_grid(records[k]['path'],task['crs'],affine,128,128,True) for k in quality_keys]
            if sensor=='s2':
                qa=quality_data[0];quality=np.isin(qa.data,(4,5,6,7))&~np.ma.getmaskarray(qa)
            else:
                qa,sat=quality_data;quality=((qa.data.astype('uint16')&63)==0)&(sat.data==0)&~np.ma.getmaskarray(qa)&~np.ma.getmaskarray(sat)
            x=[];valid=[];metadata=[];known=[];calibration={}
            for i,(band,key) in enumerate(zip(bands,keys)):
                asset=assets[key];info=asset.get('raster:bands',[{}])[0]
                if 'scale' not in info or 'offset' not in info:
                    baseline=re.search(r'_N(\d{4})_',asset.get('href',''))
                    if sensor!='s2' or not baseline:raise ValueError('No authoritative scale/offset '+key)
                    info=dict(info,scale=.0001,offset=-.1 if int(baseline.group(1))>=400 else 0.)
                data=read_grid(records[key]['path'],task['crs'],affine,128,128)
                rho=data.data.astype('float32')*info['scale']+info['offset']
                x.append(rho);valid.append(quality&~np.ma.getmaskarray(data)&np.isfinite(rho))
                eo=asset.get('eo:bands',[{}])[0]
                wave=eo.get('center_wavelength',S2_WAVES[band] if sensor=='s2' else LS_WAVES[i])
                gsd=info.get('spatial_resolution',10 if sensor=='s2' and band in ('B02','B03','B04','B08') else 60 if sensor=='s2' and band in ('B01','B09') else 20 if sensor=='s2' else 30)
                metadata.append([wave,eo.get('full_width_half_max',0),gsd]);known.append([True,'full_width_half_max' in eo,True])
                calibration[band]=dict(scale=info['scale'],offset=info['offset'],source=records[key])
            row=dict(sensor=sensor,role=task['role'],origin=task['origin'],date=item['properties']['datetime'],band_ids=list(bands),
                crs=task['crs'],transform=list(affine)[:6],bbox=task['bbox'],units='surface_reflectance',
                site_id=task.get('anchor_id',task['id']),source_item=item['id'],request=task,
                calibration=calibration,quality_sources={k:records[k] for k in quality_keys},
                date_gap_days=abs((datetime.fromisoformat(item['properties']['datetime'].replace('Z','+00:00'))-
                    datetime.fromisoformat(task['date'].replace('Z','+00:00')).replace(tzinfo=timezone.utc)).total_seconds())/86400)
            result=materialize('remote_'+task['id'],x,valid,metadata,known,row)
            save(OUT/'completion/results'/(task['id']+'.json'),result)
            return dict(id=task['id'],ok=True,source_item=item['id'])
        except remote.BudgetExceeded:raise
        except Exception as exc:
            last_error=type(exc).__name__+': '+str(exc).split('?')[0][:220]
    return dict(id=task['id'],ok=False,error=last_error)


def run(limit, max_tasks):
    directory=OUT/'completion';directory.mkdir(exist_ok=True)
    net=CompletionNetwork(limit=limit)
    net.state['authorized_meter_cap_bytes']=limit
    net.state['unmetered_bootstrap_attempts']='None; all catalog/range/SAS response bodies use the bounded ledger; HEAD bodies empty'
    net.flush(True)
    proxy=remote.AssetProxy(net,reader_workers=6)
    for previous in sorted((directory/'results').glob('*.json')):
        admitted=read(previous)
        if 'replacement' not in admitted['origin']:continue
        anchor=admitted['request']
        follow=new_request('counterpart_'+admitted['id'],'landsat',admitted['crs'],anchor['bounds'],admitted['date'],admitted['role'],
            anchor_id=admitted['id'],anchor_sha256=admitted['sha256'],origin='counterpart',priority=4)
        existing={r['id'] for r in lines(directory/'requests.jsonl')}
        if follow['id'] not in existing:append(directory/'requests.jsonl',follow)
    tasks=lines(directory/'requests.jsonl')
    history=lines(directory/'events.jsonl') if (directory/'events.jsonl').exists() else []
    attempts=Counter(r['id'] for r in history)
    attempted={r['id'] for r in history if r['ok'] or attempts[r['id']]>=3}
    tasks=[r for r in tasks if r['id'] not in attempted]
    tasks.sort(key=lambda r:(0 if r['origin']=='counterpart' and r['anchor_id'].startswith('pilot_') else
        1 if r['origin']=='firepaper_authoritative_replacement' else 2 if r['origin']=='E_authoritative_replacement' else
        3 if r['target_sensor']=='s2' else 4,r['role']!='TRAIN',r['id']))
    repairs=[r for r in tasks if r['origin']=='counterpart' and r['anchor_id'].startswith('pilot_')]
    lanes=[iter([r for r in tasks if r['origin']==origin]) for origin in
           ('firepaper_authoritative_replacement','E_authoritative_replacement')]
    lanes.append(iter([r for r in tasks if r['origin']=='counterpart' and not r['anchor_id'].startswith('pilot_')]))
    # Alternate sources so large firepaper regions cannot exhaust the batch before E or old pairs run.
    from itertools import zip_longest
    tasks=repairs+[r for group in zip_longest(*lanes) for r in group if r is not None]
    tasks=tasks[:max_tasks]
    known_tasks={r['id'] for r in lines(directory/'requests.jsonl')}
    results=directory/'results';results.mkdir(exist_ok=True)
    completed=[];start=time.time();state='running'
    def status():
        save(directory/'status.json',dict(state=state,pid=__import__('os').getpid(),elapsed_seconds=time.time()-start,
            completed=len(list(results.glob('*.json'))),attempted=len(attempted)+len(completed),queue_total=len(lines(directory/'requests.jsonl')),
            received_bytes=net.state['received_bytes'],cap_bytes=limit,site_workers=3,reader_workers=6))
    def release():
        rows=lines(OUT/'local_release/manifest.jsonl')+[read(p) for p in sorted(results.glob('*.json'))]
        publish(rows,'active_release')
    release();status()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            active={};cursor=0
            while cursor<len(tasks) or active:
                if net.state.get('cooldown_until',0)>time.time():
                    state='waiting_rate_limit';status()
                    time.sleep(min(30,net.state['cooldown_until']-time.time()))
                    state='running'
                    continue
                while cursor<len(tasks) and len(active)<3 and net.state['received_bytes']<limit-100_000_000:
                    task=tasks[cursor];cursor+=1;active[pool.submit(complete_one,net,proxy,task)]=task['id']
                if not active:state='budget_limit';break
                ready,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                for future in ready:
                    identifier=active.pop(future)
                    try:result=future.result()
                    except remote.BudgetExceeded:result=dict(id=identifier,ok=False,error='budget_limit');cursor=len(tasks);state='budget_limit'
                    except Exception as exc:result=dict(id=identifier,ok=False,error=type(exc).__name__+': '+str(exc).split('?')[0][:180])
                    append(directory/'events.jsonl',result);completed.append(result)
                    if result['ok']:
                        admitted=read(results/(identifier+'.json'))
                        if 'replacement' in admitted['origin']:
                            anchor=admitted['request']
                            follow=new_request('counterpart_'+admitted['id'],'landsat',admitted['crs'],anchor['bounds'],
                                admitted['date'],admitted['role'],anchor_id=admitted['id'],anchor_sha256=admitted['sha256'],
                                origin='counterpart',priority=4)
                            if follow['id'] not in known_tasks:
                                append(directory/'requests.jsonl',follow);tasks.append(follow);known_tasks.add(follow['id'])
                    print(json.dumps(result),flush=True)
                if ready:release()
                status()
            if state=='running':state='batch_complete'
    finally:
        release();net.flush(True);status();proxy.server.shutdown();proxy.reader.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--cap-bytes',type=int,default=9_700_000_000)
    parser.add_argument('--max-tasks',type=int,default=100000)
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.run:run(args.cap_bytes,args.max_tasks)

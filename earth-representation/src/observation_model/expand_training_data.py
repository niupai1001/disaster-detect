"""User-authorized data expansion only; never starts model training or the old queues."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed,wait,FIRST_COMPLETED
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import time
import threading

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box,shape
from shapely.strtree import STRtree

from .expansion_data import ROOT,OUT,PRE,read,lines,save,digest,append,Network,existing_native,cached_native,write_observation
from .audit_data_contract import identity
from .native_data import aerosol_valid

PC='https://planetarycomputer.microsoft.com/api/stac/v1'


def utc_date(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def publish(status):
    members=[read(p) for p in sorted((OUT/'cache').glob('*.json'))]
    path=OUT/'members.jsonl';temp=path.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in members),encoding='utf-8');temp.replace(path)
    save(OUT/'status.json',dict(status=status,optimizer_updates=0,model_training_started=False,member_count=len(members),counts=dict(Counter(r['role']+'/'+r['sensor'] for r in members)),by_provenance=dict(Counter(r['provenance'] for r in members)),manifest_sha256=digest(path),updated_utc=datetime.now(timezone.utc).isoformat()))
    print('PUBLISH',status,len(members),flush=True)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'authorization.json').exists():return
    source=PRE/'v4/canonical_observations.jsonl';raw=PRE/'shared_release.jsonl'
    save(OUT/'authorization.json',dict(authorized_work='Expand local data, use HLS or recover Landsat windows to align existing inventory; no model training.',
        imagery_download_authorized=True,scope='Missing QA/native Landsat windows for existing local locations only; HLS local resources first',
        permitted_optimizer_updates=0,G1_model_or_SRF_ingestion_started=False,old_training_revival=False,automatic_training_continuation=False,
        network_cap_bytes=10*1024**3,prior_bytes=read(ROOT/'observation_model/outputs/phase0/download_budget.json')['received_bytes'],
        old_roles_immutable=True,heldout_to_train_promotion=False))
    for name,p in [('canonical_input.jsonl',source),('shared_input.jsonl',raw)]:
        (OUT/name).write_bytes(p.read_bytes())
    save(OUT/'source_hashes.json',{str(p):digest(p) for p in [source,raw,Path(__file__),Path(__file__).with_name('expansion_data.py')]})
    records=lines(OUT/'canonical_input.jsonl');old={r['id']:r for r in lines(OUT/'shared_input.jsonl')}
    tasks=[]
    for r in records:
        if r['effective_partition'] not in ['TRAIN','VAL']:continue
        if r['data_status']=='NATIVE_METADATA_READY_PENDING_G1':kind='native'
        elif r['metadata_origin']=='legacy' and r['instrument'] and old[r['source_id']].get('physical_transforms'):kind='cached_s2' if r['sensor']=='S2' else 'cached_landsat'
        elif r['metadata_origin']=='local_unannotated' and r['effective_partition']=='TRAIN':kind='landsat_for_local_anchor'
        else:continue
        tasks.append(dict(id=r['id'],kind=kind,record=r,original=old.get(r['source_id'])))
    (OUT/'tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tasks),encoding='utf-8')
    save(OUT/'queue.json',dict(tasks=len(tasks),kinds=dict(Counter(t['kind'] for t in tasks))))
    print('PREPARED',dict(Counter(t['kind'] for t in tasks)),flush=True)


def run_local():
    tasks=[t for t in lines(OUT/'tasks.jsonl') if t['kind'] in ['native','cached_s2']]
    def one(t):
        side=OUT/'cache'/(t['id']+'.json')
        if side.exists():return dict(id=t['id'],status='reused')
        try:
            if t['kind']=='native':existing_native(t['record'])
            else:cached_native(t['record'],t['original'])
            return dict(id=t['id'],kind=t['kind'],status='accepted')
        except Exception as exc:return dict(id=t['id'],kind=t['kind'],status='excluded',error=str(exc).split('?')[0][:240])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(one,t) for t in tasks]),1):
            result=f.result();append(OUT/'local_events.jsonl',result)
            if i%100==0:print('LOCAL',i,'of',len(tasks),flush=True)
            if i%500==0:publish('local_admission_running')
    publish('local_admission_complete_remote_QA_pending')


def landsat_anchor(t,net,heldout_acq,heldout_geos):
    r=t['record'];original=t['original'];date=original.get('date',original.get('datetime'))
    if not date:raise ValueError('anchor actual date unavailable')
    bbox=r['bbox'];center=box(*bbox).centroid
    group=net.anchor_groups[t['id']]
    cache=OUT/'catalogue'/(group['key']+'.json')
    if cache.exists():features=read(cache)['features']
    else:
        d=utc_date(date);start=d-timedelta(days=16);end=d+timedelta(days=16)
        with net.catalogue_lock:
            if cache.exists():result=read(cache)
            else:
                result=net.json('GET',PC+'/search',params=dict(collections='landsat-c2-l2',bbox=','.join(map(str,group['bbox'])),datetime=start.isoformat()+'/'+end.isoformat(),limit=50))
                save(cache,result)
        features=result.get('features',[])
    options=[]
    for f in features:
        if not f['id'].startswith('LC08'):continue
        _,acq=identity(f['id'],f['properties']['datetime'],'L8')
        if acq in heldout_acq or not shape(f['geometry']).covers(center):continue
        if any(box(*bbox).intersection(heldout_geos.geometries[int(j)]).area>1e-12 for j in heldout_geos.query(box(*bbox))):continue
        options.append(f)
    if not options:raise ValueError('no existing-location L8 scene within16days outside all heldout acquisitions')
    options.sort(key=lambda f:(abs((utc_date(f['properties']['datetime'])-utc_date(date)).total_seconds()),f['properties'].get('eo:cloud_cover',100)))
    failures=[]
    for feature in options[:3]:
        try:return export_landsat_feature(t,feature,net,heldout_geos)
        except Exception as exc:
            if 'DOWNLOAD_BUDGET_EXHAUSTED' in str(exc):raise
            failures.append(str(exc).split('?')[0][:160])
    raise ValueError('up to3 existing-location scenes failed: '+'; '.join(failures))


def export_landsat_feature(t,feature,net,heldout_geos):
    r=t['record'];original=t['original'];date=original.get('date',original.get('datetime'));bbox=r['bbox']
    folder=OUT/'raw'/t['id']/feature['id'];save(folder/'product.json',dict(id=feature['id'],properties=feature['properties'],assets=feature['assets']))
    crs=original['crs'];transform=original['transform'];shape_xy=[128,128]
    keys=['blue','green','red','nir08','swir16','swir22'];band_names=['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']
    paths={}
    for k in ['qa_pixel','qa_radsat','qa_aerosol']:
        if k not in feature['assets']:raise ValueError('missing required original Landsat asset '+k)
        p=folder/(k+'.tif');net.crop(feature['assets'][k]['href'],crs,transform,shape_xy,p,qa=False);paths[k]=p
    qa=[];qa_valid=[]
    for k in ['qa_pixel','qa_radsat','qa_aerosol']:
        with rasterio.open(paths[k]) as f:qa.append(f.read(1));qa_valid.append(f.dataset_mask()>0);qt=f.transform;qc=f.crs
    quality=((qa[0].astype('uint16')&63)==0)&(qa[1]==0)&aerosol_valid(qa[2])&qa_valid[0]&qa_valid[1]&qa_valid[2]
    if quality.mean()<.5:raise ValueError('native QA coverage below50percent before reflectance transfer')
    for k in keys:
        if k not in feature['assets']:raise ValueError('missing required original Landsat asset '+k)
        p=folder/(k+'.tif');net.crop(feature['assets'][k]['href'],crs,transform,shape_xy,p,qa=False);paths[k]=p
    arrays={};bands=[]
    for k,b in zip(keys,band_names):
        with rasterio.open(paths[k]) as f:
            dn=f.read(1);valid=(f.dataset_mask()>0)&(dn!=0);tf=f.transform;cc=f.crs
        if tf!=qt or cc!=qc or quality.shape!=dn.shape:raise ValueError('Landsat bands/QA native support mismatch')
        valid &= quality;x=dn.astype('float32')*.0000275-.2;arrays[b]=(x,valid)
        bands.append(dict(band_key=b,source=str(paths[k]),crs=str(cc),transform=list(tf)[:6],shape=list(x.shape),native_pixel_support_m=[30,30],scale=.0000275,offset=-.2))
    from rasterio.warp import transform_bounds
    actual=transform_bounds(cc,'EPSG:4326',*rasterio.windows.bounds(rasterio.windows.Window(0,0,dn.shape[1],dn.shape[0]),tf))
    geometry=box(*actual)
    if any(geometry.intersection(heldout_geos.geometries[int(j)]).area>1e-12 for j in heldout_geos.query(geometry)):
        raise ValueError('enclosing native footprint touches existing heldout support')
    _,acq=identity(feature['id'],feature['properties']['datetime'],'L8')
    return write_observation(t['id']+'_L8',arrays,dict(sensor='L8',instrument='LC08',product=feature['id'],date=feature['properties']['datetime'],role='TRAIN',bbox=list(actual),requested_anchor_bbox=bbox,isolation_group=r['isolation_group'],bands=bands,processing_domain='original_L2_SR',provenance='new_native_Landsat_existing_local_anchor',anchor_id=r['id'],anchor_date=date,pair_semantics='same_geography_not_certified_cross_source_pair',acquisition_key=acq))


def run_remote():
    net=Network();records=lines(OUT/'canonical_input.jsonl')
    # Reserve the separately metered S2 calibration and the interrupted reads.
    net.budget.cap-=256*1024**2+32*1024**2
    heldout_acq={r['acquisition_key'] for r in records if r['effective_partition']!='TRAIN' and r['acquisition_key']}
    heldout_geos=STRtree([box(*r['bbox']) for r in records if r['effective_partition']!='TRAIN'])
    tasks=lines(OUT/'tasks.jsonl');tasks=[t for t in tasks if t['kind'] in ['cached_landsat','landsat_for_local_anchor']]
    previous={r['id'] for r in lines(OUT/'remote_events.jsonl') if r['status'] in ['accepted','excluded']} if (OUT/'remote_events.jsonl').exists() else set()
    groups=defaultdict(list)
    for t in tasks:
        if t['kind']=='landsat_for_local_anchor':
            b=t['record']['bbox'];o=t['original'];date=o.get('date',o.get('datetime'))
            g=(o.get('event_id','local'),date[:10],int(np.floor((b[0]+b[2])/2)),int(np.floor((b[1]+b[3])/2)))
            groups[g].append(t)
    net.anchor_groups={};net.catalogue_lock=threading.Lock()
    for g,values in groups.items():
        boxes=[t['record']['bbox'] for t in values];entry=dict(key=hashlib.sha256(str(g).encode()).hexdigest()[:24],bbox=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)])
        for t in values:net.anchor_groups[t['id']]=entry
    def one(t):
        try:
            if t['kind']=='cached_landsat':cached_native(t['record'],t['original'],net)
            else:landsat_anchor(t,net,heldout_acq,heldout_geos)
            return dict(id=t['id'],kind=t['kind'],status='accepted')
        except Exception as exc:
            error=str(exc).split('?')[0][:240]
            state='budget_pending' if 'DOWNLOAD_BUDGET_EXHAUSTED' in error else 'retryable_network_error' if any(code in error.lower() for code in ['429','timed out','timeout','403','502','503','connection']) else 'excluded'
            return dict(id=t['id'],kind=t['kind'],status=state,error=error)
    try:
        pending=iter([t for t in tasks if t['id'] not in previous]);count=0;stop=False;state='all_data_tasks_attempted_model_training_disabled';transient_streak=0
        with ThreadPoolExecutor(max_workers=4) as pool:
            active={}
            while True:
                while len(active)<4 and not stop:
                    t=next(pending,None)
                    if t is None:break
                    active[pool.submit(one,t)]=t['id']
                if not active:break
                ready,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                if not ready:print('DATA_HEARTBEAT',count,'bytes',net.budget.received,flush=True)
                for f in ready:
                    result=f.result();active.pop(f);count+=1;append(OUT/'remote_events.jsonl',result)
                    print('REMOTE',count,json.dumps(result,ensure_ascii=False),flush=True)
                    transient_streak=transient_streak+1 if result['status']=='retryable_network_error' else 0
                    if result['status']=='budget_pending':stop=True;state='download_cap_reached_remaining_tasks_pending'
                    if transient_streak>=3:stop=True;state='network_block_requires_diagnosis_resume_same_data_queue'
                    if count%25==0:publish('native_QA_and_Landsat_completion_running')
        publish(state)
    finally:net.budget.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--local',action='store_true');p.add_argument('--remote',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    if a.local:run_local()
    if a.remote:run_remote()


if __name__=='__main__':main()

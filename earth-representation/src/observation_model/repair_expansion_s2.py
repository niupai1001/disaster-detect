"""Resolve the provider's already-applied BOA offset in a new data version.

Original caches, provisional exports and the Landsat queue stay untouched.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime,timezone
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window,transform as window_transform

from .expansion_data import ROOT,OUT,Network,ParallelBudget,read,lines,save,append,digest

DEST=OUT/'s2_calibrated_v2'


def sources():
    tasks=lines(OUT/'tasks.jsonl');shared={r['id']:r for r in lines(OUT/'shared_input.jsonl')};items=[]
    for t in tasks:
        r=t['record'];side=OUT/'cache'/(t['id']+'.json')
        if r['sensor']!='S2' or not side.exists():continue
        original=t.get('original') or shared.get(r['source_id'],{});props=original.get('properties') or r.get('properties',{})
        if t['kind']=='cached_s2':href=original['physical_transforms'][0]['href']
        else:
            cal=original.get('calibration',{});c=next(iter(cal.values()),{});href=c.get('source',{}).get('asset_metadata',{}).get('href','')
        if 'sentinel-cogs.s3.' not in href:continue
        items.append(dict(id=t['id'],row=read(side),props=props,href=href,kind=t['kind']))
    return items


def reference(item,net):
    props=item['props'];product=props.get('s2:product_uri','')
    m=re.search(r'(S2[ABC])_MSIL2A_(\d{8}T\d{6}).*?_T(\d{2}[A-Z]{3})_',product)
    if not m:raise ValueError('exact ESA product timestamp/tile unavailable')
    identifier=f'{m[1]}_T{m[3]}_{m[2]}_L2A';meta=DEST/'references'/(identifier+'.json')
    if meta.exists():feature=read(meta)
    else:
        try:feature=net.json('GET','https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items/'+identifier)
        except Exception as exc:
            if getattr(getattr(exc,'response',None),'status_code',None)!=404:raise
            day=item['row']['date'][:10];b=item['row']['bbox']
            result=net.json('GET','https://earth-search.aws.element84.com/v1/search',params=dict(collections='sentinel-2-c1-l2a',bbox=','.join(map(str,b)),datetime=day+'T00:00:00Z/'+day+'T23:59:59Z',limit=10))
            date=datetime.fromisoformat(item['row']['date'].replace('Z','+00:00'));date=date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date
            matches=[f for f in result.get('features',[]) if f['id'].startswith(m[1]+'_T'+m[3]+'_') and abs((datetime.fromisoformat(f['properties']['datetime'].replace('Z','+00:00'))-date).total_seconds())<120]
            if len(matches)!=1:raise ValueError('unique exact-day/instrument/tile reference unavailable')
            feature=matches[0]
        save(meta,feature)
    row=item['row'];band=next(b for b in row['bands'] if b['band_key']=='B04')
    with np.load(row['cache']) as z:value=z['B04'];valid=z['B04__valid']
    h,w=value.shape;size=min(32,h,w);y=(h-size)//2;x=(w-size)//2
    if valid[y:y+size,x:x+size].mean()<.5:
        y,x=max(((yy,xx) for yy in range(0,h-size+1,size) for xx in range(0,w-size+1,size)),key=lambda p:valid[p[0]:p[0]+size,p[1]:p[1]+size].mean())
    tf=window_transform(Window(x,y,size,size),Affine(*band['transform']));asset=feature['assets']['red'];raw=DEST/'references'/(item['id']+'.tif')
    net.crop(asset['href'],band['crs'],list(tf)[:6],[size,size],raw,qa=True)
    with rasterio.open(raw) as f:dn=f.read(1);q=(f.dataset_mask()>0)&(dn!=0)
    q &= valid[y:y+size,x:x+size]
    if q.sum()<32:raise ValueError('not enough genuine reference pixels')
    calibration=asset['raster:bands'][0];target=dn.astype('float32')*calibration['scale']+calibration['offset'];old=value[y:y+size,x:x+size]
    e=(target-old)[q];median=float(np.median(e));candidates=[0.,.1]
    delta=min(candidates,key=lambda d:abs(median-d));rmse=float(np.sqrt(np.mean((e-delta)**2)));mae=float(np.median(np.abs(e-delta)))
    if abs(median-delta)>.01 or mae>.01 or rmse>.03:raise ValueError('independent reference does not identify a unique0/0.1 provider offset')
    return dict(id=item['id'],reference_id=feature['id'],requested_reference_id=identifier,reference_source=asset['href'],reference_band='B04',reference_window_yx_shape=[y,x,size,size],reference_processing_baseline=feature['properties'].get('s2:processing_baseline'),reference_window_sha256=digest(raw),valid_pixels=int(q.sum()),median_reference_minus_prepared=median,
                selected_delta=delta,rmse_after=rmse,median_absolute_error_after=mae,rmse_before=float(np.sqrt(np.mean(e**2))),reference_scale=calibration['scale'],reference_offset=calibration['offset'])


def write_corrected(item,delta,evidence):
    row=dict(item['row']);path=DEST/'cache'/(row['id']+'.npz');side=path.with_suffix('.json')
    if side.exists():return
    with np.load(row['cache']) as original:
        arrays={k:original[k].copy() for k in original.files}
    for b in row['bands']:
        k=b['band_key'];q=arrays[k+'__valid'];arrays[k]=np.where(q,arrays[k]+np.float32(delta),0).astype('float32')
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    row['prepared_parent_cache']=row['cache'];row['prepared_parent_sha256']=row['cache_sha256'];row['cache']=str(path);row['cache_sha256']=digest(path)
    row['processing_domain']='S2_L2_SR_provider_harmonized';row['calibration_revision']=dict(additive_correction=delta,evidence=evidence,
        source_flag=item['props'].get('earthsearch:boa_offset_applied'),source_baseline=item['props'].get('s2:processing_baseline'),
        provider_low_signal_floor='Pre-C1 provider may have clamped values before delivery; no extra clipping applied by this repair')
    for b in row['bands']:
        b['prepared_declared_offset']=b.get('offset');b['repaired_additive_delta']=delta
        if b.get('offset') is not None:b['offset']+=delta
    save(side,row)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    items=sources();print('S2_PROVIDER_ROWS',len(items),flush=True)
    main_bytes=read(OUT/'network.json')['received_bytes']
    prior=read(ROOT/'observation_model/outputs/phase0/download_budget.json')['received_bytes']
    srf=read(ROOT/'data/physics/srf/acquisition_manifest.json')['received_bytes']
    available=10*1024**3-prior-srf-main_bytes-32*1024**2
    net=Network();net.budget=ParallelBudget(OUT/'s2_calibration_network.json',min(512*1024**2,available))
    controls=[]
    known=[x for x in items if x['props'].get('earthsearch:boa_offset_applied') is True]
    for instrument in ['S2A','S2B','S2C']:
        match=next((x for x in known if x['row']['instrument']==instrument),None)
        if match:controls.append(reference(match,net))
    if not controls or any(abs(x['selected_delta']-.1)>1e-6 for x in controls):raise RuntimeError('known-applied provider policy failed independent numeric controls')
    save(DEST/'known_applied_controls.json',controls)
    uncertain=[];counts=Counter()
    for item in items:
        props=item['props'];baseline=float(props.get('s2:processing_baseline',999))
        if props.get('earthsearch:boa_offset_applied') is True:
            write_corrected(item,.1,'provider flag true plus independent Collection1 controls');counts['known_applied_corrected']+=1
        elif baseline<4:
            write_corrected(item,0.,'pre04 no additive offset');counts['pre04_retained']+=1
        else:uncertain.append(item)
    print('S2_AMBIGUOUS_REFERENCES',len(uncertain),flush=True)
    def one(item):
        try:
            result=reference(item,net);write_corrected(item,result['selected_delta'],'per-observation independent Collection1 numeric reference')
            return dict(result,status='resolved')
        except Exception as exc:return dict(id=item['id'],status='unresolved',error=str(exc).split('?')[0][:200])
    history={r['id']:r for r in lines(DEST/'reference_results.jsonl')} if (DEST/'reference_results.jsonl').exists() else {}
    pending=[x for x in uncertain if not (DEST/'cache'/(x['id']+'.json')).exists() and (x['id'] not in history or history[x['id']].get('error')!='unique exact-day/instrument/tile reference unavailable')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(one,x) for x in pending]),1):
            result=f.result();append(DEST/'reference_results.jsonl',result);counts[result['status']]+=1
            if i%20==0:print('S2_REFERENCE',i,'of',len(uncertain),dict(counts),flush=True)
    history={r['id']:r for r in lines(DEST/'reference_results.jsonl')}
    counts['resolved']=sum((DEST/'cache'/(x['id']+'.json')).exists() for x in uncertain)
    counts['unresolved']=len(uncertain)-counts['resolved']
    save(DEST/'summary.json',dict(status='CALIBRATION_RESOLVED_OR_EXPLICITLY_PENDING',counts=dict(counts),source_rows=len(items),network=read(OUT/'s2_calibration_network.json'),
                                 controls=controls,old_sources_unchanged=True,optimizer_updates=0))
    print('S2_REPAIR_COMPLETE',dict(counts),flush=True)


if __name__=='__main__':main()

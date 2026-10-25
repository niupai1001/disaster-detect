"""Prepare existing unannotated TRAIN-region S2; never read MASK or fetch data."""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform
from shapely.strtree import STRtree

from prepare_use_external import exclusion

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from landslide_seg.models.spectral_contract import S2_WAVELENGTH_UM

BANDS=tuple(S2_WAVELENGTH_UM)
OUT=ROOT/'outputs/use/external'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value):
    return value.decode('utf-8') if isinstance(value,bytes) else str(value)


def read_unannotated(path,allowed,forbidden):
    event=path.stem.rsplit('_s2_',1)[0]
    if event not in allowed:return None,'outside_train_events'
    with h5py.File(path,'r') as source:
        if text(source.attrs.get('annotated','')).lower()!='false':return None,'not_explicitly_unannotated'
        crs=text(source.attrs['crs'])
        grid=[float(v) for v in text(source['spatial_ref'].attrs['GeoTransform']).split()]
        x0,dx,rx,y0,ry,dy=grid
        if dx!=10 or dy!=-10 or rx!=0 or ry!=0:raise ValueError('Unexpected local source grid')
        footprint=transform(Transformer.from_crs(crs,4326,always_xy=True).transform,box(x0,y0-1280,x0+1280,y0))
        if any(footprint.intersects(g) for g in forbidden):return None,'withheld_event_buffer'
        scl=source['SCL'][:]
        if scl.shape[1:]!=(128,128):raise ValueError('Unexpected source SCL shape')
        quality=np.isin(scl,(4,5,6,7))
        ranking=sorted(range(len(scl)),key=lambda t:(-float(quality[t].mean()),t))
        selected=None
        for t in ranking:
            if quality[t].mean()<.8:break
            dn=np.stack([source[b][t] for b in BANDS])
            valid=np.isfinite(dn)&(dn>0)&quality[t][None]
            if dn.shape!=(10,128,128):raise ValueError('Unexpected local optical shape')
            if valid.all(0).mean()>=.8:
                selected=t;break
        if selected is None:return None,'no_80pct_valid_observation'
        units=text(source['time'].attrs['units'])
        if not units.startswith('days since '):raise ValueError('Unknown source time units')
        date=(datetime.fromisoformat(units[len('days since '):])+timedelta(days=float(source['time'][selected]))).isoformat()
        metadata=np.asarray([[S2_WAVELENGTH_UM[b],0,10 if b in ('B02','B03','B04','B08') else 20] for b in BANDS],dtype='float32')
        known=np.ones((10,3),dtype=bool);known[:,1]=False
        arrays=dict(x=dn.astype('float32')*.0001,pixel_valid=valid,metadata=metadata,known=known)
        row=dict(sensor='s2',origin='local_unannotated',event_id=event,source_uri=str(path.resolve()),
            source_sha256=sha(path),source_annotated=False,time_index=selected,date=date,band_ids=list(BANDS),
            crs=crs,transform=[dx,rx,x0,ry,dy,y0],bbox=list(footprint.bounds),geometry=mapping(footprint),
            physical_scale=.0001,physical_offset=0,processing='existing project harmonized S2',
            metadata_source='spectral_contract nominal wavelengths; bandwidth unknown',
            group=f'{crs}:{int(np.floor(x0/5120))}:{int(np.floor(y0/5120))}')
    return (arrays,row),None


def unique_locations(rows):
    geos=[shape(r['geometry']) for r in rows];tree=STRtree(geos);kept=set();duplicates=[]
    order=sorted(range(len(rows)),key=lambda i:hashlib.sha256(('1234:'+rows[i]['file']).encode()).hexdigest())
    for i in order:
        conflict=next((int(j) for j in tree.query(geos[i],predicate='intersects')
            if int(j) in kept and geos[i].intersection(geos[int(j)]).area>1e-14),None)
        if conflict is None:kept.add(i)
        else:duplicates.append(dict(file=rows[i]['file'],retained=rows[conflict]['file']))
    return [r for i,r in enumerate(rows) if i in kept],duplicates


def main():
    protocol=ROOT/'manifests/v4r1/protocol.json'
    design=json.loads(protocol.read_text())['design'];allowed=set(design['train'])
    protected=set().union(*(set(v) for k,v in design.items() if k!='train'))
    forbidden=exclusion(events=protected)
    manifest=OUT/'local_unlabeled_manifest.json';acceptance=OUT/'local_acceptance.json'
    folder=OUT/'local_unlabeled';folder.mkdir(exist_ok=True)
    previous=json.loads(manifest.read_text()) if manifest.exists() else None
    rows=[];rejected=Counter()
    for i,path in enumerate(sorted((ROOT/'data/sen12').glob('*.nc'))):
        result,reason=read_unannotated(path,allowed,forbidden)
        if result is None:
            rejected[reason]+=1;continue
        arrays,row=result;target=folder/(path.stem+'.npz')
        if target.exists():
            with np.load(target) as old:
                if not all(np.array_equal(old[k],v) for k,v in arrays.items()):raise ValueError('Existing local window changed')
        else:
            temporary=target.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays);temporary.replace(target)
        row.update(file='local_unlabeled/'+target.name,sha256=sha(target));rows.append(row)
        if len(rows)%100==0:print('accepted local',len(rows),'scanned',i+1,flush=True)
    if len(rows)<32:raise ValueError('Insufficient local unannotated observations')
    rows,duplicates=unique_locations(rows)
    rejected['overlapping_location']=len(duplicates)
    geos=[shape(r['geometry']) for r in rows]
    external=json.loads((OUT/'current_manifest.json').read_text())
    remote_tree=STRtree([box(*r['bbox']) for r in external])
    for g in geos:
        if len(remote_tree.query(g,predicate='intersects')):raise ValueError('Local/external locations overlap')
    if previous is not None and previous!=json.loads(json.dumps(rows)):raise ValueError('Frozen local manifest differs from current source scan')
    if previous is None:manifest.write_text(json.dumps(rows,indent=1),encoding='utf-8')
    groups=sorted({r['group'] for r in rows},key=lambda g:hashlib.sha256(str(('local_split1234',g)).encode()).hexdigest())
    held=set(groups[:max(1,round(len(groups)*.05))])
    train=[r['file'] for r in rows if r['group'] not in held];validation=[r['file'] for r in rows if r['group'] in held]
    if not train or not validation:raise ValueError('Empty local geographic split')
    result=dict(ok=True,locations=len(rows),by_event=dict(Counter(r['event_id'] for r in rows)),
        manifest_sha256=sha(manifest),protocol_sha256=sha(protocol),code_sha256=sha(Path(__file__)),
        source_selection='annotated=False only; never reads MASK; one clearest qualifying observation per location',
        rejection_counts=dict(rejected),overlap_coalescing=duplicates,geographic_group='CRS/5120m block across event names; seed1234 5% groups held out',
        train_ids=train,validation_ids=validation,protected_events=sorted(protected),
        target='masked standardized reflectance reconstruction; no segmentation labels',
        validation_boundary='Local holdout is inside known TRAIN events and is not unseen-event segmentation evidence')
    acceptance.write_text(json.dumps(result,indent=1),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('locations','by_event','rejection_counts')},indent=1),flush=True)


if __name__=='__main__':main()

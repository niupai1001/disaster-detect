"""Derive a closed v2 cohort; inherit roles and reject conflicting TRAIN only."""
import json
import math
import sys
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.strtree import STRtree

from audit_data import ROOT,SNAPSHOT,sha
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_use_external import exclusion


def enrich(row):
    row=dict(row)
    if sha(row['cache_uri'])!=row['sha256']:raise ValueError('Changed cache '+row['id'])
    with np.load(row['cache_uri'],allow_pickle=False) as arr:
        x,valid,metadata,known=(arr[k] for k in ('x','pixel_valid','metadata','known'))
        if not np.isfinite(x[valid]).all():raise ValueError('Nonfinite cache')
        row['shape']=list(x.shape)
        row['native_resolution']=[float(m[2]) if k[2] else None for m,k in zip(metadata,known)]
        row['native_resolution_known']=[bool(k[2]) for k in known]
        row['channel_metadata']=metadata.tolist();row['channel_metadata_known']=known.tolist()
        # Effective instrument support cannot be reconstructed from a resampled cache.
        row['effective_resolution']=[None]*len(x)
        row['effective_resolution_known']=[False]*len(x)
    a,b,c,d,e,f=row['transform'];_,h,w=row['shape']
    poly=Polygon([(a*u+b*v+c,d*u+e*v+f) for u,v in ((0,0),(w,0),(w,h),(0,h))])
    geo=transform(Transformer.from_crs(row['crs'],4326,always_xy=True).transform,poly)
    row['footprint_wgs84']=geo.__geo_interface__
    row['sampling_gsd_m']=[math.hypot(a,d),math.hypot(b,e)]
    row['validation_partition']='legacy599' if row['id'].startswith('old_') else 'additional'
    row['radiometry_evidence']={k:row[k] for k in ('units','physical_transforms','physical_scale','physical_offset','calibration','processing') if k in row}
    row['resampling_history']={'cache_grid':row['transform'],'source_window':row.get('source_window'),
                              'registration':row.get('registration'),'unknown_steps_preserved':True}
    return row,geo


def main():
    out=ROOT/'outputs/use_next_v2/data/data_release_v1'
    out.mkdir(parents=True,exist_ok=False)
    identity=json.loads((SNAPSHOT/'snapshot_identity.json').read_text())
    assert sha(SNAPSHOT/'manifest.jsonl')==identity['manifest_sha256']
    rows=[json.loads(l) for l in (SNAPSHOT/'manifest.jsonl').read_text().splitlines()]
    with ThreadPoolExecutor(4) as pool:enriched=list(pool.map(enrich,rows))
    rows,geos=map(list,zip(*enriched));tree=STRtree(geos);parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):parent[find(b)]=find(a)
    roots={}
    for i,r in enumerate(rows):
        for key in (('group',r.get('group')),('site',r.get('site_id')),('pixels',r['sha256'])):
            if key[1] is not None:
                if key in roots:join(i,roots[key])
                else:roots[key]=i
        for j in tree.query(geos[i],predicate='intersects'):join(i,int(j))
    roles=defaultdict(set)
    for i,r in enumerate(rows):roles[find(i)].add(r['role'])
    held=STRtree([g for r,g in zip(rows,geos) if r['role']!='TRAIN'])
    protected=STRtree(exclusion(events={'hiroshima','thrissur','italy','newzealand'}))
    admitted=[];rejected=[];hr=[];guard=[]
    for i,(r,g) in enumerate(zip(rows,geos)):
        r['v2_geographic_group']=rows[find(i)]['id']
        if r['sensor']=='hr':hr.append(r);continue
        if r['role']=='GUARD':guard.append(r);continue
        reasons=[]
        if r['role']=='TRAIN':
            if roles[find(i)]!={'TRAIN'}:reasons.append('connected_source_site_or_footprint_cross_role')
            local=f'+proj=aeqd +lat_0={g.centroid.y} +lon_0={g.centroid.x} +datum=WGS84 +units=m'
            buffered=transform(Transformer.from_crs(local,4326,always_xy=True).transform,
                               transform(Transformer.from_crs(4326,local,always_xy=True).transform,g).buffer(1280))
            if len(held.query(buffered,predicate='intersects')):reasons.append('heldout_1280m_buffer')
            if len(protected.query(g,predicate='intersects')):reasons.append('supervised_heldout_10km_buffer')
        if reasons:rejected.append({'id':r['id'],'reasons':reasons})
        else:admitted.append(r)
    for name,values in (('manifest',admitted),('rejected',rejected),('hr_readonly',hr),('guard',guard)):
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    pair_source=ROOT/'outputs/use_next_v2/data/pair_eligibility.jsonl'
    (out/'pairs.jsonl').write_bytes(pair_source.read_bytes())
    counts=Counter(r['role']+'/'+r['sensor'] for r in admitted)
    weights={s:math.sqrt(counts['TRAIN/'+s]) for s in ('s2','landsat','s1')}
    total=sum(weights.values())
    result=dict(ok=True,manifest_sha256=sha(out/'manifest.jsonl'),pairs_sha256=sha(out/'pairs.jsonl'),
                source_identity=identity,counts=dict(counts),rejected=len(rejected),hr_readonly=len(hr),guard=len(guard),
                legacy_validation=sum(r['role']=='PRETRAIN_VAL' and r['validation_partition']=='legacy599' for r in admitted),
                modality_probabilities={s:v/total for s,v in weights.items()},
                p4_eligible_pairs=0,p4_status='SKIP: change/registration screening absent',
                normalizations='No new fitted statistics; physical units per row, official AnySat statistics only for native AnySat; stats_core25 unchanged',
                unknown_effective_resolution='No inferred effective resolution or high-frequency targets from interpolated caches',
                admission_scope='Source integration radiometry/QA retained; full cache SHA and spatial role/buffer recheck; HR excluded',
                code_sha256=sha(__file__),formal_updates=0)
    (out/'release.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()

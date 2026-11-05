"""Read-only exact QA support geometry for candidate pairs; never admission."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import shapely
from shapely.affinity import translate
import torch
from .data import NativeOpticalDataset,sha256
from .geometry_views import northing_shift
from .pilot_protocol import load_study


def mask_region(mask,affine):
    """Union contiguous native row runs, preserving holes and pixel edges."""
    a,b,left,d,e,top=affine
    if b or d or a<=0 or e>=0:raise ValueError('North-up native metric cells required')
    boxes=[]
    for row,values in enumerate(np.asarray(mask,dtype=bool)):
        edges=np.flatnonzero(np.diff(np.pad(values.astype(np.int8),(1,1))))
        for start,end in zip(edges[::2],edges[1::2]):
            boxes.append(shapely.box(left+a*start,top+e*(row+1),left+a*end,top+e*row))
    return shapely.union_all(boxes)


def main(args):
    torch.set_num_threads(1);started=time.perf_counter();study=load_study(args.study)
    before=sha256(args.candidates);catalog=json.loads(args.candidates.read_text(encoding='utf-8'))
    if catalog['members_sha256']!=study['source_identities']['members']:raise ValueError('Different source manifest')
    datasets={role:NativeOpticalDataset(study['release'],study['contracts'],role=role) for role in ('TRAIN','VAL')}
    indices={role:{row['id']:i for i,row in enumerate(ds.rows)} for role,ds in datasets.items()}
    cache={}
    def support(role,identifier):
        key=(role,identifier)
        if key not in cache:
            obs=datasets[role][indices[role][identifier]];crs=obs.bands[0].crs;regions=[]
            for band in obs.bands:
                shift=northing_shift(band.crs,crs)
                regions.append(translate(mask_region(band.valid_mask.numpy(),band.affine),yoff=shift))
            cache[key]=dict(crs=crs,any=shapely.union_all(regions),all=shapely.intersection_all(regions),bands=len(regions))
        return cache[key]
    rows=[];counts=Counter()
    for pair in catalog['pairs']:
        row=dict(S2_id=pair['S2_id'],Landsat_id=pair['Landsat_id'],role=pair['role'],v3_admitted=False)
        if pair['geometry_status']=='common_projection_operator_required':
            row['status']='UNMEASURED_COMMON_PROJECTION_REQUIRED'
        else:
            a=support(pair['role'],pair['S2_id']);b=support(pair['role'],pair['Landsat_id'])
            shift=northing_shift(b['crs'],a['crs'])
            for kind in ('any','all'):
                other=translate(b[kind],yoff=shift);overlap=a[kind].intersection(other).area
                row[kind+'_band_valid_overlap_m2']=overlap
                row[kind+'_band_fraction_of_S2']=overlap/a[kind].area if a[kind].area else None
                row[kind+'_band_fraction_of_Landsat']=overlap/other.area if other.area else None
            row['status']='POSITIVE_ANY_BAND_QA_OVERLAP' if row['any_band_valid_overlap_m2']>0 else 'NO_ANY_BAND_QA_OVERLAP'
        rows.append(row);counts[row['status']]+=1
    if sha256(args.candidates)!=before:raise ValueError('Candidate catalog changed')
    report=dict(status='CANDIDATE_QA_SUPPORT_MEASURED_NOT_PAIR_ADMISSION',candidates_sha256=before,source_identities=study['source_identities'],counts=dict(counts),unique_observations_read=len(cache),pairs=rows,seconds=time.perf_counter()-started,optimizer_updates=0,admitted_pairs=0,limitations=['QA masks are existing release masks; parent-only QA provenance remains unresolved','Any/all refer to union/intersection of valid footprints across native bands, not task visibility','Exact native pixel support only; no reflectance resampling or image registration','No temporal/change-risk/processing-version threshold selected; no training membership changed'])
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','counts','unique_observations_read','seconds')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--candidates',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())

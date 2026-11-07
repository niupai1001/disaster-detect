"""Offset diagnostics on identical fully covered target pixels, not supervision."""
import argparse,json,math,time
from pathlib import Path
from itertools import product
import numpy as np
import torch
from .data import NativeOpticalDataset,sha256
from .geometry_views import common_metric_view
from .pair_registration import area_projection,registration_score


def fixed_support_scores(source,target,offsets_m):
    if not offsets_m or any(not math.isfinite(x) for x in offsets_m):raise ValueError('Explicit finite diagnostic offsets required')
    offsets=list(product(offsets_m,repeat=2));predictions=[];areas=[]
    for dx,dy in offsets:
        prediction,area=area_projection(source,target,dx=dx,dy=dy);predictions.append(prediction);areas.append(area)
    area_stack=np.stack(areas);pixel_area=target.affine[0]*(-target.affine[4])
    # Numerical geometry tolerance only; no QA acceptance threshold or training mask.
    full=np.isclose(area_stack,pixel_area,rtol=1e-10,atol=0).all(axis=0)&target.keep().cpu().numpy()
    fixed=np.where(full,pixel_area,0.)
    rows=[dict(dx_m=dx,dy_m=dy,**registration_score(prediction,fixed,target)) for (dx,dy),prediction in zip(offsets,predictions)]
    valid=[row for row in rows if row['correlation'] is not None]
    best=max(valid,key=lambda row:row['correlation']) if valid else None
    return dict(fixed_target_pixels=int(full.sum()),fixed_target_area_m2=float(fixed.sum()),target_valid_pixels=int(target.keep().sum()),full_coverage_numeric_rtol=1e-10,offsets=rows,best_measured_correlation=best,best_on_search_boundary=best is not None and (best['dx_m'] in (min(offsets_m),max(offsets_m)) or best['dy_m'] in (min(offsets_m),max(offsets_m))))


def main(args):
    torch.set_num_threads(2);started=time.perf_counter();study=json.loads(args.study.read_text(encoding='utf-8'));catalog=json.loads(args.catalog.read_text(encoding='utf-8'));ds=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
    if ds.identities!=study['source_identities'] or catalog['members_sha256']!=ds.identities['members']:raise ValueError('Frozen source identity differs')
    lookup={row['id']:i for i,row in enumerate(ds.rows)};results=[]
    for value in args.pair:
        a_id,b_id=value.split(':');pair=next(p for p in catalog['pairs'] if p['S2_id']==a_id and p['Landsat_id']==b_id)
        if pair['role']!='TRAIN' or pair['geometry_status']=='common_projection_operator_required':raise ValueError('Explicit TRAIN exact-geometry pair required')
        a=ds[lookup[a_id]];b=common_metric_view(ds[lookup[b_id]],a.bands[0].crs);aa={x.record_id:x for x in a.bands};bb={x.record_id:x for x in b.bands};directions=[]
        for reverse in (False,True):
            bands=[]
            for names in args.band_pair:
                an,bn=names.split(':');source,target=(bb[bn],aa[an]) if reverse else (aa[an],bb[bn]);bands.append(dict(source_band=source.record_id,target_band=target.record_id,**fixed_support_scores(source,target,args.offsets_m)))
            directions.append(dict(source_id=b_id if reverse else a_id,target_id=a_id if reverse else b_id,bands=bands))
        for obs in (a,b):
            if sha256(obs.metadata['cache'])!=obs.metadata['cache_sha256']:raise ValueError('Cache changed during fixed-support audit')
        results.append(dict(S2_id=a_id,Landsat_id=b_id,time_gap_seconds=pair['time_gap_seconds'],S2_cache_sha256=a.metadata['cache_sha256'],Landsat_cache_sha256=b.metadata['cache_sha256'],directions=directions))
    report=dict(status='FIXED_FULL_COVERAGE_DIAGNOSTIC_NOT_ADMISSION',source_identities=ds.identities,catalog_sha256=sha256(args.catalog),offsets_m=args.offsets_m,selection='Explicit requested candidate IDs; no learned model, temporal cutoff or training selection',rows=results,optimizer_updates=0,admitted_pairs=0,seconds=time.perf_counter()-started,source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('pair_fixed_support.py','pair_registration.py')},limitations=['Diagnostic only: original B2 supervision and QA masks unchanged','Only target pixels fully covered by valid source at every offset retained for comparable scores','Spectral differences, texture and real changes remain confounded; high correlation is not admission','Finite offset grid gives no subpixel registration guarantee'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],pairs=len(results),seconds=report['seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--pair',action='append',required=True);p.add_argument('--band-pair',action='append',required=True);p.add_argument('--offsets-m',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

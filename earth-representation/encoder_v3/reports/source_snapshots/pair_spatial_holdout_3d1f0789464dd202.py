"""Spatial holdout registration diagnostics, never pair admission or coordinate edits."""
import argparse,json,time
from pathlib import Path
from itertools import product
import numpy as np
import torch
from .data import NativeOpticalDataset,sha256
from .geometry_views import common_metric_view
from .pair_registration import area_projection,registration_score


def spatial_holdout(source,target,offsets_m):
    offsets=list(product(offsets_m,repeat=2))
    if (0.,0.) not in offsets or not np.isfinite(offsets).all():raise ValueError('Finite explicit grid including zero required')
    predictions=[];areas=[]
    for dx,dy in offsets:
        p,a=area_projection(source,target,dx=dx,dy=dy);predictions.append(p);areas.append(a)
    pixel_area=target.affine[0]*(-target.affine[4]);full=np.isclose(np.stack(areas),pixel_area,rtol=1e-10,atol=0).all(0)&target.keep().cpu().numpy()
    h,w=full.shape;yy,xx=np.meshgrid(np.arange(h)+.5,np.arange(w)+.5,indexing='ij');labels=(yy>=h/2).astype(np.int64)*2+(xx>=w/2)
    fixed=np.where(full,pixel_area,0.);zero=offsets.index((0.,0.));rows=[]
    for quadrant in range(4):
        held=full&(labels==quadrant);fit=full&(labels!=quadrant)
        fit_area=np.where(fit,pixel_area,0.);held_area=np.where(held,pixel_area,0.)
        fit_scores=[registration_score(p,fit_area,target) for p in predictions];held_scores=[registration_score(p,held_area,target) for p in predictions]
        eligible=[i for i,s in enumerate(fit_scores) if s['correlation'] is not None]
        chosen=max(eligible,key=lambda i:fit_scores[i]['correlation']) if eligible else None
        valid=[i for i,s in enumerate(held_scores) if s['correlation'] is not None];own=max(valid,key=lambda i:held_scores[i]['correlation']) if valid else None
        baseline=held_scores[zero];after=None if chosen is None else held_scores[chosen]
        rows.append(dict(heldout_quadrant=quadrant,heldout_pixels=int(held.sum()),fit_pixels=int(fit.sum()),selected_source_shift_m=None if chosen is None else offsets[chosen],fit_score=None if chosen is None else fit_scores[chosen],heldout_zero=baseline,heldout_selected=after,heldout_own_best_shift_m=None if own is None else offsets[own],heldout_correlation_gain=None if after is None or after['correlation'] is None or baseline['correlation'] is None else after['correlation']-baseline['correlation'],heldout_rmse_gain=None if after is None or after['rmse'] is None or baseline['rmse'] is None else baseline['rmse']-after['rmse']))
    assert sum(r['heldout_pixels'] for r in rows)==int(full.sum())
    return dict(fixed_target_pixels=int(full.sum()),quadrant_definition='Split target-native centers at native window x/y midpoints; no resampling',offsets_m=offsets_m,rows=rows)


def main(args):
    if args.output.exists():raise ValueError('Preserve existing diagnostic')
    root=Path(__file__).resolve().parent;r=root/'reports';started=time.perf_counter();torch.set_num_threads(2)
    reference=json.loads(args.reference.read_text());study=json.loads((r/'pilot_study_candidate.json').read_text(encoding='utf-8'));dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
    if dataset.identities!=study['source_identities'] or reference['source_identities']!=dataset.identities:raise ValueError('Frozen source identity differs')
    lookup={row['id']:i for i,row in enumerate(dataset.rows)};rows=[]
    report=dict(status='RUNNING',optimizer_updates=0,admitted_pairs=0,correction_applied=False,source_identities=dataset.identities,reference_sha256=sha256(args.reference),source_sha256={n:sha256(root/n) for n in ('pair_spatial_holdout.py','pair_registration.py','geometry_views.py')},rows=rows,limits=['Fit offsets use three native quadrants; fourth quadrant is scored independently at that fixed offset','Not a temporal-change detector, alignment uncertainty bound or pair-admission policy','No native affine, value, QA, task, split, model or threshold changed'])
    def save():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        for original in reference['rows']:
            a=dataset[lookup[original['S2_id']]];b=common_metric_view(dataset[lookup[original['Landsat_id']]],a.bands[0].crs);observations={a.identifier:a,b.identifier:b};directions=[]
            if a.metadata['cache_sha256']!=original['S2_cache_sha256'] or b.metadata['cache_sha256']!=original['Landsat_cache_sha256']:raise ValueError('Reference cache differs')
            for direction in original['directions']:
                source=observations[direction['source_id']];target=observations[direction['target_id']];aa={band.record_id:band for band in source.bands};bb={band.record_id:band for band in target.bands};bands=[]
                for band in direction['bands']:
                    measured=spatial_holdout(aa[band['source_band']],bb[band['target_band']],reference['offsets_m'])
                    if measured['fixed_target_pixels']!=band['fixed_target_pixels']:raise ValueError('Fixed diagnostic support changed')
                    bands.append(dict(source_band=band['source_band'],target_band=band['target_band'],**measured))
                directions.append(dict(source_id=source.identifier,target_id=target.identifier,bands=bands))
            for observation in (a,b):
                if sha256(observation.metadata['cache'])!=observation.metadata['cache_sha256']:raise ValueError('Cache changed during diagnostic')
            rows.append(dict(S2_id=a.identifier,Landsat_id=b.identifier,time_gap_seconds=original['time_gap_seconds'],directions=directions));save()
        report['status']='SPATIAL_HOLDOUT_DIAGNOSTIC_COMPLETE_NOT_ADMISSION'
    except BaseException as exc:report.update(status='FAILED',error=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:report['seconds']=time.perf_counter()-started;save()
    print(json.dumps(dict(status=report['status'],pairs=len(rows),seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

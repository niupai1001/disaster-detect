"""Native-area cross-source registration evidence, never automatic pair admission."""
import argparse
import math
from itertools import product
import json
from pathlib import Path
import time
import numpy as np
import torch
from .adaptive_census import srf_centroid
from .data import NativeOpticalDataset,sha256
from .geometry_views import common_metric_view


def area_projection(source,target,*,dx,dy):
    if not math.isfinite(dx) or not math.isfinite(dy):raise ValueError("Finite diagnostic offsets required")
    source.validate();target.validate()
    if source.crs!=target.crs:raise ValueError('Explicit common metric coordinate view required')
    sh,sw=source.reflectance.shape;th,tw=target.reflectance.shape
    a,_,left,_,e,top=source.affine;ta,_,tl,_,te,tt=target.affine
    sx=left+np.arange(sw)*a+dx;sy=top+np.arange(sh)*e+dy
    tx=tl+np.arange(tw)*ta;ty=tt+np.arange(th)*te
    wx=np.maximum(0,np.minimum(tx[:,None]+ta,sx[None]+a)-np.maximum(tx[:,None],sx[None]))
    wy=np.maximum(0,np.minimum(ty[:,None],sy[None])-np.maximum(ty[:,None]+te,sy[None]+e))
    keep=source.keep().detach().cpu().numpy();values=source.reflectance.detach().cpu().double().numpy()
    if not np.isfinite(values[keep]).all():raise ValueError('Nonfinite visible source')
    support=wy@keep.astype(np.float64)@wx.T
    numerator=wy@np.where(keep,values,0.)@wx.T
    return np.divide(numerator,support,out=np.full_like(numerator,np.nan),where=support>0),support


def registration_score(projected,area,target):
    truth=target.reflectance.detach().cpu().double().numpy();valid=target.keep().detach().cpu().numpy()&(area>0)
    if not np.isfinite(truth[valid]).all():raise ValueError('Nonfinite visible target')
    x,y,w=projected[valid],truth[valid],area[valid]
    if not len(w):return dict(pixels=0,support_m2=0.,correlation=None,rmse=None,bias=None,source_std=None,target_std=None)
    mass=float(w.sum());w=w/mass;xc=x-x[0];yc=y-y[0];xc=xc-float(w@xc);yc=yc-float(w@yc);vx=float(w@(xc**2));vy=float(w@(yc**2))
    return dict(pixels=len(w),support_m2=mass,correlation=float(w@(xc*yc)/np.sqrt(vx*vy)) if vx>0 and vy>0 else None,rmse=float(np.sqrt(w@((x-y)**2))),bias=float(w@(x-y)),source_std=float(np.sqrt(vx)),target_std=float(np.sqrt(vy)))


def run(args):
    torch.set_num_threads(2);start=time.perf_counter();study=json.loads(args.study.read_text(encoding='utf-8'));catalog=json.loads(args.catalog.read_text(encoding='utf-8'))
    pair=next(p for p in catalog['pairs'] if p['S2_id']==args.source_id and p['Landsat_id']==args.target_id)
    if pair['role']!='TRAIN' or pair['geometry_status']=='common_projection_operator_required':raise ValueError('TRAIN same-projection diagnostic pair required')
    ds=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
    if ds.identities!=study['source_identities'] or catalog['members_sha256']!=ds.identities['members']:raise ValueError('Source catalog differs')
    lookup={row['id']:i for i,row in enumerate(ds.rows)};source=ds[lookup[args.source_id]];target=common_metric_view(ds[lookup[args.target_id]],source.bands[0].crs)
    by_source={b.record_id:b for b in source.bands};by_target={b.record_id:b for b in target.bands};rows=[]
    for names in args.band_pair:
        sa,tb=names.split(':');a,b=by_source[sa],by_target[tb];values=[]
        for dx,dy in product(args.offsets_m,repeat=2):
            pred,area=area_projection(a,b,dx=dx,dy=dy);values.append(dict(dx_m=dx,dy_m=dy,**registration_score(pred,area,b)))
        scored=[v for v in values if v['correlation'] is not None];best=max(scored,key=lambda row:row['correlation']) if scored else None
        rows.append(dict(source_band=sa,target_band=tb,source_srf_centroid_nm=srf_centroid(a),target_srf_centroid_nm=srf_centroid(b),offsets=values,best_measured_correlation=best))
    for obs in (source,target):
        if sha256(obs.metadata['cache'])!=obs.metadata['cache_sha256']:raise ValueError('Cache changed during registration measurement')
    report=dict(status='REGISTRATION_DIAGNOSTIC_NOT_ADMISSION',source_id=source.identifier,target_id=target.identifier,role='TRAIN',source_identities=ds.identities,pair=pair,source_cache_sha256=source.metadata['cache_sha256'],target_cache_sha256=target.metadata['cache_sha256'],source_product=source.metadata['product'],target_product=target.metadata['product'],source_processing_domain=source.metadata.get('processing_domain'),target_processing_domain=target.metadata.get('processing_domain'),source_calibration_revision=source.metadata.get('calibration_revision'),target_calibration_revision=target.metadata.get('calibration_revision'),source_QA_provenance=source.metadata.get('QA_provenance',source.metadata.get('provenance')),target_QA_provenance=target.metadata.get('QA_provenance',target.metadata.get('provenance')),metric='Overlap-area-weighted Pearson correlation, RMSE and bias; partial valid coverage retained with its true area',offset_sign='Translate source native cell coordinates by (dx,dy); values and input files unchanged',offsets_m=args.offsets_m,band_pairs=rows,optimizer_updates=0,admitted_pairs=0,seconds=time.perf_counter()-start,source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('adaptive_census.py')]},limitations=['Existing release QA is engineering-only; parent QA not independently rederived','Diagnostic area aggregation is not a new resampled training input or grid','Spectral response differences and real surface changes can affect correlation; a best offset alone is not registration proof','No temporal, registration, change-risk or calibration threshold frozen','One explicit candidate pair only; no nearest-only teacher or training-pair policy adopted'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(seconds=report['seconds'],admitted_pairs=0,band_pairs=[dict(source_band=x['source_band'],target_band=x['target_band'],best=x['best_measured_correlation']) for x in rows]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--source-id',required=True);p.add_argument('--target-id',required=True);p.add_argument('--band-pair',action='append',required=True);p.add_argument('--offsets-m',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

"""Read-only paired reflectance/QA risk evidence; never admission or training masks."""
import json,time
from pathlib import Path
from itertools import product
from collections import defaultdict
import numpy as np
import torch
import rasterio
import shapely
from shapely.affinity import translate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from .data import NativeOpticalDataset,sha256
from .geometry_views import common_metric_view,northing_shift
from .pair_registration import area_projection,registration_score
from .pair_support_audit import mask_region
from .adaptive_census import srf_centroid


def fixed_candidate_comparison(source,target,offsets_m,candidate):
    offsets=list(product(offsets_m,repeat=2));candidate=tuple(candidate)
    if (0.,0.) not in offsets or candidate not in offsets:raise ValueError('Explicit candidate and zero must be on existing diagnostic grid')
    pixel_area=target.affine[0]*(-target.affine[4]);full=target.keep().cpu().numpy().copy();selected={}
    for dx,dy in offsets:
        prediction,area=area_projection(source,target,dx=dx,dy=dy);full &= np.isclose(area,pixel_area,rtol=1e-10,atol=0)
        if (dx,dy) in ((0.,0.),candidate):selected[(dx,dy)]=prediction
    weights=np.where(full,pixel_area,0.);truth=target.reflectance.detach().cpu().double().numpy();reports=[]
    for delta in ((0.,0.),candidate):
        prediction=selected[delta];error=prediction[full]-truth[full];score=registration_score(prediction,weights,target)
        reports.append(dict(translation_m=delta,**score,absolute_error_quantiles=None if not len(error) else dict(zip(('p50','p90','p95','p99'),map(float,np.quantile(abs(error),(.5,.9,.95,.99)))))))
    return dict(fixed_pixels=int(full.sum()),zero=reports[0],candidate=reports[1]),selected[(0.,0.)],selected[candidate],truth,full


def categorical_area_profile(values,affine,qa_valid,valid_region):
    rows=[]
    for value in np.unique(values[qa_valid]):
        region=mask_region(qa_valid&(values==value),affine);area=float(region.intersection(valid_region).area)
        if area:rows.append(dict(value=int(value),valid_overlap_m2=area))
    denominator=float(valid_region.area);covered=sum(r['valid_overlap_m2'] for r in rows)
    return dict(reference_valid_area_m2=denominator,QA_covered_area_m2=covered,uncovered_area_m2=max(0.,denominator-covered),values=rows)


def band_summary(band):
    values=band.reflectance.detach().cpu().double().numpy()[band.keep().cpu().numpy()]
    return dict(band=band.record_id,native_shape=list(band.reflectance.shape),gsd_xy_m=[band.affine[0],-band.affine[4]],valid_pixels=len(values),minimum=None if not len(values) else float(values.min()),maximum=None if not len(values) else float(values.max()),quantiles=None if not len(values) else list(map(float,np.quantile(values,(.01,.5,.99)))),negative_pixels=int((values<0).sum()),zero_pixels=int((values==0).sum()),above_one_pixels=int((values>1).sum()))


def main():
    root=Path(__file__).resolve().parent;r=root/'reports';output=r/'pair_quality_risk.json'
    if output.exists():raise ValueError('Preserve existing quality audit')
    started=time.perf_counter();torch.set_num_threads(2);proposal=json.loads((r/'B2_pair_geometry_proposal.json').read_text(encoding='utf-8'));study=json.loads((r/'pilot_study_candidate.json').read_text(encoding='utf-8'));lineage=json.loads((r/'pair_full_QA_lineage.json').read_text());fine=json.loads((r/'pair_fixed_support_fine_grid.json').read_text());ds=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
    if ds.identities!=proposal['source_identities'] or ds.identities!=study['source_identities']:raise ValueError('Frozen diagnostic source differs')
    by_id={x['id']:i for i,x in enumerate(ds.rows)};qa_by_id={x['id']:x for x in lineage['rows']};pairs=[];members={};font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');fig,axes=plt.subplots(3,4,figsize=(15,11),layout='constrained');image=None
    for ordinal,case in enumerate(proposal['candidate']['cases']):
        a=ds[by_id[case['S2_id']]];b=common_metric_view(ds[by_id[case['Landsat_id']]],a.bands[0].crs)
        if a.metadata['cache_sha256']!=case['S2_cache_sha256'] or b.metadata['cache_sha256']!=case['Landsat_cache_sha256'] or case['role']!='TRAIN':raise ValueError('Candidate identity or role changed')
        band_pairs=(('B03','SR_B3'),('B04','SR_B4'),('B8A','SR_B5'),('B11','SR_B6'));aa={x.record_id:x for x in a.bands};bb={x.record_id:x for x in b.bands};delta=case['candidate']['S2_relative_to_Landsat_translation_m'];directions=[];display={}
        for reverse in (False,True):
            bands=[];shift=[-v for v in delta] if reverse else delta
            for source_key,target_key in band_pairs:
                source,target=(bb[target_key],aa[source_key]) if reverse else (aa[source_key],bb[target_key]);stats,p0,pc,truth,full=fixed_candidate_comparison(source,target,fine['offsets_m'],shift)
                bands.append(dict(source_band=source.record_id,target_band=target.record_id,source_srf_centroid_nm=srf_centroid(source),target_srf_centroid_nm=srf_centroid(target),**stats))
                if not reverse:display[source_key]=(p0,pc,truth,full)
            directions.append(dict(source_id=b.identifier if reverse else a.identifier,target_id=a.identifier if reverse else b.identifier,source_shift_m=shift,bands=bands))
        # Diagnostic area-averaging only; original inputs and training grids stay native.
        mask=np.logical_and.reduce([display[k][3] for k in ('B03','B04','B8A','B11')]);colors=('B8A','B04','B03');pictures=[np.stack([display[k][index] for k in colors],-1) for index in (2,0,1)];extent=(0,bb['SR_B3'].reflectance.shape[1]*30,0,bb['SR_B3'].reflectance.shape[0]*30)
        for col,(picture,label) in enumerate(zip(pictures,('Landsat原生目标','S2面积聚合：零偏移','S2面积聚合：候选偏移'),strict=True)):
            axes[ordinal,col].imshow(np.where(mask[...,None],np.clip(picture/.3,0,1),.75),origin='upper',extent=extent,interpolation='nearest');axes[ordinal,col].set_title(label,fontproperties=font,fontsize=10)
        residual=np.mean(np.stack([abs(display[k][1]-display[k][2]) for k in ('B03','B04','B8A','B11')]),0);image=axes[ordinal,3].imshow(np.where(mask,residual,np.nan),origin='upper',extent=extent,cmap='magma',vmin=0,vmax=.08,interpolation='nearest');axes[ordinal,3].set_title('四波段平均绝对差',fontproperties=font,fontsize=10)
        for ax in axes[ordinal]:ax.set_aspect('equal');ax.set_xlabel('m');ax.set_ylabel(f"{case['time_gap_seconds']/60:.2f} min")
        pairs.append(dict(S2_id=a.identifier,Landsat_id=b.identifier,time_gap_seconds=case['time_gap_seconds'],candidate_delta_m=delta,directions=directions,display_common_target_pixels=int(mask.sum())))
        for obs,green in ((a,'B03'),(b,'SR_B3')):
            qa=qa_by_id[obs.identifier]
            if not qa['raw_QA_independently_recomputed']:raise ValueError('Only independently reconstructed raw-QA cases audited here')
            band=next(x for x in obs.bands if x.record_id==green);region=mask_region(band.valid_mask.numpy(),band.affine);headers=[]
            for header in qa['qa_headers']:
                path=Path(header['path'])
                if sha256(path)!=header['sha256']:raise ValueError('Raw QA changed')
                with rasterio.open(path) as src:
                    values=src.read(1);valid=src.dataset_mask()>0;affine=tuple(src.transform)[:6];shift_y=northing_shift(src.crs.to_string(),band.crs)
                if shift_y:affine=(*affine[:5],affine[5]+shift_y)
                profile=categorical_area_profile(values,affine,valid,region);profile.update(path=str(path),sha256=header['sha256'],native_raw_value_histogram=header['raw_value_histogram'])
                if sha256(path)!=header['sha256']:raise ValueError('Raw QA changed during read')
                name=path.name.lower();denominator=profile['reference_valid_area_m2'];fractions=defaultdict(float)
                for entry in profile['values']:
                    value=entry['value'];area=entry['valid_overlap_m2']
                    if name in ('scl.tif','scl.tiff'):
                        if value in (2,7):fractions['SCL_2_or_7']+=area/denominator
                    if name=='qa_aerosol.tif':
                        fractions['aerosol_level_'+str((value>>6)&3)]+=area/denominator
                        if value&2:fractions['retrieval_valid']+=area/denominator
                        if value&32:fractions['interpolated']+=area/denominator
                profile['reference_valid_area_fractions']=dict(fractions);headers.append(profile)
            members[obs.identifier]=dict(id=obs.identifier,cache_sha256=obs.metadata['cache_sha256'],processing_domain=obs.metadata.get('processing_domain'),calibration_revision=obs.metadata.get('calibration_revision'),processing_baseline=obs.metadata['physical_contract'].get('processing_baseline'),reference_band_for_QA_area=green,QA_profiles=headers,band_value_summaries=[band_summary(x) for x in obs.bands])
            if sha256(obs.metadata['cache'])!=obs.metadata['cache_sha256']:raise ValueError('Native cache changed')
    fig.colorbar(image,ax=axes[:,3].tolist(),shrink=.75,label='Reflectance absolute difference');fig.suptitle('Diagnostic false colour: NIR / red / green; identical support; gray=excluded; display0–0.3',fontsize=11);figure=r/'pair_quality_risk_maps.png';fig.savefig(figure,dpi=150);plt.close(fig)
    report=dict(status='TRAIN_PAIR_QUALITY_RISK_EVIDENCE_NOT_ADMISSION',source_identities=ds.identities,pairs=pairs,members=list(members.values()),optimizer_updates=0,real_model_forward_passes=0,training_coordinate_corrections=0,admitted_pairs=0,training_masks_changed=False,display_only=dict(false_colour_channels='NIR/red/green',reflectance_range=[0.,.3],difference_range=[0.,.08]),source_reports={name:sha256(r/name) for name in ('B2_pair_geometry_proposal.json','pair_full_QA_lineage.json','pair_fixed_support_fine_grid.json')},source_sha256={name:sha256(root/name) for name in ('pair_quality_review.py','pair_registration.py','pair_support_audit.py','audit_lineage.py')},figure_sha256=sha256(figure),seconds=time.perf_counter()-started,limits=['Candidate shifts are offline audit hypotheses, not approved query-frame records or applied training corrections','Area projection for diagnostic plots does not replace original reflectance grids','All raw QA/reflectance values preserved; described alternatives do not change the mask','Differences confound SRF, atmosphere, view geometry, registration and real surface change; not an unchanged-surface proof','QA area profiles use each observation green-band valid footprint, not whole-machine policy acceptance','No VAL/protected pixels, learned model, optimizer or new pairing selection used'])
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],pairs=len(pairs),members=len(members),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()

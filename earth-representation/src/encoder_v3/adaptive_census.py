"""Count exact adaptive windows without constructing learned tensors or cell lists."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import torch
from .adaptive_window import window_scale
from .data import NativeOpticalDataset,sha256
from .geometry_views import common_metric_view


def band_census(band,*,exponent,origin_xy):
    keep=band.keep().detach().cpu().numpy();rows,cols=np.nonzero(keep)
    gx,_,left,_,e,top=band.affine;gy=-e
    window=np.array([gx*window_scale(gx,exponent),gy*window_scale(gy,exponent)])
    centers=np.column_stack((left+(cols+.5)*gx,top+(rows+.5)*e))
    ids=np.floor((centers-np.asarray(origin_xy))/window).astype(np.int64)
    _,counts=np.unique(ids,axis=0,return_counts=True)
    return dict(band=band.record_id,gsd_m=[gx,gy],native_shape=list(keep.shape),visible_pixels=int(keep.sum()),tokens=len(counts),local_sequence_max=int(counts.max()) if len(counts) else 0,window_m=window.tolist())


def srf_centroid(band):
    wave=band.wavelengths_nm.detach().cpu().double().numpy();response=band.response.detach().cpu().double().numpy()
    if len(wave)<2 or np.any(np.diff(wave)<=0) or not np.isfinite(wave).all() or not np.isfinite(response).all() or np.any(response<0):raise ValueError('Invalid true SRF')
    area=np.trapezoid(response,wave)
    if area<=0:raise ValueError('Positive true SRF area required')
    return float(np.trapezoid(wave*response,wave)/area)


def spectral_panel(bands):
    """All singles/pairs; cyclic spectral runs cover every remaining hidden count."""
    ordered=sorted([(srf_centroid(b),b.record_id) for b in bands]);names=[name for _,name in ordered];c=len(names)
    if c<2 or len(set(names))!=c:raise ValueError('B1 requires distinct real band identities')
    hidden={tuple([name]) for name in names}
    hidden.update(tuple(sorted(pair)) for pair in combinations(names,2) if c>2)
    for size in range(3,c):
        for start in range(c):hidden.add(tuple(sorted(names[(start+j)%c] for j in range(size))))
    cases=sorted(hidden,key=lambda group:(len(group),group))
    assert all(0<len(group)<c for group in cases)
    return dict(status='DIAGNOSTIC_PANEL_NOT_FORMAL_SCHEDULER',spectral_order=[dict(band=name,srf_centroid_nm=center) for center,name in ordered],hidden_band_combinations=[list(group) for group in cases],counts_by_hidden_number=dict(Counter(map(len,cases))),rule='All singles and pairs; all cyclic runs in true-SRF centroid order for hidden counts 3..C-1; deduplicated',training_sampling_changed=False)


def run(args):
    study=json.loads(args.study.read_text(encoding='utf-8'));datasets={role:NativeOpticalDataset(study['release'],study['contracts'],role=role) for role in ('TRAIN','VAL')};rows=[];panels=[]
    for role,key in [('TRAIN','training_members'),('VAL','validation_members')]:
        ds=datasets[role]
        if ds.identities!=study['source_identities']:raise ValueError('Source identity differs from reviewed study')
        indices={r['id']:i for i,r in enumerate(ds.rows)}
        for frozen in study[key]:
            obs=ds[indices[frozen['id']]]
            if obs.metadata['cache_sha256']!=frozen['cache_sha256']:raise ValueError('Frozen cache differs')
            obs=common_metric_view(obs,obs.bands[0].crs)
            origin=(min(b.affine[2] for b in obs.bands),min(b.affine[5]+b.affine[4]*b.reflectance.shape[0] for b in obs.bands))
            bands=[band_census(b,exponent=args.exponent,origin_xy=origin) for b in obs.bands];n=sum(b['tokens'] for b in bands)
            entry=dict(observation=obs.identifier,role=role,sensor=obs.metadata['sensor'],cache_sha256=obs.metadata['cache_sha256'],origin_xy=origin,bands=bands,tokens_all_visible=n,one_fp32_attention_score_matrix_gib=4*n*n*4/1024**3,score_matrix_formula='4 heads * N * N * 4 bytes; one matrix, one layer, batch 1; not measured peak',exponent_can_change_count=any(min(b['gsd_m'])<10 for b in bands))
            rows.append(entry)
            if role=='VAL':
                panel=spectral_panel(obs.bands);panel['observation']=obs.identifier;panel['cache_sha256']=obs.metadata['cache_sha256'];panel['tokens_by_hidden_combination']=[sum(b['tokens'] for b in bands if b['band'] not in hidden) for hidden in panel['hidden_band_combinations']];panels.append(panel)
    report=dict(status='REAL_REVIEWED_MEMBER_GEOMETRY_CENSUS',source_identities=study['source_identities'],study_sha256=sha256(args.study),exponent_fixture=args.exponent,exponent_frozen=False,optimizer_updates=0,learned_forward_passes=0,observations=rows,validation_panels=panels,limits=['Exact mask/grid-derived token counts; no reflectance model fit or formal data admission','All-visible counts are not masked A or B1 training counts; every panel has its own visible-band count','Score-matrix byte estimates describe a math implementation, not every fused attention backend','Expanded panel is independent of training sampling and is not adopted as the formal validation metric'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(observations=len(rows),min_tokens=min(r['tokens_all_visible'] for r in rows),max_tokens=max(r['tokens_all_visible'] for r in rows),validation_combinations=sum(len(p['hidden_band_combinations']) for p in panels)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--exponent',type=float,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

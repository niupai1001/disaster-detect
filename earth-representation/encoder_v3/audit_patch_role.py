"""Read-only geometry audit of old P and continuous Merge roles; no model forward."""
import hashlib,json,math
from pathlib import Path
from dataclasses import replace
import torch
from .model import BandObservation
from .data import Observation
from .prepare_pilot import rectangle
from .tasks import spatial_task
from .participation_layout import AreaEquivalentHillRate,participation_layout


def main():
    root=Path(__file__).resolve().parent;r=root/'reports';out=r/'physical_patch_role_audit.json'
    if out.exists():raise ValueError('Preserve committed audit')
    confirmed=json.loads((r/'merge_rate_confirmation.json').read_text());curve=confirmed['curve'];rate=AreaEquivalentHillRate(curve['amplitude'],curve['scale_m'],curve['power']);rows=[]
    for requested,gsd in [(60.,10.),(120.,10.),(1440.,10.),(120.,1.21)]:
        n=math.ceil(requested/gsd);actual=n*gsd;band=BandObservation(torch.zeros(n,n),torch.ones(n,n,dtype=torch.bool),torch.tensor([450.,500.,550.]),torch.tensor([0.,1.,0.]),(gsd,0.,0.,0.,-gsd,actual),'EPSG:32631','synthetic_geometry')
        metadata=dict(id='synthetic_window',role='TRAIN');record=dict(id=metadata['id'],bands=[dict(shape=[n,n],transform=list(band.affine))]);before=participation_layout(band,rate_curve=rate,selection_seed=0,origin_xy=(0.,0.));before_tokens=len(before['centers'])
        for patch in (60.,120.):
            box=rectangle(record,patch,seed=0,update=0);task=spatial_task(Observation([band],metadata),box);keep=int(task.visible[0].keep().sum());targets=int(task.targets[0].mask.sum());x0,y0,x1,y1=box;intersection=max(0.,min(actual,x1)-max(0.,x0))*max(0.,min(actual,y1)-max(0.,y0));rows.append(dict(requested_input_span_m=requested,actual_input_span_m=actual,gsd_m=gsd,native_shape=[n,n],mask_grid_P_m=patch,rectangle=box,hidden_native_positions=n*n-keep,visible_native_positions=keep,supervised_target_positions=targets,protection_positions=task.metadata['protection_pixels'],rectangle_fraction_of_actual_window=intersection/(actual*actual),unmasked_Merge_tokens=before_tokens))
    assert all(x['visible_native_positions']==0 for x in rows if x['requested_input_span_m']==60. and x['gsd_m']==10.)
    assert next(x for x in rows if x['requested_input_span_m']==120. and x['gsd_m']==10. and x['mask_grid_P_m']==120.)['visible_native_positions']==0
    assert all(0<x['visible_native_positions']<math.prod(x['native_shape']) for x in rows if x['requested_input_span_m']==1440.)
    for span,gsd in [(60.,10.),(120.,10.),(1440.,10.),(120.,1.21)]:assert len({x['unmasked_Merge_tokens'] for x in rows if x['requested_input_span_m']==span and x['gsd_m']==gsd})==1
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();report=dict(status='PHYSICAL_P_ROLE_GEOMETRY_AUDIT_NOT_A_POLICY_CHANGE',rows=rows,current_behavior='Complete native input windows; P60/120 enters old rectangle grid and geometry-cache context, not continuous Participation window formula',confirmed_Merge='r(g),s=1/(1-r); no P argument',optimizer_updates=0,model_forward_calls=0,real_image_reads=0,source_sha256={n:sha(root/n) for n in ['audit_patch_role.py','prepare_pilot.py','tasks.py','participation_layout.py']},source_reports={n:sha(r/n) for n in ['spatial_mask_preflight_review.json','merge_rate_confirmation.json','pilot_members_panels_review.json']},limits=['One explicit synthetic geometry instance per configuration, not a sampled mask distribution or trained model result','No physical input crop or new masking rule adopted','Original document is reference material; later user-confirmed Merge B and rectangle preflight instructions govern implementation','Formal eligible native-window manifest and mask scope still need freezing; P must not silently become an input crop'])
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],cases=len(rows),examples=[x for x in rows if x['requested_input_span_m'] in (60.,1440.)]),ensure_ascii=False))


if __name__=='__main__':main()

"""Explicit fixed VAL panels and read-only per-case collection; no metric choice."""
import copy,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from .data import sha256
from .gates import require_training
from .joint import PlannedView
from .pair_geometry import approved_payload,reviewed_pair_task,frame_for_task
from .participation_encoder import ParticipationEncoder,merge_recipe,prepare_participation_task
from .plan_prefetch import move_views
from .route_control import frozen_identity,_sha
from .route_step import update_signature
from .tasks import spatial_task,fixed_spectral_task
from .adaptive_streaming import stream_adaptive_task


class FrozenValidationPanels:
    def __init__(self,path,*,config,frozen_ids,dataset,rate_curve,selection_seed):
        require_training(config)
        if frozen_ids.get('formal_config_sha256')!=frozen_identity(config):raise ValueError('Frozen formal configuration differs')
        expected=frozen_ids.get('validation_panel_sha256');_sha(expected);path=Path(path)
        if sha256(path)!=expected:raise ValueError('Frozen validation panel differs')
        doc=json.loads(path.read_text(encoding='utf-8-sig'))
        if sha256(path)!=expected:raise ValueError('Panel changed during read')
        if doc.get('kind')!='frozen_validation_panels_v1' or dataset.role!='VAL' or doc.get('source_identities')!=dataset.identities or dataset.identities!=frozen_ids.get('source_identities'):raise ValueError('Exact frozen VAL release required')
        if merge_recipe(rate_curve,selection_seed)!=frozen_ids.get('merge_recipe'):raise ValueError('Merge recipe differs')
        eligible=doc.get('eligible_members');panels=doc.get('panels');pairs=doc.get('pair_admissions')
        if not isinstance(eligible,dict) or not eligible or not isinstance(panels,list) or not panels or not isinstance(pairs,dict):raise ValueError('Explicit panel members/cases/pairs required')
        rows={r['id']:r for r in dataset.rows}
        if len(rows)!=len(dataset.rows):raise ValueError('Duplicate VAL member')
        for identifier,version in eligible.items():
            _sha(version)
            if identifier not in rows or rows[identifier]['role']!='VAL' or rows[identifier]['cache_sha256']!=version:raise ValueError('Panel member role/version differs')
        ids=[p['panel_id'] for p in panels]
        if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):raise ValueError('Unique panel identifiers required')
        for case in panels:
            if case['source_id'] not in eligible or case['kind'] not in ('A','B1','B2') or case['P_m'] not in (60.,120.):raise ValueError('Invalid explicit validation case')
            origin=case['origin_xy']
            if len(origin)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in origin):raise ValueError('Finite native origin required')
            if case['kind']=='A':
                box=case['rectangle']
                if len(box)!=4 or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in box) or box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Explicit fixed rectangle required')
            elif case['kind']=='B1':
                hidden=case['hidden_band_ids'];bands=[b['band_key'] for b in rows[case['source_id']]['bands']]
                if not hidden or len(set(hidden))!=len(hidden) or not set(hidden)<set(bands):raise ValueError('Fixed spectral panel must leave visible bands')
            elif case['pair_id'] not in pairs or pairs[case['pair_id']]['source_id']!=case['source_id']:raise ValueError('Fixed B2 case lacks its own pair')
        for pair in pairs.values():
            checks=('space','registration','time_difference','surface_change','qa','srf','role_isolation')
            if pair.get('source_id') not in eligible or pair.get('target_id') not in eligible or pair['source_id']==pair['target_id'] or any(pair.get('checks',{}).get(k) is not True for k in checks):raise ValueError('Independent VAL pair admission required')
            family={'S2':'S2','L8':'Landsat','L9':'Landsat','Landsat':'Landsat'}
            if {family.get(rows[pair['source_id']].get('sensor')),family.get(rows[pair['target_id']].get('sensor'))}!={'S2','Landsat'}:raise ValueError('Fixed B2 panel must be cross-sensor')
            if pair.get('pair_geometry') is not None:approved_payload(pair['pair_geometry'])
        chunk=config.get('formal_step',{}).get('chunk_pixels')
        if type(chunk) is not int or chunk<1:raise ValueError('Explicit Decoder chunk required')
        self._doc=doc;self._dataset=dataset;self._indices={row['id']:i for i,row in enumerate(dataset.rows)};self._rows=rows;self._curve=rate_curve;self._seed=selection_seed;self._frozen=copy.deepcopy(frozen_ids);self.chunk=chunk;self.identity=expected;self._path=path;st=path.stat();self._stat=(st.st_size,st.st_mtime_ns)

    def iter_views(self,*,cell_indexed,retain_layouts):
        st=self._path.stat()
        if (st.st_size,st.st_mtime_ns)!=self._stat:raise ValueError('Frozen panel file changed')
        if type(cell_indexed) is not bool or type(retain_layouts) is not bool:raise ValueError('Explicit geometry execution choices required')
        cache={};stats={}
        def read(identifier):
            if identifier not in cache:
                obs=self._dataset[self._indices[identifier]]
                if obs.identifier!=identifier or obs.metadata.get('role')!='VAL' or obs.metadata.get('cache_sha256')!=self._doc['eligible_members'][identifier] or obs.metadata.get('frozen_source_ids')!=self._dataset.identities or [b.record_id for b in obs.bands]!=[b['band_key'] for b in self._rows[identifier]['bands']] or any(b.reflectance.device.type!='cpu' for b in obs.bands):raise ValueError('Materialized VAL identity/bands/CPU placement differs')
                cache[identifier]=obs
            return cache[identifier]
        for case in self._doc['panels']:
            source=read(case['source_id']);target=source
            if case['kind']=='A':task=spatial_task(source,case['rectangle'])
            elif case['kind']=='B1':task=fixed_spectral_task(source,case['hidden_band_ids'])
            else:
                pair=self._doc['pair_admissions'][case['pair_id']];target=read(pair['target_id']);task=reviewed_pair_task(source,target,pair);frame=frame_for_task(task)
                if frame is not None and frame.record_sha256 not in self._frozen.get('pair_geometry_records',[]):raise ValueError('VAL pair geometry not frozen for this run')
            if target.identifier not in stats:stats[target.identifier]=self._dataset.engineering_train_stds(target)
            stds=stats[target.identifier]
            if any(t.band.record_id not in stds or not math.isfinite(stds[t.band.record_id]) or stds[t.band.record_id]<=0 for t in task.targets):raise ValueError('Frozen TRAIN target statistics required, no VAL fitting')
            plan=prepare_participation_task(task,rate_curve=self._curve,selection_seed=self._seed,origin_xy=case['origin_xy'],chunk_pixels=self.chunk,cell_indexed=cell_indexed,retain_layouts=retain_layouts,physical_patch_m=case['P_m'])
            yield case['panel_id'],PlannedView(task,plan,stds,source.metadata['sensor'])

    def collect(self,encoder,decoder,*,cell_indexed,retain_layouts):
        if not isinstance(encoder,ParticipationEncoder) or encoder.merge_recipe!=self._frozen['merge_recipe']:raise ValueError('Frozen Participation encoder required')
        if any(p.is_floating_point() and p.dtype!=torch.float32 for module in (encoder,decoder) for p in module.parameters()):raise ValueError('FP32 validation requires FP32 learned parameters')
        devices={p.device for module in (encoder,decoder) for p in module.parameters()}
        if len(devices)!=1:raise ValueError('Explicit common model device required')
        device=next(iter(devices));parts={id(m):m for root in (encoder,decoder) for m in root.modules()};flags={i:m.training for i,m in parts.items()};cpu=torch.get_rng_state();python=random.getstate();numpy=np.random.get_state();cuda={device.index:torch.cuda.get_rng_state(device)} if device.type=='cuda' else {};rows=[];start=time.perf_counter()
        try:
            encoder.eval();decoder.eval()
            with torch.no_grad():
                for label,view in self.iter_views(cell_indexed=cell_indexed,retain_layouts=retain_layouts):
                    signature=update_signature([view]);moved=move_views([view],device)[0];result=stream_adaptive_task(encoder,decoder,moved.task,moved.plan,train_stds=moved.train_stds,chunk_pixels=self.chunk,loss_weight=1.,backward=False,precision='fp32');rows.append(dict(panel_id=label,observation=view.task.metadata['id'],kind=view.task.kind,source=view.loss_source,exposure_signature=signature,result=result))
            if device.type=='cuda':torch.cuda.synchronize(device)
            return dict(status='FIXED_PANEL_RESULTS_COLLECTED',panel_sha256=self.identity,rows=rows,seconds_wall=time.perf_counter()-start,metric=None,metric_status='REQUIRES_REVIEWED_AGGREGATION',optimizer_updates=0)
        finally:
            for i,m in parts.items():m.training=flags[i]
            random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
            for index,state in cuda.items():torch.cuda.set_rng_state(state,device=index)

"""Hash-bound, CPU-only A/B task materialization; never samples or trains."""
import copy,json,math
from pathlib import Path
from .data import sha256
from .gates import require_training
from .joint import PlannedView
from .pair_geometry import approved_payload,reviewed_pair_task
from .participation_encoder import merge_recipe,prepare_participation_task
from .route_control import frozen_identity,_sha
from .route_step import update_signature,validate_frozen_task_source
from .tasks import Task,spatial_task,fixed_spectral_task


class FrozenABSchedule:
    def __init__(self,path,*,config,frozen_ids,dataset,rate_curve,selection_seed):
        require_training(config)
        if frozen_ids.get('formal_config_sha256')!=frozen_identity(config):raise ValueError('Frozen formal configuration differs')
        path=Path(path);expected=frozen_ids.get('AB_schedule_sha256');_sha(expected)
        if sha256(path)!=expected:raise ValueError('Frozen schedule file differs')
        document=json.loads(path.read_text(encoding='utf-8-sig'))
        if sha256(path)!=expected:raise ValueError('Schedule changed during read')
        if document.get('kind')!='frozen_AB_schedule_v1' or dataset.role!='TRAIN' or dataset.identities!=frozen_ids.get('source_identities') or document.get('source_identities')!=dataset.identities:raise ValueError('Exact frozen TRAIN release and schedule required')
        if merge_recipe(rate_curve,selection_seed)!=frozen_ids.get('merge_recipe'):raise ValueError('Frozen Merge recipe differs')
        eligible=document.get('eligible_members');entries=document.get('updates');pairs=document.get('pair_admissions')
        if not isinstance(eligible,dict) or not eligible or not isinstance(entries,list) or not 1<=len(entries)<=10000 or not isinstance(pairs,dict):raise ValueError('Complete eligible members, bounded schedule and pair table required')
        rows={r['id']:r for r in dataset.rows}
        if len(rows)!=len(dataset.rows):raise ValueError('Duplicate dataset identities')
        for identifier,version in eligible.items():
            _sha(version);row=rows.get(identifier)
            if row is None or row['role']!='TRAIN' or row['cache_sha256']!=version:raise ValueError('Eligible member version/role differs')
        for admission in pairs.values():
            if admission.get('source_id') not in eligible or admission.get('target_id') not in eligible or admission['source_id']==admission['target_id'] or any(admission.get('checks',{}).get(k) is not True for k in ('space','registration','time_difference','surface_change','qa','srf','role_isolation')):raise ValueError('Only frozen admitted pairs belong in the pair table')
            families={'S2':'S2','L8':'Landsat','L9':'Landsat','Landsat':'Landsat'}
            if {families.get(rows[admission['source_id']].get('sensor')),families.get(rows[admission['target_id']].get('sensor'))}!={'S2','Landsat'}:raise ValueError('B2 requires the frozen S2/Landsat cross-sensor pair')
            if admission.get('pair_geometry') is not None:approved_payload(admission['pair_geometry'])
        for index,entry in enumerate(entries):
            slots=entry.get('slots',[])
            if entry.get('logical_update')!=index or type(entry['logical_update']) is not int or entry.get('P_m') not in (60.,120.) or len(slots)!=8 or len({s['source_id'] for s in slots})!=8:raise ValueError('Sequential schedule and exactly eight distinct input members required')
            for slot in slots:
                identifier=slot['source_id'];B=slot['B'];kind=B.get('kind')
                if identifier not in eligible or kind not in ('B1','B2','B1_ABSENT','B2_ABSENT'):raise ValueError('Unknown member or task slot')
                origin=slot['origin_xy'];box=slot['A_rectangle']
                if len(origin)!=2 or len(box)!=4 or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in list(origin)+list(box)) or box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Explicit finite origin and rectangle required')
                bands=[b['band_key'] for b in rows[identifier]['bands']]
                if kind=='B1':
                    hidden=B.get('hidden_band_ids',[])
                    if not hidden or len(set(hidden))!=len(hidden) or not set(hidden)<set(bands):raise ValueError('Explicit hidden bands must leave visible support')
                elif kind=='B1_ABSENT':
                    if len(bands)!=1:raise ValueError('False NO_HIDEABLE_BAND declaration')
                elif kind=='B2':
                    admission=pairs.get(B.get('pair_id'))
                    if admission is None or admission['source_id']!=identifier:raise ValueError('Slot does not reference its frozen admitted pair')
                elif any(a['source_id']==identifier for a in pairs.values()):raise ValueError('False NO_ADMITTED_PAIR declaration')
                if kind.endswith('_ABSENT') and set(B)!={'kind'}:raise ValueError('Absent slot cannot hide targets or substitute a task')
            if pairs and sum(s['B']['kind'] in ('B1','B1_ABSENT') for s in slots)!=4:raise ValueError('When qualified pairs exist, B1/B2 slots remain half each')
        if pairs and not any(s['B']['kind']=='B2' for e in entries for s in e['slots']):raise ValueError('Admitted pairs cannot have zero B2 exposure in the full schedule')
        chunk=config.get('formal_step',{}).get('chunk_pixels')
        if type(chunk) is not int or chunk<1:raise ValueError('Explicit frozen Decoder chunk required')
        self._document=document;self._rows=rows;self._indices={r['id']:i for i,r in enumerate(dataset.rows)};self.dataset=dataset;self.rate_curve=rate_curve;self.selection_seed=selection_seed;self.chunk=chunk;self.frozen_ids=copy.deepcopy(frozen_ids);self.path=path;self.identity=expected;stat=path.stat();self._file_stat=(stat.st_size,stat.st_mtime_ns)

    def prepare(self,index,*,cell_indexed,retain_layouts):
        if type(index) is not int or not 0<=index<len(self._document['updates']):raise ValueError('Logical update outside frozen schedule')
        if type(cell_indexed) is not bool or type(retain_layouts) is not bool:raise ValueError('Explicit geometry execution choices required')
        stat=self.path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=self._file_stat:raise ValueError('Frozen schedule file changed')
        entry=self._document['updates'][index];loaded={};statistics={};views=[];absences=[]
        def read(identifier):
            if identifier not in loaded:
                obs=self.dataset[self._indices[identifier]];row=self._rows[identifier]
                if obs.identifier!=identifier or obs.metadata.get('role')!='TRAIN' or obs.metadata.get('cache_sha256')!=self._document['eligible_members'][identifier] or obs.metadata.get('frozen_source_ids')!=self.dataset.identities or [b.record_id for b in obs.bands]!=[b['band_key'] for b in row['bands']] or any(b.reflectance.device.type!='cpu' for b in obs.bands):raise ValueError('Materialized native member version, role, bands or CPU placement differs')
                loaded[identifier]=obs
            return loaded[identifier]
        def stds_for(obs):
            if obs.identifier not in statistics:statistics[obs.identifier]=self.dataset.engineering_train_stds(obs)
            return statistics[obs.identifier]
        for slot in entry['slots']:
            source=read(slot['source_id']);B=slot['B'];kind=B['kind'];tasks=[(spatial_task(source,slot['A_rectangle']),source,None)]
            if kind=='B1':tasks.append((fixed_spectral_task(source,B['hidden_band_ids']),source,None))
            elif kind=='B2':
                admission=self._document['pair_admissions'][B['pair_id']];target=read(admission['target_id']);tasks.append((reviewed_pair_task(source,target,admission),target,None))
            else:
                reason='NO_HIDEABLE_BAND' if kind=='B1_ABSENT' else 'NO_ADMITTED_PAIR';task=Task('B1' if kind=='B1_ABSENT' else 'B2',source.bands,[],dict(source.metadata));tasks.append((task,None,reason));absences.append(dict(source_id=source.identifier,kind=task.kind,reason=reason))
            for task,target,reason in tasks:
                validate_frozen_task_source(task,self.frozen_ids)
                plan=None if reason is not None else prepare_participation_task(task,rate_curve=self.rate_curve,selection_seed=self.selection_seed,origin_xy=slot['origin_xy'],chunk_pixels=self.chunk,cell_indexed=cell_indexed,retain_layouts=retain_layouts,physical_patch_m=entry['P_m'])
                stds={} if target is None else stds_for(target)
                if any(b.band.record_id not in stds or not math.isfinite(stds[b.band.record_id]) or stds[b.band.record_id]<=0 for b in task.targets):raise ValueError('Frozen TRAIN target statistics missing or invalid')
                views.append(PlannedView(task,plan,stds,source.metadata['sensor'],reason))
        return dict(logical_update=index,views=views,AB_input_signature=update_signature(views),source_members=[s['source_id'] for s in entry['slots']],loaded_native_ids=list(loaded),absent_tasks=absences,schedule_sha256=self.identity)

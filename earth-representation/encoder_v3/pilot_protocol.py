"""Hash-bound preflight inputs and explicit execution review; never starts updates."""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from .data import to_device
from .prepare_pilot import member,bounds
from .tasks import spatial_task,spectral_task,fixed_spectral_task
from .streaming import prepare_task,task_signature
from .joint import PlannedView


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_study(path):
    study=json.loads(Path(path).read_text(encoding='utf-8-sig'));content=dict(study);expected=content.pop('candidate_sha256')
    if canonical_sha(content)!=expected:raise ValueError('Pilot candidate content changed')
    if study['effective_batch']!=8 or study['per_candidate_limit']!=200 or study['total_limit']!=400 or study['optimizers']!=['AdamW','CAdamW']:raise ValueError('Pilot budget/comparison differs from plan')
    ids=[r['id'] for r in study['training_members']];val=[r['id'] for r in study['validation_members']]
    if len(ids)!=8 or len(set(ids))!=8 or len(val)!=8 or len(set(val))!=8 or set(ids)&set(val):raise ValueError('Pilot role membership invalid')
    if any(r['role']!=role for role,key in [('TRAIN','training_members'),('VAL','validation_members')] for r in study[key]):raise ValueError('Pilot role changed')
    if {r['release_isolation_group'] for r in study['training_members']}&{r['release_isolation_group'] for r in study['validation_members']}:raise ValueError('Pilot isolation groups overlap')
    if len(study['schedule'])!=200:raise ValueError('Expected bounded 200-entry matched schedule')
    for index,entry in enumerate(study['schedule']):
        if entry['logical_update']!=index or entry['members']!=ids or entry['patch_m'] not in (60.,120.):raise ValueError('Matched schedule identity invalid')
        if [t['observation'] for t in entry['tasks']]!=ids:raise ValueError('Task membership invalid')
    return study


def require_update_review(config,study,root):
    root=Path(root);confirmed=config.get('confirmed_preflight',{})
    if confirmed.get('hypernetwork',{}).get('hidden')!=64:raise ValueError('Confirmed preflight Hypernetwork missing')
    if any(confirmed.get('spatial_mask',{}).get(key)!=value for key,value in study['rectangle_proposal'].items()):raise ValueError('Confirmed mask rule differs')
    path=confirmed.get('members_and_panels',{}).get('review')
    if not path:raise ValueError('Pilot member/panel review missing')
    reviewed=json.loads((root/path).read_text(encoding='utf-8-sig'))
    if reviewed.get('review_type')!='explicit_user_design_confirmation' or reviewed.get('scope')!='pilot_members_and_validation_panels_only' or reviewed.get('candidate_sha256')!=study['candidate_sha256']:raise ValueError('Pilot member/panel review does not bind this candidate')
    if reviewed['members']!={role:[r['id'] for r in study[key]] for role,key in [('TRAIN','training_members'),('VAL','validation_members')]} or reviewed['validation_panels_sha256']!=canonical_sha(study['validation_panels']):raise ValueError('Approved panel/members changed')
    execution=config.get('required_before_pilot',{}).get('execution_review')
    if not execution:raise ValueError('Pilot blocked: execution_review missing (device, precision, resource caps and budgeted update/recovery checks)')
    review=json.loads((root/execution).read_text(encoding='utf-8-sig'))
    if review.get('review_type')!='explicit_user_design_confirmation' or review.get('scope')!='bounded_optimizer_update_and_resume_check' or review.get('candidate_sha256')!=study['candidate_sha256']:raise ValueError('Execution review does not bind this bounded check')
    if review.get('precision')!='fp32' or review.get('steps_per_optimizer')!=4 or review.get('total_steps')!=8 or review.get('reserve_from_pilot_budget') is not True:raise ValueError('Matched check must reserve 4+4 steps from the 400-step pilot cap')
    expected_path=dict(microbatch_observations=1,local_batch_size=64,chunk_pixels=512,native_mode='grouped',layout_mode='indexed',indexed_decoder=True,reference_cell_indexed=False,candidate_cell_indexed=True)
    if review.get('execution_path')!=expected_path:raise ValueError('Reviewed execution path differs from this matched check')
    keys=('peak_lr','warmup_updates','weight_decay','clip_norm','betas','eps')
    if review.get('optimization')!={key:config['candidate'][key] for key in keys}:raise ValueError('Reviewed optimizer conditions changed')
    if not isinstance(review.get('device'),str) or not review['device'].startswith('cuda:'):raise ValueError('Reviewed CUDA device required')
    for key in ('max_wall_seconds','max_allocated_bytes','max_output_bytes'):
        value=review.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Explicit positive resource limit required: '+key)
    return review


def optimizer_groups(modules,weight_decay):
    if not math.isfinite(weight_decay) or weight_decay<0:raise ValueError('Invalid weight decay')
    groups=[dict(params=[],weight_decay=weight_decay),dict(params=[],weight_decay=0.)];seen=set()
    for module in modules:
        normalization={id(p) for layer in module.modules() if isinstance(layer,nn.LayerNorm) for p in layer.parameters(recurse=False)}
        for name,p in module.named_parameters():
            if not p.requires_grad or id(p) in seen:continue
            seen.add(id(p));groups[int(name.endswith('bias') or id(p) in normalization)]['params'].append(p)
    return [group for group in groups if group['params']]


class LinearWarmup:
    def __init__(self,optimizer,*,peak,warmup):
        if not math.isfinite(peak) or peak<=0 or type(warmup) is not int or warmup<1:raise ValueError('Explicit positive warmup required')
        self.optimizer=optimizer;self.peak=peak;self.warmup=warmup;self.last_update=-1
    def apply(self,update):
        if type(update) is not int or update<0:raise ValueError('Invalid logical warmup update')
        for group in self.optimizer.param_groups:group['lr']=self.peak*min((update+1)/self.warmup,1.)
        self.last_update=update
    def state_dict(self):return dict(peak=self.peak,warmup=self.warmup,last_update=self.last_update)
    def load_state_dict(self,state):
        if state['peak']!=self.peak or state['warmup']!=self.warmup:raise ValueError('Warmup configuration changed')
        self.last_update=state['last_update']


def prepare_update(study,dataset,update,device,*,cell_indexed_geometry=False):
    if type(update) is not int or not 0<=update<len(study['schedule']):raise ValueError('Invalid pilot logical update')
    if dataset.role!='TRAIN' or dataset.identities!=study['source_identities']:raise ValueError('Pilot training source identity changed')
    entry=study['schedule'][update];lookup={row['id']:i for i,row in enumerate(dataset.rows)}
    frozen={row['id']:row for row in study['training_members']};views=[]
    for task_entry in entry['tasks']:
        identifier=task_entry['observation'];index=lookup[identifier]
        if member(dataset.rows[index])!=frozen[identifier]:raise ValueError('Pilot native metadata differs from reviewed member')
        observation=dataset[index];stds=dataset.engineering_train_stds(observation)
        left,bottom,_,_=bounds(dataset.rows[index]);origin=(left,bottom);rectangle=task_entry['A_rectangle']
        cpu=(spatial_task(observation,rectangle),spectral_task(observation,seed=study['seed'],update=update))
        if cpu[1].metadata['hidden_band_ids']!=task_entry['B1_hidden']:raise ValueError('Hidden-band schedule differs from reviewed candidate')
        plans=[prepare_task(task,patch_m=entry['patch_m'],origin_xy=origin,chunk_pixels=512,cell_indexed=cell_indexed_geometry) for task in cpu]
        moved=to_device(observation,device)
        tasks=(spatial_task(moved,rectangle),spectral_task(moved,seed=study['seed'],update=update))
        views.extend(PlannedView(task,plan,stds,observation.metadata['sensor']) for task,plan in zip(tasks,plans,strict=True))
    return views,canonical_sha(dict(patch_m=entry['patch_m'],origins=[view.plan.origin_xy if view.plan else None for view in views],tasks=[task_signature(view.task) for view in views]))


def prepare_validation(study,dataset,*,cell_indexed_geometry=False):
    if dataset.role!='VAL' or dataset.identities!=study['source_identities']:raise ValueError('Validation source identity changed')
    lookup={r['id']:i for i,r in enumerate(dataset.rows)};frozen={r['id']:r for r in study['validation_members']};observations={};result=[]
    for panel in study['validation_panels']:
        identifier=panel['observation'];index=lookup[identifier]
        if member(dataset.rows[index])!=frozen[identifier]:raise ValueError('Validation member metadata changed')
        if identifier not in observations:observations[identifier]=dataset[index]
        observation=observations[identifier];left,bottom,_,_=bounds(dataset.rows[index]);stds=dataset.engineering_train_stds(observation)
        tasks=[('A',spatial_task(observation,panel['A_rectangle'])),('B1_single',fixed_spectral_task(observation,panel['B1_single'])),('B1_multi',fixed_spectral_task(observation,panel['B1_multi']))]
        for label,task in tasks:
            plan=prepare_task(task,patch_m=panel['patch_m'],origin_xy=(left,bottom),chunk_pixels=512,cell_indexed=cell_indexed_geometry)
            result.append((label,panel['patch_m'],PlannedView(task,plan,stds,observation.metadata['sensor'])))
    return result

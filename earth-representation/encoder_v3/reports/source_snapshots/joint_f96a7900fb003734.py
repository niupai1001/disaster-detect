"""One effective batch of eight: exact hierarchical A/B gradient accumulation."""
from collections import Counter,defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import math
import hashlib
import torch
from .streaming import stream_task,stream_task_roots,task_signature
from .tasks import curriculum


@dataclass
class PlannedView:
    task: object
    plan: object
    train_stds: dict
    loss_source: str


def active(view):
    return view.plan is not None and any(b.supported_count>0 for b in view.plan.bands)


def layout(views):
    """Task membership and geometric support, never prediction/label values."""
    families={'A':[],'B':[]};members=defaultdict(list)
    for view in views:
        task=view.task
        if task.kind not in ('A','B1','B2') or not view.loss_source:raise ValueError('Explicit task and loss source required')
        if view.plan is not None and view.plan.signature!=task_signature(task):raise ValueError('Task changed after planning')
        if task.kind=='B2':
            admission=task.metadata.get('pair_admission',{})
            checks=('space','registration','time_difference','surface_change','qa','srf','role_isolation')
            if any(admission.get('checks',{}).get(key) is not True for key in checks):raise ValueError('B2 lacks reviewed physical pair admission')
        family='A' if task.kind=='A' else 'B'
        members[task.metadata['id']].append(family)
        if active(view):families[family].append(view)
    if len(members)!=8 or any(sorted(family)!=['A','B'] for family in members.values()):
        raise ValueError('Logical batch requires eight distinct observations, one A and one B per observation')
    if any(not values for values in families.values()):raise ValueError('A and B both require supported supervision')
    weights={}
    for family,values in families.items():
        counts=Counter(view.loss_source for view in values)
        for view in values:weights[id(view)]=1/(len(counts)*counts[view.loss_source])
    return families,weights


def diagnostic_mean(rows):
    sources=defaultdict(list)
    for row in rows:
        if row['result']['loss'] is not None:sources[row['source']].append(row['result']['loss'])
    if not sources:return None
    return sum(sum(values)/len(values) for values in sources.values())/len(sources)


def accumulate_joint(encoder,decoder,views,*,logical_update,chunk_pixels,precision,clip_norm,backward=True,microbatch_observations=1):
    """No optimizer step. Execution microbatches preserve the global batch of eight."""
    if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm<=0):raise ValueError('Invalid gradient clipping candidate')
    if type(microbatch_observations) is not int or microbatch_observations not in (1,2,4,8):raise ValueError('Observation microbatch must be 1,2,4 or 8')
    if precision not in ('fp32','bf16'):raise ValueError('Explicit FP32/BF16 execution mode required')
    if precision=='bf16' and next(encoder.parameters()).device.type!='cuda':raise ValueError('BF16 candidate requires CUDA')
    families,weights=layout(views)
    participation=hasattr(encoder,'merge_recipe')
    if participation and precision!='fp32':raise ValueError('Participation batching has only an FP32 reference')
    if participation and any(view.plan is not None and getattr(view.plan,'merge_recipe',None)!=encoder.merge_recipe for view in views):raise ValueError('Merge recipe differs from supervision plan')
    if participation:
        for view in views:
            if view.plan is not None:view.plan.validate_retained(view.task)
    a,b=curriculum(logical_update);coefficients={'A':a,'B':b}
    if backward:encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
    rows=[];padded_tokens=0;real_tokens=0
    def record(view,result):
        family='A' if view.task.kind=='A' else 'B'
        exposure=dict(rectangle=view.task.rectangle,protection_pixels=view.task.metadata.get('protection_pixels'),hidden_band_ids=view.task.metadata.get('hidden_band_ids'),visible_band_ids=[band.record_id for band in view.task.visible],target_index_sha256={} if view.plan is None else {band.target.band.record_id:hashlib.sha256(band.indices.numpy().tobytes()).hexdigest() for band in view.plan.bands})
        rows.append(dict(kind=view.task.kind,observation=view.task.metadata['id'],source=view.loss_source,coefficient=None if not active(view) else coefficients[family]*weights[id(view)],exposure=exposure,result=result))
    def multiplier(view):return coefficients['A' if view.task.kind=='A' else 'B']*weights[id(view)]
    single_stream=stream_task
    if participation:
        from .adaptive_streaming import stream_adaptive_task
        single_stream=stream_adaptive_task
    try:
        if microbatch_observations==1:
            for view in views:
                result=single_stream(encoder,decoder,view.task,view.plan,train_stds=view.train_stds,chunk_pixels=chunk_pixels,loss_weight=multiplier(view),backward=backward,precision=precision) if active(view) else dict(status='NO_VALID_SUPPORTED_TARGET',loss=None,bands=[])
                record(view,result)
                real_tokens+=result.get('tokens',0);padded_tokens+=result.get('tokens',0)
        else:
            for view in views:
                if not active(view):record(view,dict(status='NO_VALID_SUPPORTED_TARGET',loss=None,bands=[]))
            members=list(dict.fromkeys(view.task.metadata['id'] for view in views))
            for start in range(0,len(members),microbatch_observations):
                chosen=set(members[start:start+microbatch_observations])
                for family in ('A','B'):
                    batch=[view for view in families[family] if view.task.metadata['id'] in chosen]
                    if not batch:continue
                    context=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
                    with torch.set_grad_enabled(backward),context:
                        representations=encoder.forward_batch([(view.task.visible,view.plan.origin_xy) for view in batch] if participation else [(view.task.visible,view.plan.patch_m,view.plan.origin_xy) for view in batch],**({'geometry_plans':[view.plan if view.plan.retained_layouts is not None else None for view in batch]} if participation and any(view.plan.retained_layouts is not None for view in batch) else {}))
                    lengths=[len(r.tokens) for r in representations];real_tokens+=sum(lengths);padded_tokens+=sum(lengths) if participation else len(lengths)*max(lengths)
                    roots=[];gradients=[]
                    for view,representation in zip(batch,representations,strict=True):
                        result,r,g=stream_task_roots(encoder,decoder,view.task,view.plan,train_stds=view.train_stds,chunk_pixels=chunk_pixels,loss_weight=multiplier(view),backward=backward,precision=precision,representation=representation)
                        record(view,result);roots.extend(r);gradients.extend(g)
                    if backward:torch.autograd.backward(roots,gradients)
                    del roots,gradients,representations,representation
        norm=None
        if backward:
            norm=float(torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()),float('inf') if clip_norm is None else clip_norm,error_if_nonfinite=True))
    except Exception:
        if backward:encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
        raise
    family_losses={family:sum(weights[id(view)]*next(row['result']['loss'] for row in rows if row['kind']==view.task.kind and row['observation']==view.task.metadata['id']) for view in values) for family,values in families.items()}
    return dict(optimizer_updates=0,logical_update=logical_update,effective_batch=8,execution_microbatch_observations=microbatch_observations,global_tokens=real_tokens,global_padded_tokens=padded_tokens,global_padding_fraction=1-real_tokens/padded_tokens if padded_tokens else None,planned_views=len(views),active_views=sum(active(view) for view in views),A_weight=a,B_weight=b,A_loss=family_losses['A'],B_loss=family_losses['B'],weighted_loss=a*family_losses['A']+b*family_losses['B'],losses_by_kind={kind:diagnostic_mean([row for row in rows if row['kind']==kind]) for kind in ('A','B1','B2')},gradient_norm_before_clip=norm,clipped=None if norm is None or clip_norm is None else norm>clip_norm,rows=rows)

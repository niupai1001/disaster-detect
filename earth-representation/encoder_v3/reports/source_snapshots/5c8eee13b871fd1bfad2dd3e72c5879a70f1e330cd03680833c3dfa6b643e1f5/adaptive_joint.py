"""Batch-8 A/B + paired JEPA gradient accumulation; no optimizer/EMA step."""
from collections import Counter,defaultdict
from dataclasses import dataclass
import math
import hashlib
import torch
from .adaptive_streaming import stream_adaptive_task
from .joint import layout,active
from .jepa import cross_source_latent_loss
from .tasks import curriculum
from .pair_geometry import reviewed_pair_task
from .streaming import task_signature


@dataclass
class PairedLatentView:
    source: object
    target: object
    admission: dict
    source_encoder_kwargs: dict
    target_encoder_kwargs: dict
    loss_source: str


def accumulate_adaptive_joint(encoder,decoder,views,*,logical_update,chunk_pixels,clip_norm,jepa_views=(),teacher=None,predictor=None,jepa_weight=None,latent_loss_fn=None,curriculum_end_update=None,ab_stable=False,microbatch_observations=1):
    if type(microbatch_observations) is not int or microbatch_observations not in (1,2,4,8):raise ValueError('Observation microbatch must be 1,2,4 or 8')
    if microbatch_observations!=1 and not hasattr(encoder,'merge_recipe'):raise ValueError('Batched adaptive execution requires Participation B')
    families,weights=layout(views);members={v.task.metadata['id'] for v in views}
    if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm<=0):raise ValueError('Positive explicit clipping threshold required')
    if jepa_views:
        if teacher is None or predictor is None or not callable(latent_loss_fn) or jepa_weight is None or not math.isfinite(jepa_weight) or jepa_weight<=0:raise ValueError('JEPA modules, weight and reviewed loss required')
        if ab_stable is not True or curriculum_end_update!=2000 or logical_update<curriculum_end_update:raise ValueError('JEPA requires stable A/B and the unchanged 2000-update curriculum endpoint')
        ids=[v.source.identifier for v in jepa_views]
        if len(set(ids))!=len(ids) or not set(ids)<=members:raise ValueError('JEPA sources must be distinct members of the same effective batch 8')
        for view in jepa_views:
            if not view.loss_source:raise ValueError('Explicit JEPA loss source required')
            reviewed_pair_task(view.source,view.target,view.admission)
    modules=[encoder,decoder]+([predictor] if predictor is not None else [])
    parameters=[p for module in modules for p in module.parameters()]
    if len({id(p) for p in parameters})!=len(parameters):raise ValueError('Independent Encoder/Decoder/Predictor parameter sets required')
    for module in modules:module.zero_grad(set_to_none=True)
    coefficients=dict(zip(('A','B'),curriculum(logical_update)));rows=[];latent_rows=[]
    try:
        batched=None
        if microbatch_observations!=1:
            from .joint import accumulate_joint
            result=accumulate_joint(encoder,decoder,views,logical_update=logical_update,chunk_pixels=chunk_pixels,precision='fp32',clip_norm=None,backward=True,microbatch_observations=microbatch_observations)
            batched={(row['observation'],row['kind']):row['result'] for row in result['rows']}
        for view in views:
            family='A' if view.task.kind=='A' else 'B'
            result=batched[(view.task.metadata['id'],view.task.kind)] if batched is not None else (stream_adaptive_task(encoder,decoder,view.task,view.plan,train_stds=view.train_stds,chunk_pixels=chunk_pixels,loss_weight=coefficients[family]*weights[id(view)],backward=True) if active(view) else dict(status='NO_VALID_SUPPORTED_TARGET',loss=None,bands=[]))
            exposure=dict(task_signature=task_signature(view.task),rectangle=view.task.rectangle,hidden_band_ids=view.task.metadata.get('hidden_band_ids'),visible_band_ids=[b.record_id for b in view.task.visible],origin_xy=None if view.plan is None else view.plan.origin_xy,target_index_sha256={} if view.plan is None else {b.target.band.record_id:hashlib.sha256(b.indices.numpy().tobytes()).hexdigest() for b in view.plan.bands})
            rows.append(dict(observation=view.task.metadata['id'],kind=view.task.kind,source=view.loss_source,exposure=exposure,result=result))
        ab_loss=sum(coefficients[family]*sum(weights[id(v)]*row['result']['loss'] for v,row in zip(views,rows,strict=True) if active(v) and ('A' if v.task.kind=='A' else 'B')==family) for family in ('A','B'))
        if jepa_views:
            latent_parameters=list(encoder.parameters())+list(predictor.parameters())
            base=[None if p.grad is None else p.grad.detach().clone() for p in latent_parameters]
            buffers={};counts=Counter();losses=defaultdict(list)
            for view in jepa_views:
                for p in latent_parameters:p.grad=None
                loss,result=cross_source_latent_loss(encoder,teacher,predictor,view.source,view.target,admission=view.admission,source_encoder_kwargs=view.source_encoder_kwargs,target_encoder_kwargs=view.target_encoder_kwargs,loss_fn=latent_loss_fn,logical_update=logical_update,curriculum_end_update=curriculum_end_update,ab_stable=ab_stable,weight=jepa_weight)
                latent_rows.append(dict(source=view.loss_source,pair_signature=task_signature(reviewed_pair_task(view.source,view.target,view.admission)),source_encoder_kwargs=view.source_encoder_kwargs,target_encoder_kwargs=view.target_encoder_kwargs,result=result))
                if loss is None:continue
                loss.backward();counts[view.loss_source]+=1;losses[view.loss_source].append(result['loss'])
                destination=buffers.setdefault(view.loss_source,[None]*len(latent_parameters))
                for i,p in enumerate(latent_parameters):
                    if p.grad is not None:
                        if destination[i] is None:destination[i]=p.grad.detach().clone()
                        else:destination[i].add_(p.grad)
            for i,p in enumerate(latent_parameters):
                gradient=base[i]
                for source in counts:
                    contribution=buffers[source][i]
                    if contribution is not None:
                        value=contribution/(len(counts)*counts[source]);gradient=value.clone() if gradient is None else gradient+value
                p.grad=gradient
            latent_mean=sum(sum(v)/len(v) for v in losses.values())/len(losses) if losses else None
        else:latent_mean=None
        norm=None if clip_norm is None else float(torch.nn.utils.clip_grad_norm_(parameters,clip_norm))
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):raise FloatingPointError('Nonfinite joint gradients')
    except BaseException:
        for module in modules:module.zero_grad(set_to_none=True)
        raise
    return dict(effective_batch=8,optimizer_updates=0,logical_update=logical_update,AB_weighted_loss=ab_loss,JEPA_loss=latent_mean,weighted_loss=ab_loss+(jepa_weight*latent_mean if latent_mean is not None else 0.),AB=rows,JEPA=latent_rows,teacher_target_ids=[v.target.identifier for v in jepa_views],gradient_norm_before_clip=norm,normalization='A/B unchanged; JEPA active observation mean within each source, then equal source mean',execution_microbatch_observations=microbatch_observations,execution='independent A/B observation microbatches; per-source JEPA gradient buffers' if microbatch_observations!=1 else 'sequential observations; per-source parameter-gradient buffers; no retained batch of latent graphs')

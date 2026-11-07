"""Participation B Encoder and supervision planning. Curve/seed must be explicit."""
from dataclasses import dataclass
import hashlib,json,math
import torch
from .model import Encoder,positional
from .adaptive_window import AdaptiveWindowEncoder,AdaptiveRepresentation
from .participation_layout import participation_layout
from .adaptive_streaming import prepare_geometry_task
from .streaming import task_signature
from .tasks import Task
from .pair_geometry import frame_for_task


def merge_recipe(rate_curve,selection_seed):
    if isinstance(rate_curve,torch.nn.Module):raise ValueError('Participation curve must not be learned')
    if not callable(rate_curve) or not callable(getattr(rate_curve,'get_config',None)):raise ValueError('Explicit serializable physical rate curve required')
    if type(selection_seed) is not int:raise ValueError('Explicit integer selection seed required')
    return dict(architecture='participation_selected_windows_v1',rate_curve=rate_curve.get_config(),selection_seed=selection_seed,span_rule='1/(1-r)',selection='nearest_complete_group_prefix_physical_sha256',window_learning=False,A_retrieval='represented_boundary_v1')


def _visible_signature(bands):
    return task_signature(Task('merge_geometry',list(bands),[],dict(id='visible_inputs')))


def _geometry_metadata_signature(task,physical_patch_m):
    encoded=json.dumps(dict(metadata=task.metadata,physical_patch_m=physical_patch_m),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ParticipationRepresentation(AdaptiveRepresentation):
    participation_audit: tuple|None=None
    a_adjacency_rule: str='represented_boundary_v1'


class ParticipationEncoder(AdaptiveWindowEncoder):
    def __init__(self,*,rate_curve,selection_seed,hyper_hidden,local_batch_size=64,packed_local_output=False):
        merge_recipe(rate_curve,selection_seed)
        # Keep the existing learned networks; do not instantiate a legacy exponent.
        Encoder.__init__(self,hyper_hidden=hyper_hidden,local_batch_size=local_batch_size,native_mode='grouped',layout_mode='indexed')
        if type(packed_local_output) is not bool:raise ValueError('Explicit boolean packed-output flag required')
        self.packed_local_output=packed_local_output
        self.rate_curve=rate_curve;self.selection_seed=selection_seed;self.global_query_chunk_size=None

    def _local_tokens(self,groups):
        if not self.packed_local_output:return super()._local_tokens(groups)
        if not groups:raise ValueError('Nonempty local groups required')
        buckets={}
        for index,group in enumerate(groups):buckets.setdefault(len(group),[]).append(index)
        chunks=[];order=[]
        for indices in buckets.values():
            for start in range(0,len(indices),self.local_batch_size):
                chosen=indices[start:start+self.local_batch_size];batch=torch.stack([groups[i] for i in chosen])
                chunks.append(self.local(torch.cat((self.cls.expand(len(chosen),-1,-1),batch),1))[:,0]);order.extend(chosen)
        # One permutation restores native group order without one output view per token.
        restore=[0]*len(order)
        for row,index in enumerate(order):restore[index]=row
        outputs=torch.cat(chunks,0)
        return outputs.index_select(0,torch.tensor(restore,dtype=torch.long,device=outputs.device))

    @property
    def merge_recipe(self):return merge_recipe(self.rate_curve,self.selection_seed)

    def get_extra_state(self):return dict(self.merge_recipe,hyper_hidden=self.hyper[0].out_features)

    def _layout_fn(self,origin_xy,audit,layouts=None):
        retained=iter(layouts) if layouts is not None else None
        def layout_fn(band):
            layout=next(retained) if retained is not None else participation_layout(band,rate_curve=self.rate_curve,selection_seed=self.selection_seed,origin_xy=origin_xy)
            audit.append(dict(band=band.record_id,status='NO_VISIBLE_INPUT',visible_positions=0) if layout is None else dict(band=band.record_id,status='OK',**layout['participation_audit']))
            return layout
        return layout_fn

    def _checked_layouts(self,bands,origin_xy,geometry_plan):
        if geometry_plan is None:return None
        if geometry_plan.retained_layouts is None:raise ValueError('Plan does not retain Merge layouts')
        if geometry_plan.merge_recipe!=self.merge_recipe or geometry_plan.origin_xy!=tuple(origin_xy) or geometry_plan.input_signature!=_visible_signature(bands) or len(geometry_plan.retained_layouts)!=len(bands):raise ValueError('Retained Merge geometry differs from current inputs/recipe')
        return geometry_plan.retained_layouts

    def forward(self,bands,*,origin_xy,geometry_plan=None):
        audit=[];layouts=self._checked_layouts(bands,origin_xy,geometry_plan)
        representation=self._forward_with_layouts(bands,origin_xy=origin_xy,layout_fn=self._layout_fn(origin_xy,audit,layouts))
        return ParticipationRepresentation(**vars(representation),participation_audit=tuple(audit))


    def forward_batch(self,observations,*,geometry_plans=None):
        """Each (visible bands, origin) stays independent; no global padding."""
        observations=list(observations)
        if not observations or any(len(item)!=2 or not item[0] for item in observations):raise ValueError('Nonempty (bands, origin) observation pairs required')
        if self.global_query_chunk_size is not None:raise ValueError('Batched execution requires the complete-attention reference')
        plans=[None]*len(observations) if geometry_plans is None else list(geometry_plans)
        if len(plans)!=len(observations):raise ValueError('One geometry plan per independent observation required')
        native=self.native_grouped([band for bands,_ in observations for band in bands]);prepared=[];offset=0
        for (bands,origin),plan in zip(observations,plans,strict=True):
            audit=[];layouts=self._checked_layouts(bands,origin,plan);groups,metadata=self._prepare_with_layouts(bands,origin_xy=origin,layout_fn=self._layout_fn(origin,audit,layouts),native=native[offset:offset+len(bands)])
            prepared.append((groups,metadata,tuple(audit)));offset+=len(bands)
        lengths=[len(groups) for groups,_,_ in prepared]
        tokens=self._local_tokens([group for groups,_,_ in prepared for group in groups]).split(lengths)
        positioned=[z+positional((metadata['centers']-metadata['centers'].new_tensor(metadata['origin_xy'])).to(z.dtype)) for z,(_,metadata,_) in zip(tokens,prepared,strict=True)]
        buckets={}
        for index,n in enumerate(lengths):buckets.setdefault(n,[]).append(index)
        output=[None]*len(observations)
        for indices in buckets.values():
            z=self.global_net(torch.stack([positioned[i] for i in indices]))
            for i,value in zip(indices,z,strict=True):output[i]=value
        return [ParticipationRepresentation(tokens=z,**metadata,participation_audit=audit) for z,(_,metadata,audit) in zip(output,prepared,strict=True)]



@dataclass
class ParticipationTaskPlan:
    geometry: object
    bands: list
    origin_xy: tuple
    kind: str
    observation_id: str
    signature: str
    merge_recipe: dict
    retained_layouts: tuple|None=None
    input_signature: str|None=None
    metadata_signature: str|None=None
    physical_patch_m: float|None=None
    pair_query_frame: object|None=None

    def validate_retained(self,task):
        if self.pair_query_frame!=frame_for_task(task):raise ValueError('Pair query geometry changed after planning')
        if self.retained_layouts is not None and (self.signature!=task_signature(task) or self.metadata_signature!=_geometry_metadata_signature(task,self.physical_patch_m)):
            raise ValueError('Retained geometry data version, physical Patch or task changed')


def prepare_participation_task(task,*,rate_curve,selection_seed,origin_xy,chunk_pixels,cell_indexed=False,retain_layouts=False,physical_patch_m=None):
    recipe=merge_recipe(rate_curve,selection_seed);frame=frame_for_task(task)
    if type(retain_layouts) is not bool:raise ValueError('Explicit boolean geometry-retention flag required')
    if physical_patch_m is not None and (not math.isfinite(physical_patch_m) or physical_patch_m<=0):raise ValueError('Positive physical Patch or explicit not-applicable None required')
    if retain_layouts and not (task.metadata.get('cache_sha256') or task.metadata.get('geometry_data_version')):raise ValueError('Explicit input data version required for retained geometry')
    layouts=[]
    def layout_fn(band):
        layout=participation_layout(band,rate_curve=rate_curve,selection_seed=selection_seed,origin_xy=origin_xy)
        if retain_layouts:layouts.append(layout)
        return layout
    result=prepare_geometry_task(task,layout_fn=layout_fn,origin_xy=origin_xy,chunk_pixels=chunk_pixels,cell_indexed=cell_indexed,a_adjacency_rule='represented_boundary_v1',query_offset_xy=(0.,0.) if frame is None else frame.query_offset_xy)
    if result is None:return None
    geometry,bands=result
    return ParticipationTaskPlan(geometry,bands,tuple(origin_xy),task.kind,task.metadata['id'],task_signature(task),recipe,tuple(layouts) if retain_layouts else None,_visible_signature(task.visible) if retain_layouts else None,_geometry_metadata_signature(task,physical_patch_m) if retain_layouts else None,physical_patch_m,frame)

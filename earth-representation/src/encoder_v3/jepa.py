"""Teacher-support JEPA correspondence and loss graph; no default latent objective."""
from dataclasses import dataclass
import math
import numpy as np
import shapely
import torch
from .pair_geometry import reviewed_pair_task,frame_for_task


@dataclass
class TeacherQueries:
    indices: torch.Tensor
    overlap_m2: torch.Tensor
    overlap_fraction: torch.Tensor
    relative_xy: torch.Tensor
    support_xy: torch.Tensor
    srf_condition: torch.Tensor
    gsd_xy: torch.Tensor
    source_bands: tuple


def _regions(groups,valid,offset_xy=(0.,0.)):
    if len(groups)!=len(valid):raise ValueError('Support count differs from token validity')
    regions=[]
    for cells,enabled in zip(groups,valid.detach().cpu().tolist(),strict=True):
        array=cells.detach().cpu().double().numpy()
        if array.ndim!=2 or array.shape[1]!=4 or not np.isfinite(array).all() or np.any(array[:,2:]<=array[:,:2]):raise ValueError('Exact finite positive native cells required')
        if any(offset_xy):array=array+np.array(tuple(offset_xy)*2,dtype=np.float64)
        regions.append(shapely.union_all(shapely.box(array[:,0],array[:,1],array[:,2],array[:,3])) if enabled else shapely.GeometryCollection())
    return regions


def teacher_support_queries(student,teacher,*,query_offset_xy=(0.,0.)):
    """Geometry only. Student visible footprints deduplicated across all bands."""
    if student.crs!=teacher.crs or student.origin_xy is None:raise ValueError('Common CRS and explicit student physical origin required')
    if len(query_offset_xy)!=2 or not all(math.isfinite(v) for v in query_offset_xy):raise ValueError('Finite Teacher query offset required')
    visible=getattr(student,'visible_cells',None)
    if visible is None:raise ValueError('Actual visible support required; represented cells cannot substitute')
    if any(getattr(teacher,name) is None for name in ('gsd_xy','srf_condition','source_bands')):raise ValueError('Teacher physical conditions required')
    footprint=shapely.union_all(_regions(visible,student.valid));regions=_regions(teacher.cells,teacher.valid,query_offset_xy)
    areas=np.array([region.area for region in regions]);overlap=np.array([region.intersection(footprint).area for region in regions]);selected=np.flatnonzero(overlap>0)
    device=student.tokens.device;dtype=student.tokens.dtype;teacher_ids=torch.as_tensor(selected,device=teacher.tokens.device,dtype=torch.long)
    def condition(value):return value[teacher_ids].detach().to(device=device,dtype=dtype)
    indices=torch.as_tensor(selected,device=device,dtype=torch.long)
    centers=teacher.centers[teacher_ids].detach().double()
    if any(query_offset_xy):centers=centers+centers.new_tensor(query_offset_xy)
    relative=(centers-centers.new_tensor(student.origin_xy)).to(device=device,dtype=dtype)
    bounds=teacher.bounds[teacher_ids].detach().double()
    support=(bounds[:,2:]-bounds[:,:2]).to(device=device,dtype=dtype)
    return TeacherQueries(indices,torch.as_tensor(overlap[selected],device=device,dtype=torch.float64),torch.as_tensor(overlap[selected]/areas[selected],device=device,dtype=torch.float64),relative,support,condition(teacher.srf_condition),condition(teacher.gsd_xy),tuple(teacher.source_bands[int(i)] for i in selected))


def predict_teacher_targets(student,teacher,predictor,*,query_offset_xy=(0.,0.)):
    query=teacher_support_queries(student,teacher,query_offset_xy=query_offset_xy)
    if not len(query.indices):return None,None,query
    prediction=predictor(student.tokens,relative_xy=query.relative_xy,support_xy=query.support_xy,srf_condition=query.srf_condition,gsd_xy=query.gsd_xy)
    teacher_ids=query.indices.to(teacher.tokens.device)
    target=teacher.tokens[teacher_ids].detach().to(prediction)
    return prediction,target,query


def cross_source_latent_loss(student,teacher,predictor,source,target,*,admission,source_encoder_kwargs,target_encoder_kwargs,loss_fn,logical_update,curriculum_end_update,ab_stable,weight):
    """Return a scalar graph for the shared training backward, never step/EMA here.

    loss_fn(prediction, target, teacher_band_ids) must implement the reviewed loss
    and its normalization. Passing a test loss does not freeze a research choice.
    """
    if type(logical_update) is not int or type(curriculum_end_update) is not int or curriculum_end_update<1 or logical_update<curriculum_end_update or ab_stable is not True:
        raise ValueError('JEPA requires completed A/B curriculum and verified stable A/B')
    if not math.isfinite(weight) or weight<=0 or not callable(loss_fn):raise ValueError('Explicit positive JEPA weight and reviewed latent loss required')
    if type(student) is not type(teacher) or any(p.requires_grad for p in teacher.parameters()) or teacher.training:
        raise ValueError('Frozen eval EMA copy of the same Encoder architecture required')
    if [(n,tuple(p.shape)) for n,p in student.named_parameters()]!=[(n,tuple(p.shape)) for n,p in teacher.named_parameters()]:raise ValueError('EMA parameter structure mismatch')
    if type(student).get_extra_state is not torch.nn.Module.get_extra_state and student.get_extra_state()!=teacher.get_extra_state():raise ValueError('EMA Encoder configuration mismatch')
    pair_task=reviewed_pair_task(source,target,admission);frame=frame_for_task(pair_task)
    if source.identifier==target.identifier:raise ValueError('Cross-source JEPA requires distinct observations')
    if source.metadata['role']!='TRAIN':raise ValueError('Training loss only accepts TRAIN pair members')
    representation=student(source.bands,**source_encoder_kwargs)
    with torch.no_grad():target_representation=teacher(target.bands,**target_encoder_kwargs)
    prediction,latent,query=predict_teacher_targets(representation,target_representation,predictor,query_offset_xy=(0.,0.) if frame is None else frame.query_offset_xy)
    report=dict(source_id=source.identifier,target_id=target.identifier,student_tokens=len(representation.tokens),teacher_tokens=len(target_representation.tokens),supported_queries=len(query.indices),unsupported_queries=len(target_representation.tokens)-len(query.indices),optimizer_updates=0,teacher_gradients=False,decoder_called=False)
    if frame is not None:report.update(pair_geometry_record_sha256=frame.record_sha256,query_offset_xy=frame.query_offset_xy)
    if prediction is None:return None,dict(report,status='NO_VALID_SPATIAL_SUPPORT',loss=None)
    loss=loss_fn(prediction,latent,query.source_bands)
    if not isinstance(loss,torch.Tensor) or loss.ndim!=0 or not bool(torch.isfinite(loss)) or not loss.requires_grad:raise ValueError('Latent loss must return one finite differentiable scalar with explicit normalization')
    return weight*loss,dict(report,status='LATENT_GRAPH_READY_NOT_OPTIMIZER_STEP',loss=float(loss.detach()),weight=weight,overlap_fraction_min=float(query.overlap_fraction.min()),overlap_fraction_max=float(query.overlap_fraction.max()),teacher_band_counts={name:query.source_bands.count(name) for name in set(query.source_bands)})

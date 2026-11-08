"""Reviewed query-frame translation; native observations and Encoder stay untouched."""
from dataclasses import dataclass,asdict,replace
import hashlib,json,math
from pathlib import Path
from .tasks import cross_source_task


MECHANISM='query_offset_native_encoder_v1'


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def observation_version(observation):
    version=observation.metadata.get('cache_sha256') or observation.metadata.get('geometry_data_version')
    if not isinstance(version,str) or not version:raise ValueError('Explicit pair data version required')
    return dict(data_version=version,release_ids=observation.metadata.get('frozen_source_ids',{}))


def approved_payload(record):
    payload={k:v for k,v in record.items() if k not in ('offset_review_file','offset_review_sha256')}
    if payload.get('mechanism')!=MECHANISM:raise ValueError('Unknown pair geometry mechanism')
    ids=(payload.get('S2_id'),payload.get('Landsat_id'));delta=payload.get('translation_m')
    if any(not isinstance(x,str) or not x for x in ids) or ids[0]==ids[1]:raise ValueError('Distinct reviewed pair identities required')
    if not isinstance(delta,(list,tuple)) or len(delta)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in delta):raise ValueError('Explicit finite shared pair translation required')
    if not isinstance(payload.get('crs'),str) or not payload['crs'] or set(payload.get('versions',{}))!=set(ids):raise ValueError('Reviewed CRS and both data versions required')
    path=record.get('offset_review_file');expected=record.get('offset_review_sha256')
    if not path or not expected:raise ValueError('Confirmed pair offset value review required; mechanism approval is insufficient')
    blob=Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError('Pair offset review file changed')
    review=json.loads(blob)
    if review.get('status')!='USER_CONFIRMED_PAIR_OFFSET_VALUES' or review.get('record_sha256')!=canonical_sha(payload):raise ValueError('Confirmed pair offset value review does not bind this record')
    return payload


@dataclass(frozen=True)
class PairQueryFrame:
    source_id: str
    target_id: str
    crs: str
    query_offset_xy: tuple
    record_sha256: str

    # tuple repetition -> [dx,dy,dx,dy]; one translation, not a doubled one.
    def boxes(self,boxes):return boxes+boxes.new_tensor(self.query_offset_xy*2)


def _frame(payload,source_id,target_id):
    if {source_id,target_id}!={payload['S2_id'],payload['Landsat_id']}:raise ValueError('Pair geometry identities differ')
    sign=-1 if source_id==payload['S2_id'] else 1
    return PairQueryFrame(source_id,target_id,payload['crs'],tuple(sign*float(v) for v in payload['translation_m']),canonical_sha(payload))


def reviewed_pair_task(source,target,admission):
    task=cross_source_task(source,target,admission)
    record=admission.get('pair_geometry')
    if record is None:return task
    payload=approved_payload(record)
    if payload.get('role')!=source.metadata['role'] or payload['versions'][source.identifier]!=observation_version(source) or payload['versions'][target.identifier]!=observation_version(target):raise ValueError('Pair geometry role or data version differs')
    if any(b.crs!=payload['crs'] for b in source.bands+target.bands):raise ValueError('Pair geometry requires the reviewed common metric coordinate view')
    frame=_frame(payload,source.identifier,target.identifier)
    frozen=dict(source.metadata.get('frozen_source_ids',{}),pair_geometry_record_sha256=frame.record_sha256)
    return replace(task,metadata=dict(task.metadata,pair_query_frame=asdict(frame),pair_target_version=observation_version(target),frozen_source_ids=frozen))


def frame_for_task(task):
    record=task.metadata.get('pair_admission',{}).get('pair_geometry');saved=task.metadata.get('pair_query_frame')
    if task.kind!='B2':
        if saved is not None:raise ValueError('Pair query translation applies only to B2')
        return None
    if record is None:
        if saved is not None:raise ValueError('Pair query context lacks reviewed record')
        return None
    payload=approved_payload(record);frame=_frame(payload,task.metadata['id'],task.metadata['target_id'])
    if saved is None or canonical_sha(saved)!=canonical_sha(asdict(frame)):raise ValueError('Reviewed pair query context missing or changed; use reviewed_pair_task')
    native_ids={k:v for k,v in task.metadata.get('frozen_source_ids',{}).items() if k!='pair_geometry_record_sha256'}
    actual=dict(data_version=task.metadata.get('cache_sha256') or task.metadata.get('geometry_data_version'),release_ids=native_ids)
    if payload.get('role')!=task.metadata['role'] or actual!=payload['versions'][frame.source_id] or task.metadata.get('pair_target_version')!=payload['versions'][frame.target_id]:raise ValueError('Pair query data version/role changed')
    if task.metadata['frozen_source_ids'].get('pair_geometry_record_sha256')!=frame.record_sha256 or any(b.crs!=frame.crs for b in task.visible+[t.band for t in task.targets]):raise ValueError('Pair query frame/CRS changed')
    return frame


class QueryFrameDecoder:
    """Geometry-only call adapter; owns no learned parameters or optimizer state."""
    def __init__(self,decoder,frame):self.decoder=decoder;self.frame=frame
    def __call__(self,representation,targets,srf_condition,*,kind,mask_bounds=None):
        if kind!='B2' or representation.crs!=self.frame.crs:raise ValueError('Pair query Decoder requires matching B2 source frame')
        return self.decoder(representation,self.frame.boxes(targets),srf_condition,kind=kind,mask_bounds=mask_bounds)


def decoder_for_plan(decoder,plan):
    frame=getattr(plan,'pair_query_frame',None)
    return decoder if frame is None else QueryFrameDecoder(decoder,frame)

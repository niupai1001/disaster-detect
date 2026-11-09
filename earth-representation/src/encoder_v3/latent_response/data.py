"""Hash-bound legacy cache adapter, frozen TRAIN z transform and value-isolated A/B1 views."""
from dataclasses import dataclass
import math
import json
import torch
from encoder_v3.data import NativeOpticalDataset, checked_file, sha256
from encoder_v3.tasks import spatial_task, fixed_spectral_task
from .types import SensorKey, ChannelObservation
from .registry import ChannelCatalog
from .geometry import FullGeometry, native_cells


def load_catalogs(path):
    doc = json.loads(path.read_text(encoding='utf-8-sig'))
    return {SensorKey(s['platform'], s['instrument']):ChannelCatalog(s['total_channels'], s['ordinals']) for s in doc['sensors']}


class LRDataset(NativeOpticalDataset):
    def __init__(self, release, physical_contracts, *, role, catalogs):
        super().__init__(release, physical_contracts, role=role)
        self.catalogs = catalogs
        path = self.root/'statistics.json'
        checked_file(path, self.release_metadata['statistics_sha256'])
        self.statistics = json.loads(path.read_text(encoding='utf-8-sig'))
        if not self.statistics.get('scope', '').startswith('TRAIN-only'):
            raise ValueError('TRAIN-only statistics required')

    def channels(self, observation, device):
        key = SensorKey(observation.metadata['sensor'], observation.metadata['instrument'])
        channels = []
        for b in observation.bands:
            stat = self.statistics['bands'][key.instrument+':'+b.record_id]
            mu, std = stat['mean'], stat['std']
            if not math.isfinite(mu) or not math.isfinite(std) or std<=0:
                raise ValueError('Invalid frozen mean/std')
            channels.append(ChannelObservation(key, self.catalogs[key].meta(b.record_id),
                (b.reflectance.to(device)-mu)/std, b.valid_mask.to(device), b.keep().to(device), b.affine, b.crs))
        return channels


@dataclass
class View:
    kind: str
    channels: list
    geometry: FullGeometry
    targets: list


def make_view(observation, channels, kind):
    geometry = FullGeometry.from_channels(channels)
    if kind=='A':
        x0,y0=geometry.origin; w,h=geometry.extent
        # A deterministic central rectangle hides complete Sub-Patches when the scene permits.
        side=geometry.patch_size
        x1=x0+max(0, math.floor(w/side/4))*side
        y1=y0+max(0, math.floor(h/side/4))*side
        rectangle=(x1,y1,min(x0+w,x1+max(side,math.floor(w/side/2)*side)),
                   min(y0+h,y1+max(side,math.floor(h/side/2)*side)))
        task=spatial_task(observation,rectangle)
    elif kind=='B1':
        task=fixed_spectral_task(observation,[observation.bands[-1].record_id])
    else:
        raise ValueError('Only A and B1 are enabled')
    by_name={c.channel_meta.channel_id:c for c in channels}
    visible=[]
    for b in task.visible:
        c=by_name[b.record_id]
        visible.append(ChannelObservation(c.sensor_key,c.channel_meta,c.values.clone(),c.valid.clone(),
                                          b.keep().to(c.values.device),c.affine,c.crs))
    targets=[]
    for t in task.targets:
        c=by_name[t.band.record_id]
        mask=t.mask.to(c.values.device)&c.valid
        if bool(mask.any()):
            # Clone truth: hidden-input poisoning must never mutate labels.
            targets.append((c, mask, c.values.clone()))
    if not targets:
        raise ValueError('NO_VALID_TARGETS')
    return View(kind,visible,geometry,targets)


def target_chunks(channel, mask, truth, size):
    rc=mask.nonzero()
    a,_,x0,_,e,y0=channel.affine
    for start in range(0,len(rc),size):
        part=rc[start:start+size]
        points=torch.stack((x0+(part[:,1].double()+.5)*a,y0+(part[:,0].double()+.5)*e),-1)
        yield points,truth[part[:,0],part[:,1]]


def pool_recipe(rows):
    from collections import Counter
    selected=[r for r in rows if r['role']=='TRAIN' and len(r['bands'])>=2 and r['sensor'] in ('S2','L8','L9')]
    return dict(rule='all TRAIN optical members with >=2 bands; no pixel or side cap',
                ids=[r['id'] for r in selected], counts=dict(Counter(r['sensor'] for r in selected)),
                member_count=len(selected))

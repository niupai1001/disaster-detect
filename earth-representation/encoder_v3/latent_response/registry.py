"""One persistent P32 per SensorKey plus a non-learnable native channel catalog."""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping
import torch
from torch import nn
from .types import SensorKey, ChannelMeta


@dataclass(frozen=True)
class ChannelCatalog:
    """Native catalog ranks; never a learned per-band embedding table."""
    total_channels: int
    ordinals: Mapping[str, int]

    def __post_init__(self):
        ordinals = dict(self.ordinals)
        if type(self.total_channels) is not int or self.total_channels < 1:
            raise ValueError('Positive integer catalog size required')
        if not ordinals:
            raise ValueError('Nonempty channel catalog required')
        for channel_id, ordinal in ordinals.items():
            if not isinstance(channel_id, str) or not channel_id:
                raise ValueError('Nonempty string channel ids required')
            if type(ordinal) is not int or not 0 <= ordinal < self.total_channels:
                raise ValueError(f'Ordinal of {channel_id} must lie inside the catalog')
        if len(set(ordinals.values())) != len(ordinals):
            raise ValueError('Canonical ordinals must be unique')
        object.__setattr__(self, 'ordinals', MappingProxyType(ordinals))

    @classmethod
    def from_ordered(cls, channel_ids, *, total_channels=None):
        ids = tuple(channel_ids)
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate channel ids')
        total = len(ids) if total_channels is None else total_channels
        if total < len(ids):
            raise ValueError('Catalog size cannot be smaller than the listed channels')
        return cls(total, {channel_id: index for index, channel_id in enumerate(ids)})

    @property
    def channel_ids(self):
        return tuple(sorted(self.ordinals, key=self.ordinals.get))

    def meta(self, channel_id):
        if channel_id not in self.ordinals:
            raise KeyError(f'Channel {channel_id!r} absent from the native catalog; refusing a default identity')
        return ChannelMeta(channel_id, self.ordinals[channel_id], self.total_channels)

    def as_dict(self):
        return dict(total_channels=self.total_channels,
                    ordinals={key: self.ordinals[key] for key in self.channel_ids})


class SensorRegistry(nn.Module):
    """SensorKey -> exactly one [p_dim] parameter. Unknown keys and channels fail closed."""

    def __init__(self, catalogs: Mapping[SensorKey, ChannelCatalog], *, seed, p_dim=32, init_std=0.02):
        super().__init__()
        catalogs = dict(catalogs)
        if not catalogs:
            raise ValueError('At least one registered SensorKey required')
        if any(not isinstance(key, SensorKey) or not isinstance(value, ChannelCatalog) for key, value in catalogs.items()):
            raise ValueError('Typed SensorKey -> ChannelCatalog mapping required')
        if type(seed) is not int or type(p_dim) is not int or p_dim < 1:
            raise ValueError('Integer seed and positive descriptor width required')
        if not isinstance(init_std, (int, float)) or not init_std > 0:
            raise ValueError('Positive descriptor initialization scale required')
        self.catalogs = MappingProxyType(catalogs)
        self.p_dim = p_dim
        self.seed = seed
        self.init_std = float(init_std)
        self.p = nn.ParameterDict()
        for key in sorted(catalogs, key=lambda item: item.uid):
            # Per-key seed: adding a sensor never shifts another sensor's initialization.
            digest = hashlib.sha256(f'{seed}:{key.uid}'.encode()).digest()[:8]
            generator = torch.Generator().manual_seed(int.from_bytes(digest, 'big'))
            value = torch.empty(p_dim).normal_(0.0, self.init_std, generator=generator)
            self.p[key.uid] = nn.Parameter(value)

    def get(self, sensor_key):
        if sensor_key not in self.catalogs:
            raise KeyError(f'Unregistered SensorKey {sensor_key.uid}; refusing a default descriptor')
        return self.p[sensor_key.uid]

    def forward(self, sensor_key):
        return self.get(sensor_key)

    def meta(self, sensor_key, channel_id):
        if sensor_key not in self.catalogs:
            raise KeyError(f'Unregistered SensorKey {sensor_key.uid}')
        return self.catalogs[sensor_key].meta(channel_id)

    def manifest(self):
        return dict(schema='latent_response_sensor_registry_v1', p_dim=self.p_dim,
                    init_std=self.init_std, seed=self.seed,
                    catalogs=[dict(sensor_key=dict(platform=key.platform, instrument=key.instrument),
                                   **self.catalogs[key].as_dict())
                              for key in sorted(self.catalogs, key=lambda item: item.uid)])

    @classmethod
    def from_manifest(cls, manifest):
        if manifest.get('schema') != 'latent_response_sensor_registry_v1':
            raise ValueError('Unknown sensor registry schema')
        catalogs = {}
        for entry in manifest['catalogs']:
            key = SensorKey(**entry['sensor_key'])
            if key in catalogs:
                raise ValueError(f'Duplicate SensorKey {key.uid} in manifest')
            catalogs[key] = ChannelCatalog(entry['total_channels'], entry['ordinals'])
        return cls(catalogs, seed=manifest['seed'], p_dim=manifest['p_dim'], init_std=manifest['init_std'])

    def identity_sha256(self):
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(payload).hexdigest()

    @torch.no_grad()
    def parameter_digest(self):
        digest = hashlib.sha256()
        for uid in sorted(self.p.keys()):
            digest.update(uid.encode())
            digest.update(self.p[uid].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

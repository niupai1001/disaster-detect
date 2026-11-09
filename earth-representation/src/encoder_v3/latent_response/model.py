"""LR complete single-scene representation, batch scenes remain independent."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .registry import SensorRegistry
from .sensor_memory import SensorMemory
from .module_query import ModuleQuery
from .memory_read import read_memory
from .dynamic_conv import DynamicKernelBank
from .fusion import LocalEmbed, ChannelFusion, channel_front
from .spatial import SpatialBackbone
from .decoder import Decoder


@dataclass
class Representation:
    geometry: object
    index: torch.Tensor
    centers: torch.Tensor
    fine: torch.Tensor
    latent: torch.Tensor
    latent_valid: torch.Tensor
    global_latent: torch.Tensor
    memory: tuple
    sensor_key: object
    census: dict


class LatentResponseModel(nn.Module):
    def __init__(self, catalogs, *, seed=17):
        super().__init__()
        self.registry = SensorRegistry(catalogs, seed=seed)
        self.memory = SensorMemory()
        self.module_query = ModuleQuery(frequencies=(1.,2.,4.))
        self.kernels = DynamicKernelBank()
        self.embed = LocalEmbed()
        self.fusion = ChannelFusion()
        self.spatial = SpatialBackbone()
        self.decoder = Decoder()

    def condition(self, kind, key, meta, gsd, memory):
        if meta != self.registry.meta(key, meta.channel_id):
            raise ValueError('Channel metadata disagrees with registry')
        return read_memory(self.module_query(kind, meta, gsd), *memory)[1]

    def encode(self, channels, geometry, *, tile_rows=128, chunk_rows=None):
        geometry.check_channels(channels)
        if not channels or len({c.sensor_key for c in channels})!=1:
            raise ValueError('Exactly one observation sensor per representation required')
        if len({c.channel_meta.channel_id for c in channels})!=len(channels):
            raise ValueError('Duplicate input channel identity')
        channels = sorted(channels, key=lambda c:c.channel_meta.canonical_ordinal)
        key = channels[0].sensor_key
        memory = self.memory(self.registry.get(key))
        h, support, conditions = [], [], []
        for c in channels:
            c.validate()
            if not bool(c.keep().any()):
                continue
            z = self.condition('conv', key, c.channel_meta, c.gsd_xy, memory)
            kernel = self.kernels.kernels(z[None])
            # Bind each channel explicitly: checkpoint recomputation must not capture the final loop item.
            def front(values, keep, weight, bias, channel=c):
                return channel_front(values, keep, weight, bias, channel, geometry, self.embed, tile_rows)
            args = (c.values, c.keep(), kernel, self.kernels.bias)
            a, b = checkpoint(front, *args, use_reentrant=False) if torch.is_grad_enabled() else front(*args)
            h.append(a); support.append(b)
            conditions.append(self.condition('channel', key, c.channel_meta, c.gsd_xy, memory))
        if not h:
            raise ValueError('NO_VISIBLE_CONTEXT')
        ids, summary, latent, latent_valid, counts = self.fusion(torch.stack(h,1), torch.stack(support,1), torch.stack(conditions))
        index, centers = geometry.patch_centers(summary.device)
        index, centers = index[ids], centers[ids]
        fine, latent, global_latent, census = self.spatial(summary, latent, latent_valid, index, centers, geometry, chunk_rows)
        return Representation(geometry, index, centers, fine, latent, latent_valid, global_latent, memory, key, census)

    def predict(self, representation, meta, gsd, points, *, chunk_pixels=128):
        z = self.condition('decoder', representation.sensor_key, meta, gsd, representation.memory)
        return self.decoder(representation, points, gsd, z, chunk_pixels)

    def encode_batch(self, scenes, **kwargs):
        return [self.encode(channels, geometry, **kwargs) for channels, geometry in scenes]

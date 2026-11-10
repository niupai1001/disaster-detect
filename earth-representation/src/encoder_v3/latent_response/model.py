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
from .fusion import LocalEmbed, ChannelFusion, channel_front, grouped_channel_front
from .geometry_cache import GeometryCache
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
        self.geometry_cache = GeometryCache()

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
        # Metadata groups share geometry and execute one grouped convolution per tile.
        groups={}
        for i,c in enumerate(channels):
            c.validate();groups.setdefault(c.grid,[]).append((i,c))
        entries=[]
        for grid,members in groups.items():
            present=torch.stack([c.keep().any() for _,c in members]).cpu().tolist()
            members=[m for m,valid in zip(members,present) if valid]
            if not members:continue
            per_channel=(geometry.shape[0]*geometry.shape[1]*16*32+min(tile_rows,grid.shape[0])*grid.shape[1]*32)*channels[0].values.element_size()
            capacity=max(1,(128*1024**2)//per_channel)
            plans=tuple(self.geometry_cache.plan(grid,geometry,channels[0].values.device,begin,min(begin+tile_rows,grid.shape[0])) for begin in range(0,grid.shape[0],tile_rows))
            for start in range(0,len(members),capacity):
                subset=members[start:start+capacity];metas=[c.channel_meta for _,c in subset]
                if any(meta!=self.registry.meta(key,meta.channel_id) for meta in metas):raise ValueError('Metadata differs from registry')
                z=read_memory(self.module_query.forward_many('conv',metas,[grid.gsd_xy]*len(metas)),*memory)[1]
                kernels=self.kernels.kernels(z)
                values=torch.stack([c.values for _,c in subset]);keep=torch.stack([c.keep() for _,c in subset])
                def front(v,k,w,b,plans=plans):
                    return grouped_channel_front(v,k,w,b,plans,geometry,self.embed,tile_rows)
                args=(values,keep,kernels,self.kernels.bias)
                h_group,s_group=checkpoint(front,*args,use_reentrant=False) if torch.is_grad_enabled() else front(*args)
                for position,(original,c) in enumerate(subset):entries.append((original,c,h_group[position],s_group[position]))
        entries.sort(key=lambda item:item[0])
        h=[e[2] for e in entries];support=[e[3] for e in entries]
        if h:
            metas=[e[1].channel_meta for e in entries];gsds=[e[1].gsd_xy for e in entries]
            conditions=read_memory(self.module_query.forward_many('channel',metas,gsds),*memory)[1]
        if not h:
            raise ValueError('NO_VISIBLE_CONTEXT')
        ids, summary, latent, latent_valid, counts = self.fusion(torch.stack(h,1), torch.stack(support,1), conditions)
        index, centers = geometry.patch_centers(summary.device)
        index, centers = index[ids], centers[ids]
        fine, latent, global_latent, census = self.spatial(summary, latent, latent_valid, index, centers, geometry, chunk_rows)
        return Representation(geometry, index, centers, fine, latent, latent_valid, global_latent, memory, key, census)

    def predict(self, representation, meta, gsd, points, *, chunk_pixels=128):
        z = self.condition('decoder', representation.sensor_key, meta, gsd, representation.memory)
        return self.decoder(representation, points, gsd, z, chunk_pixels)

    def encode_batch(self, scenes, **kwargs):
        return [self.encode(channels, geometry, **kwargs) for channels, geometry in scenes]

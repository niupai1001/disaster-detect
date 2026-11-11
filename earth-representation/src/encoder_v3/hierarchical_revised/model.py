"""One-way Fine → Middle → Regional → Global encoder and native readout."""
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch import nn

from .attention import Read, ff, full_graph
from .readout import ReadoutTiles
from .grouped import Factorized
from .frontend import Descriptor, NativeConvolution
from .geometry import CSR, Grid, middle_map, regional_ids
from .routes import (NativeIndex, combine, fixed_lookup, grouped_keys, groups_graph,
                     middle_spatial, native_graph, target_groups)


@dataclass(frozen=True)
class Config:
    fine: int
    middle: int
    coarse: int
    descriptor: int
    bases: int
    spatial_queries: int
    joint_queries: int
    regional_queries: int
    global_queries: int
    depth: int
    heads: int
    ff_ratio: int
    channel_cap: int
    middle_metres: int
    regional_side: int
    decoder: int
    query_chunk: int
    conv_tile_rows: int
    mask_patch_metres: float
    mask_ratio: float
    decoder_query_chunk: int = 32768

    def __post_init__(self):
        for name, value in vars(self).items():
            if name not in ("mask_patch_metres", "mask_ratio") and (type(value) is not int or value <= 0):
                raise ValueError("Expected positive integer: " + name)
        if (not math.isfinite(self.mask_patch_metres) or self.mask_patch_metres <= 0
                or not 0 < self.mask_ratio < 1):
            raise ValueError("Invalid A mask configuration")
        if min(self.spatial_queries, self.joint_queries, self.regional_queries, self.global_queries) < 2:
            raise ValueError("Multiple queries required at every compression level")
        if self.regional_side < 2 or any(width % self.heads for width in
                                        (self.fine, self.middle, self.coarse, self.decoder)):
            raise ValueError("Invalid regional partition or attention width")


@dataclass(frozen=True)
class VisibleChannel:
    values: torch.Tensor
    keep: torch.Tensor
    grid: Grid
    physical: object


@dataclass(frozen=True)
class Encoded:
    fine: torch.Tensor
    native: NativeIndex
    descriptor: torch.Tensor
    preserving: torch.Tensor
    middle_cells: np.ndarray
    middle_channels: np.ndarray
    joint: torch.Tensor
    joint_cells: np.ndarray
    regional: torch.Tensor
    regional_cells: np.ndarray
    global_latents: torch.Tensor
    middle_areas: np.ndarray
    # Sparse per-channel Middle coverage; Regional support is the union of child supports.
    middle_support: tuple


def queries(count, width):
    return nn.Parameter(torch.randn(count, width)*.02)


def repeat_queries(parameter, groups):
    return parameter[None].expand(groups, -1, -1).flatten(0, 1)


def geo_features(positions, ref):
    coords = torch.as_tensor(positions, device=ref.device, dtype=ref.dtype)
    rad = torch.deg2rad(coords)
    return torch.cat((rad.sin(), rad.cos()), dim=-1)


def grid_position(cells, step, ref):
    coordinates = torch.as_tensor(cells, device=ref.device, dtype=ref.dtype)
    return (coordinates + .5)*step/1_000_000


class FineBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.space = Read(config.fine, config.fine, config.heads)
        self.cross = Read(config.fine, config.fine, config.heads)
        self.ff = ff(config.fine, config.ff_ratio)

    def forward(self, x, descriptor, space, cross, chunk, metrics):
        x = self.space(x, x, space, chunk, "fine_space", metrics)
        conditioned = x + descriptor
        updated = self.cross(conditioned, conditioned, cross, chunk, "fine_cross", metrics)
        x = x + (updated-conditioned)
        return x + self.ff(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        f, m, c, h = config.fine, config.middle, config.coarse, config.heads
        self.descriptor = Descriptor(config.descriptor)
        self.conv = NativeConvolution(config.descriptor, f, config.bases)
        self.fine_cond = nn.Linear(config.descriptor, f, bias=False)
        self.fine_position = nn.Linear(4, f, bias=False)
        self.fine_blocks = nn.ModuleList(FineBlock(config) for _ in range(config.depth))
        self.compress_queries = queries(config.spatial_queries, m)
        self.middle_cond = nn.Linear(config.descriptor, m, bias=False)
        self.middle_pos = nn.Linear(2, m, bias=False)
        self.compress = Read(m, f, h)
        self.middle_space = nn.ModuleList(Read(m, m, h) for _ in range(config.depth))
        self.middle_channel = nn.ModuleList(Read(m, m, h) for _ in range(config.depth))
        self.middle_ff = nn.ModuleList(ff(m, config.ff_ratio) for _ in range(config.depth))
        self.joint_queries = queries(config.joint_queries, m)
        self.joint_read = Read(m, m, h)
        self.region_queries = queries(config.regional_queries, c)
        self.region_pos = nn.Linear(2, c, bias=False)
        self.region_read = Read(c, m, h)
        self.global_queries = queries(config.global_queries, c)
        self.global_read = Read(c, c, h)
        self.global_blocks = nn.ModuleList(Read(c, c, h) for _ in range(config.depth))
        self.global_ff = nn.ModuleList(ff(c, config.ff_ratio) for _ in range(config.depth))

    def forward(self, channels, metrics=None):
        cfg = self.config
        grids = [channel.grid for channel in channels]
        specs = [channel.physical for channel in channels]
        before = time.perf_counter()
        maps = [middle_map(grid, cfg.middle_metres) for grid in grids]
        geometry_seconds = time.perf_counter()-before
        native = NativeIndex.build(grids, [channel.keep for channel in channels])
        if not native.offsets[-1]:
            raise ValueError("NO_VISIBLE_CONTEXT")
        active = [i for i, ids in enumerate(native.ids) if len(ids)]
        visible_descriptor = self.descriptor([specs[i] for i in active])
        active_index = torch.tensor(active, device=visible_descriptor.device)
        descriptor = visible_descriptor.new_zeros(len(channels), cfg.descriptor).index_copy(
            0, active_index, visible_descriptor)
        kernels, biases = self.conv.generator(visible_descriptor)
        conditioned = self.fine_cond(visible_descriptor)
        feats, cond = [], []
        for j, i in enumerate(active):
            channel = channels[i]
            ids = native.ids[i]
            values = self.conv(channel.values, channel.keep, kernels[j], biases[j], cfg.conv_tile_rows)
            feats.append(values + self.fine_position(geo_features(maps[i].positions[ids], values)))
            cond.append(conditioned[j].expand(len(ids), -1))
        fine, fine_cond = torch.cat(feats), torch.cat(cond)
        del feats, cond
        space = native_graph(native, native, specs, cfg.channel_cap, "space")
        cross = native_graph(native, native, specs, cfg.channel_cap, "cross", metrics)
        for block in self.fine_blocks:
            fine = block(fine, fine_cond, space, cross, cfg.query_chunk, metrics)
        del fine_cond

        cells, band, key_pixels, key_groups, coverage, support = [], [], [], [], [], []
        count = 0
        for c in active:
            mapping = maps[c]
            packed = native.packed[c][mapping.pixels]
            valid = packed >= 0
            group, index = np.unique(mapping.cell_index[valid], return_inverse=True)
            cells.append(mapping.cells[group])
            band.append(np.full(len(group), c, dtype=np.int32))
            key_pixels.append(packed[valid])
            key_groups.append(index+count)
            coverage.append(np.bincount(index, weights=mapping.area[valid], minlength=len(group)))
            support.append((c, mapping, valid))
            count += len(group)
        cells, band = np.concatenate(cells), np.concatenate(band)
        cell_ids, local_group = np.unique(cells, axis=0, return_inverse=True)
        spatial = CSR.from_pairs(np.concatenate(key_groups), np.concatenate(key_pixels), count)
        q = repeat_queries(self.compress_queries, count)
        q = q + (self.middle_cond(descriptor[band]) +
                 self.middle_pos(grid_position(cells, cfg.middle_metres, q))).repeat_interleave(cfg.spatial_queries, 0)
        before_middle = time.perf_counter()
        preserving = self.compress(q, fine, groups_graph(np.repeat(np.arange(count), cfg.spatial_queries), spatial),
                                   cfg.query_chunk, "fine_to_middle", metrics)
        middle_space = middle_spatial(cells, band, cfg.spatial_queries)
        channel_group = np.repeat(local_group, cfg.spatial_queries)
        middle_channel = groups_graph(channel_group, grouped_keys(channel_group))
        for space, channel, mlp in zip(self.middle_space, self.middle_channel, self.middle_ff):
            preserving = space(preserving, preserving, middle_space, cfg.query_chunk, "middle_space", metrics)
            # Exclude keys from the same channel in the cross-channel step.
            def inter(start, end):
                src, dst = middle_channel(start, end)
                keep = band[(src+start)//cfg.spatial_queries] != band[dst//cfg.spatial_queries]
                return src[keep], dst[keep]
            preserving = channel(preserving, preserving, inter, cfg.query_chunk, "middle_channel", metrics)
            preserving = preserving + mlp(preserving)
        joint = self.joint_read(repeat_queries(self.joint_queries, len(cell_ids)), preserving,
                                groups_graph(np.repeat(np.arange(len(cell_ids)), cfg.joint_queries),
                                             grouped_keys(channel_group)),
                                cfg.query_chunk, "middle_joint", metrics)

        regions, regional_group = np.unique(regional_ids(cell_ids, cfg.regional_side), axis=0, return_inverse=True)
        q = repeat_queries(self.region_queries, len(regions))
        q = q + self.region_pos(grid_position(regions, cfg.middle_metres*cfg.regional_side, q)
                                ).repeat_interleave(cfg.regional_queries, 0)
        regional = self.region_read(q, joint, groups_graph(np.repeat(np.arange(len(regions)), cfg.regional_queries),
                                    grouped_keys(np.repeat(regional_group, cfg.joint_queries))),
                                    cfg.query_chunk, "regional", metrics)
        global_latents = self.global_read(self.global_queries, regional, full_graph(len(regional)),
                                          cfg.query_chunk, "global_read", metrics)
        for block, mlp in zip(self.global_blocks, self.global_ff):
            global_latents = block(global_latents, global_latents, full_graph(len(global_latents)),
                                   cfg.query_chunk, "global_self", metrics)
            global_latents = global_latents + mlp(global_latents)
        if metrics is not None:
            metrics.tokens.update(fine=len(fine), preserving=len(preserving), joint=len(joint),
                                  regional=len(regional), global_latents=len(global_latents))
            metrics.geometry = dict(middle_cache_wall_seconds=geometry_seconds,
                                    through_fine_and_middle_index_wall_seconds=before_middle-before,
                                    max_projection_area_error=max(float(m.area_error) for m in maps))
        return Encoded(fine, native, descriptor, preserving, cells, band, joint, cell_ids, regional,
                       regions, global_latents, np.concatenate(coverage), tuple(support))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, h = config.decoder, config.heads
        self.condition = nn.Linear(config.descriptor + 5, d)
        self.local = Read(d, (config.fine, config.middle), h)
        self.global_read = Read(d, config.coarse, h)
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def prepare_coarse(self, encoded, native, target_descriptor, to_region, metrics):
        cfg = self.config
        empty = lambda start, end: (np.empty(0, dtype=int), np.empty(0, dtype=int))
        graph = Factorized(empty, to_region, None, len(encoded.global_latents), cfg.regional_queries,
                           len(encoded.global_latents))
        kv = self.global_read.project_context(
            (torch.cat((encoded.global_latents, encoded.regional)),), "decoder_coarse", metrics)
        return graph, kv

    def forward(self, encoded, targets, specs, target_descriptor, metrics=None):
        cfg = self.config
        native = targets
        if not native.offsets[-1]:
            raise ValueError("NO_VALID_TARGETS")
        local_native = native_graph(native, encoded.native, specs, cfg.channel_cap, "decoder")
        # Every geometrically corresponding visible channel remains available at Middle.
        lookup = {}
        for i, cell in enumerate(encoded.middle_cells):
            lookup.setdefault(tuple(cell), []).append(i)
        mid_coords = {cell: i for i, cell in enumerate(lookup)}
        mid_groups = CSR.from_pairs(np.repeat(np.arange(len(lookup)), [len(v) for v in lookup.values()]),
                                    np.concatenate(list(lookup.values())), len(lookup))
        to_middle = target_groups(native, cfg.middle_metres, fixed_lookup(mid_coords))
        local_graph = Factorized(local_native, to_middle, mid_groups, len(encoded.fine), cfg.spatial_queries)
        to_region = target_groups(native, cfg.middle_metres,
                                  fixed_lookup({tuple(cell): i for i, cell in enumerate(encoded.regional_cells)}),
                                  cfg.regional_side)
        local_kv = self.local.project_context((encoded.fine, encoded.preserving), "decoder_local", metrics)
        coarse_graph, coarse_kv = self.prepare_coarse(encoded, native, target_descriptor, to_region, metrics)
        def query_features(start, end):
            descriptor_rows, geography = [], []
            for c, _, ids in native.slices(start, end):
                mapping = middle_map(native.grids[c], cfg.middle_metres)
                descriptor_rows.append(np.full(len(ids), c, dtype=np.int64))
                features = geo_features(mapping.positions[ids], target_descriptor)
                area = features.new_tensor(mapping.pixel_areas[ids]).log()[:, None]
                geography.append(torch.cat((features, area), dim=1))
            return np.concatenate(descriptor_rows), torch.cat(geography)
        if metrics is not None:
            metrics.tokens["targets"] = int(native.offsets[-1])
        parameters = (list(self.condition.parameters()) + list(self.output.parameters()) +
                      [p for block in (self.local, self.global_read)
                       for part in (block.query_norm, block.q, block.output) for p in part.parameters()])
        return ReadoutTiles.apply(target_descriptor, *local_kv, *coarse_kv, self,
                                  query_features, local_graph, coarse_graph,
                                  int(native.offsets[-1]), metrics, *parameters)

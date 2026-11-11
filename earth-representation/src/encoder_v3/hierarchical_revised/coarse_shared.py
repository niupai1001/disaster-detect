"""Experimental support/channel-shared coarse decoder; baseline stays per pixel."""
from dataclasses import dataclass
from functools import lru_cache
import time

import numpy as np
import torch

from .geometry import CSR, expand, middle_map, persistent_geometry, regional_ids
from .model import Decoder
from .readout import SharedCoarseContext
from .routes import expand_latents, groups_graph


@dataclass
class RegionalSupports:
    pixel_group: np.ndarray
    cells: np.ndarray
    offsets: np.ndarray
    keys: np.ndarray


@lru_cache(maxsize=24)
def regional_supports(grid, step, side):
    def build():
        mapping = middle_map(grid, step)
        cells, region = np.unique(regional_ids(mapping.cells, side), axis=0, return_inverse=True)
        code = np.unique(mapping.pixels.astype(np.int64)*len(cells) + region[mapping.cell_index])
        rows, keys = code//len(cells), code%len(cells)
        counts = np.bincount(rows, minlength=grid.size)
        matrix = np.full((grid.size, int(counts.max())), -1, dtype=np.int32)
        _, slot = expand(counts)
        matrix[rows, slot] = keys
        signatures, groups = np.unique(matrix, axis=0, return_inverse=True)
        owners, positions = np.nonzero(signatures >= 0)
        support = CSR.from_pairs(owners, signatures[owners, positions], len(signatures))
        return RegionalSupports(groups.astype(np.int32), cells, support.offsets, support.keys)
    return persistent_geometry("decoder_regional_supports_v1", (grid, step, side), RegionalSupports, build)


class SharedCoarseDecoder(Decoder):
    """One coarse read per target channel and complete Regional support signature."""

    def prepare_coarse(self, encoded, native, target_descriptor, to_region, metrics):
        cfg = self.config
        before = time.perf_counter()
        visible = {tuple(cell): i for i, cell in enumerate(encoded.regional_cells)}
        bands, positions, target_groups, row_ids, key_ids = [], [], [], [], []
        feature_cache = {}
        count = 0
        for channel, (grid, ids) in enumerate(zip(native.grids, native.ids)):
            if not len(ids):
                continue
            support = regional_supports(grid, cfg.middle_metres, cfg.regional_side)
            signature = support.pixel_group[ids]
            groups, inverse = np.unique(signature, return_inverse=True)
            target_groups.append((inverse + count).astype(np.int32))
            bands.extend([channel]*len(groups))
            if grid not in feature_cache:
                mapping = middle_map(grid, cfg.middle_metres)
                radians = np.deg2rad(mapping.positions).astype(np.float32)
                feature_cache[grid] = np.column_stack(
                    (np.sin(radians), np.cos(radians), np.log(mapping.pixel_areas))).astype(np.float32)
            geography = feature_cache[grid][ids]
            denominator = np.bincount(inverse, minlength=len(groups))
            means = np.column_stack([np.bincount(inverse, weights=geography[:, j],
                                                minlength=len(groups))/denominator for j in range(5)])
            positions.append(means)
            relation = CSR(support.offsets, support.keys)
            rows, cell_ids = relation.take(groups)
            translated = np.array([visible.get(tuple(cell), -1) for cell in support.cells], dtype=np.int32)
            active = translated[cell_ids] >= 0
            regional = translated[cell_ids[active]]
            row_ids.extend((np.repeat(rows[active]+count, cfg.regional_queries),
                            np.repeat(np.arange(count, count+len(groups)), cfg.global_queries)))
            key_ids.extend((len(encoded.global_latents)+expand_latents(regional, cfg.regional_queries),
                            np.tile(np.arange(len(encoded.global_latents)), len(groups))))
            count += len(groups)
        graph = CSR.from_pairs(np.concatenate(row_ids), np.concatenate(key_ids), count)
        support_index = np.concatenate(target_groups)
        index_seconds = time.perf_counter()-before
        geography = target_descriptor.new_tensor(np.concatenate(positions))
        condition = self.condition(torch.cat((target_descriptor[bands], geography), dim=-1))
        coarse = torch.cat((encoded.global_latents, encoded.regional))
        completed = self.global_read(condition, coarse, groups_graph(np.arange(count), graph),
                                     cfg.query_chunk, "decoder_coarse", metrics)
        delta = completed-condition
        if metrics is not None:
            metrics.tokens["coarse_readout_groups"] = count
            metrics.attention["decoder_coarse"]["group_index_seconds"] = index_seconds
        return SharedCoarseContext(support_index), (delta, delta.new_empty((0, cfg.decoder)))

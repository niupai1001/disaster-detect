"""Packed index routes. Only view filtering is repeated after geometry caching."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .frontend import attribute
from .geometry import CSR, expand, middle_map, overlaps, stencil


def priority(specs):
    """Round-robin physical types, with evenly spaced optical ranks first."""
    def key(spec):
        modality = attribute(spec, "modality")
        center = attribute(spec, "center_um")
        if modality == "optical" and center is not None and np.isfinite(center) and center > 0:
            return "optical"
        if modality == "sar":
            return "sar:" + str(attribute(spec, "polarization") or "unknown")
        return "unknown:" + str(modality)

    if not specs:
        return []
    groups = {}
    for i, spec in enumerate(specs):
        groups.setdefault(key(spec), []).append(i)
    orders = []
    for group in sorted(groups):
        indices = sorted(groups[group], key=lambda i: tuple(str(attribute(specs[i], field))
                         for field in ("quantity", "gx", "gy", "frequency_ghz", "incidence_deg", "source")))
        if group == "optical":
            indices.sort(key=lambda i: attribute(specs[i], "center_um"))
            # Endpoints, then successive bisections of spectral rank intervals.
            order = [indices[0]]
            if len(indices) > 1:
                order.append(indices[-1])
            ranges = [(0, len(indices)-1)]
            while ranges:
                finer = []
                for lo, hi in ranges:
                    if hi-lo > 1:
                        mid = (lo+hi)//2
                        order.append(indices[mid])
                        finer.extend(((lo, mid), (mid, hi)))
                ranges = finer
            indices = order
        orders.append(indices)
    return [indices[i] for i in range(max(map(len, orders))) for indices in orders if i < len(indices)]


@dataclass
class NativeIndex:
    grids: tuple
    ids: tuple
    offsets: np.ndarray
    packed: tuple

    @classmethod
    def build(cls, grids, masks):
        ids = tuple(np.flatnonzero(mask.detach().cpu().numpy().ravel()) for mask in masks)
        offsets = np.r_[0, np.cumsum([len(a) for a in ids])]
        lookup = []
        for grid, idx, offset in zip(grids, ids, offsets):
            back = np.full(grid.size, -1, dtype=np.int32)
            back[idx] = np.arange(len(idx))+offset
            lookup.append(back)
        return cls(tuple(grids), ids, offsets, tuple(lookup))

    def slices(self, start, end):
        first = int(np.searchsorted(self.offsets[1:], start, side="right"))
        for c in range(first, len(self.grids)):
            lo, hi = max(start, self.offsets[c]), min(end, self.offsets[c+1])
            if lo >= end:
                break
            if hi > lo:
                yield c, lo-start, self.ids[c][lo-self.offsets[c]:hi-self.offsets[c]]


def native_graph(query, visible, specs, cap, mode, metrics=None):
    active_channels = [c for c, ids in enumerate(visible.ids) if len(ids)]
    order = [active_channels[i] for i in priority([specs[c] for c in active_channels])]
    grid_visibility = {}
    if mode in ("cross", "decoder"):
        for c in active_channels:
            grid = visible.grids[c]
            if grid not in grid_visibility:
                grid_visibility[grid] = visible.packed[c] >= 0
            else:
                np.logical_or(grid_visibility[grid], visible.packed[c] >= 0, out=grid_visibility[grid])
        # Dense coverage cannot benefit enough from a separate union-overlap pass.
        grid_visibility = {grid: union for grid, union in grid_visibility.items()
                           if np.count_nonzero(union) < .9*grid.size}
    def graph(start, end, measure=False):
        rows, keys = [], []
        for c, shift, ids in query.slices(start, end):
            grid = query.grids[c]
            degree = np.zeros(len(ids), dtype=np.int16)
            if mode in ("space", "decoder"):
                q, k = stencil(ids, grid)
                packed = visible.packed[c][k]
                active = packed >= 0
                rows.append(q[active] + shift)
                keys.append(packed[active])
            if mode in ("cross", "decoder"):
                possible = {}
                for other_grid, union in grid_visibility.items():
                    query_rows, pixels = overlaps(grid, other_grid, ids)
                    hit = np.zeros(len(ids), dtype=bool)
                    hit[query_rows[union[pixels]]] = True
                    possible[other_grid] = hit
                for other in order:
                    if other == c:
                        continue
                    remaining = np.flatnonzero((degree < cap) & possible.get(visible.grids[other], True))
                    if not len(remaining):
                        continue
                    q, k = overlaps(grid, visible.grids[other], ids[remaining])
                    packed = visible.packed[other][k]
                    active = packed >= 0
                    q = remaining[q[active]]
                    rows.append(q+shift)
                    keys.append(packed[active])
                    # Overlap CSR is grouped by query.
                    degree[q] += 1
                if measure and metrics is not None and mode == "cross":
                    hist = np.bincount(degree)
                    for d, count in enumerate(hist):
                        if count:
                            metrics.channel_degree[d] += int(count)
        return join(rows), join(keys)
    return graph


def join(values):
    return np.concatenate(values) if values else np.empty(0, dtype=np.int64)


def groups_graph(group_ids, groups):
    """Repeat each group's key list for all queries assigned to it."""
    def graph(start, end):
        return groups.take(group_ids[start:end])
    return graph


def grouped_keys(groups):
    groups = np.asarray(groups)
    return CSR.from_pairs(groups, np.arange(len(groups)), int(groups.max())+1)


def expand_latents(indices, count):
    return (indices[:, None]*count + np.arange(count)).ravel()


def middle_spatial(cells, channels, count):
    lookup = {(*cell, c): i for i, (cell, c) in enumerate(zip(cells, channels))}
    neighbors, query = [], []
    for i, ((x, y), c) in enumerate(zip(cells, channels)):
        keys = [lookup[(x+dx, y+dy, c)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (x+dx, y+dy, c) in lookup]
        neighbors.extend(expand_latents(np.asarray(keys), count))
        query.extend([i]*len(keys)*count)
    return groups_graph(np.repeat(np.arange(len(cells)), count),
                        CSR.from_pairs(np.asarray(query), np.asarray(neighbors), len(cells)))


@lru_cache(maxsize=24)
def native_to_middle(grid, step):
    mapping = middle_map(grid, step)
    # Pieces split at the dateline can map to the same cell only at degenerate poles.
    return CSR.from_pairs(mapping.pixels, mapping.cell_index, grid.size)


def target_groups(query, step, lookup, per_side=None):
    """Sparse target pixel to visible latent-group IDs, keeping all support crossings."""
    def graph(start, end):
        rows, keys = [], []
        for c, shift, ids in query.slices(start, end):
            grid = query.grids[c]
            mapping = middle_map(grid, step)
            src, dst = native_to_middle(grid, step).take(ids)
            # Translate each unique geography once, rather than once per target.
            cells = mapping.cells if per_side is None else mapping.cells//per_side
            group = lookup(grid, cells)[dst]
            keep = group >= 0
            rows.append(src[keep]+shift)
            keys.append(group[keep])
        source, dest = join(rows), join(keys)
        if per_side is not None and len(source):
            pair = np.unique(np.column_stack((source, dest)), axis=0)
            source, dest = pair.T
        return source, dest
    return graph


def combine(first, second, offset, nlatent, global_count=0):
    def graph(start, end):
        q1, k1 = first(start, end)
        q2, k2 = second(start, end)
        return (np.concatenate((q1, np.repeat(q2, nlatent),
                                np.repeat(np.arange(end-start), global_count))),
                np.concatenate((k1, offset+expand_latents(k2, nlatent),
                                np.tile(np.arange(global_count), end-start))))
    return graph


def fixed_lookup(index):
    cached = {}
    def lookup(grid, cells):
        if grid not in cached:
            cached[grid] = np.asarray([index.get(tuple(cell), -1) for cell in cells], dtype=np.int32)
        return cached[grid]
    return lookup

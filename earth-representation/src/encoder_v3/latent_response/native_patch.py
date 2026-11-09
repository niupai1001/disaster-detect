"""Exact native-cell -> geographic Sub-Patch / 4x4 micro-cell geometry.

Geometry never reads reflectance. A Sub-Patch has size ``8*g_ref`` and is split
into ``4x4`` micro-cells; every native cell contributes to **every** micro-cell it
intersects with positive area, not only to the patch that contains its centre.
Axes ascend with world coordinates, so micro-cell row 0 is the smallest-y strip.
"""
import math
import torch

SUB_PATCH_CELLS = 8
MICRO = 4
MAX_EDGES_PER_CELL = 256  # resource guard: a cell spanning more is rejected, never silently cropped


def reference_gsd(grids):
    """g_ref = min over the complete native catalog geometry, independent of visibility."""
    values = [value for grid in grids for value in grid.gsd_xy]
    if not values:
        raise ValueError('At least one native grid required for the reference scale')
    g_ref = min(values)
    if not math.isfinite(g_ref) or g_ref <= 0:
        raise ValueError('Positive finite reference GSD required')
    return g_ref


def subpatch_size(grids):
    return SUB_PATCH_CELLS * reference_gsd(grids)


def microcell_index(centers, *, origin_xy, patch_size, micro=MICRO):
    """Centre-assigned patch/micro indices. Not the area aggregation rule."""
    if centers.ndim != 2 or centers.shape[1] != 2 or not centers.is_floating_point():
        raise ValueError('Floating [N,2] native cell centres required')
    if len(origin_xy) != 2 or not all(math.isfinite(float(value)) for value in origin_xy):
        raise ValueError('Finite two-term geographic origin required')
    if not math.isfinite(patch_size) or patch_size <= 0 or type(micro) is not int or micro < 1:
        raise ValueError('Positive finite Sub-Patch scale and positive micro grid required')
    origin = centers.new_tensor([float(value) for value in origin_xy])
    patch = torch.floor((centers - origin) / patch_size).to(torch.long)
    local = centers - origin - patch.to(centers.dtype) * patch_size
    cell = patch_size / micro
    micro_rc = torch.floor(local / cell).clamp(0, micro - 1).to(torch.long)
    return patch, micro_rc


def aggregate_microcells(features, cells, keep, *, origin_xy, patch_size, micro=MICRO):
    """Area-weighted visible aggregation into micro-cells per geographic Sub-Patch.

    features [N,d] are per-channel native features; cells [N,4] exact native boxes
    (xmin,ymin,xmax,ymax); keep [N] = valid & visible. Each native cell is assigned
    to every micro-cell whose box has positive-area intersection with it, so coarse
    cells and offset grids may feed several Sub-Patches. Returns per-Sub-Patch
    U [P,16,d], support [P,16] = visible overlap / micro-cell area, and
    count [P,16] = number of visible native cells touching that micro-cell.
    Fully empty micro-cells keep U=0, support=0, count=0 instead of a fake value.
    """
    if features.ndim != 2 or not features.is_floating_point() or len(features) == 0:
        raise ValueError('Nonempty floating [N,d] native features required')
    count_cells, width = features.shape
    if cells.shape != (count_cells, 4) or not cells.is_floating_point():
        raise ValueError('Exact [N,4] native cell boxes required')
    if keep.shape != (count_cells,) or keep.dtype is not torch.bool:
        raise ValueError('Boolean keep mask per native cell required')
    if len(origin_xy) != 2 or not all(math.isfinite(float(value)) for value in origin_xy):
        raise ValueError('Finite two-term geographic origin required')
    if not math.isfinite(patch_size) or patch_size <= 0 or type(micro) is not int or micro < 1:
        raise ValueError('Positive finite Sub-Patch scale and positive micro grid required')
    if not bool(torch.isfinite(cells).all()):
        raise ValueError('Finite native cell boxes required')
    if not bool((cells[:, 2:] > cells[:, :2]).all()):
        raise ValueError('Positive-area native cell boxes required')
    if not bool(keep.any()):
        raise ValueError('No visible native cell; the caller must skip this channel')
    if not bool(torch.isfinite(features[keep]).all()):
        raise ValueError('Nonfinite visible native feature')

    cell = patch_size / micro
    origin = cells.new_tensor([float(value) for value in origin_xy])
    low = torch.floor((cells[:, :2] - origin) / cell).to(torch.long)
    high = torch.ceil((cells[:, 2:] - origin) / cell).to(torch.long) - 1
    spans = high - low + 1
    if int(spans.prod(-1).max()) > MAX_EDGES_PER_CELL:
        raise ValueError('Native cell spans too many micro-cells; refusing an unbounded geometry expansion')
    lengths = spans.prod(-1)
    edges = int(lengths.sum())
    index = torch.repeat_interleave(torch.arange(count_cells, device=features.device), lengths)
    starts = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
    offset = torch.arange(edges, device=features.device) - starts
    nx = spans[:, 0][index]
    i = low[:, 0][index] + offset % nx
    j = low[:, 1][index] + offset // nx
    x0 = origin[0] + i.to(cells.dtype) * cell
    y0 = origin[1] + j.to(cells.dtype) * cell
    box = torch.stack((x0, y0, x0 + cell, y0 + cell), -1)
    overlap = (torch.minimum(cells[index, 2:], box[:, 2:]) - torch.maximum(cells[index, :2], box[:, :2])).clamp_min(0).prod(-1)
    positive = (overlap > 0) & keep[index]
    patch_y = torch.div(j, micro, rounding_mode='floor')
    patch_x = torch.div(i, micro, rounding_mode='floor')
    slot = (j - patch_y * micro) * micro + (i - patch_x * micro)
    keys, inverse = torch.unique(torch.stack((patch_x, patch_y), -1), dim=0, return_inverse=True)
    patches, slots = len(keys), micro * micro
    destination = inverse * slots + slot
    area = torch.where(positive, overlap, overlap.new_zeros(())).to(features.dtype)
    visible = torch.where(keep[:, None], features, features.new_zeros(()))
    support = features.new_zeros(patches * slots).index_add_(0, destination, area).view(patches, slots)
    counts = torch.zeros(patches * slots, dtype=torch.long, device=features.device).index_add_(
        0, destination, positive.to(torch.long)).view(patches, slots)
    total = features.new_zeros((patches * slots, width)).index_add_(
        0, destination, area[:, None] * visible[index]).view(patches, slots, width)
    micro_area = torch.as_tensor(cell * cell, dtype=features.dtype, device=features.device)
    support = support / micro_area
    total = total / (support * micro_area).clamp_min(torch.finfo(features.dtype).tiny)[:, :, None]
    return dict(patch=keys, U=total, support=support, count=counts)

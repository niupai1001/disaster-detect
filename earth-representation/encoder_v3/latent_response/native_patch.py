"""Exact native-cell -> geographic Sub-Patch / 4x4 micro-cell geometry.

Geometry never reads reflectance. A Sub-Patch has size ``8*g_ref`` and is split
into ``4x4`` micro-cells; a native cell contributes by its true overlap area.
Axes ascend with world coordinates, so micro-cell row 0 is the smallest-y strip.
"""
import math
import torch

SUB_PATCH_CELLS = 8
MICRO = 4


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
    (xmin,ymin,xmax,ymax); keep [N] = valid & visible. Returns per-Sub-Patch
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
    if not bool(keep.any()):
        raise ValueError('No visible native cell; the caller must skip this channel')
    if not bool(torch.isfinite(features[keep]).all()):
        raise ValueError('Nonfinite visible native feature')
    centers = (cells[:, :2] + cells[:, 2:]) * 0.5
    patch, _ = microcell_index(centers, origin_xy=origin_xy, patch_size=patch_size, micro=micro)
    keys, inverse = torch.unique(patch, dim=0, return_inverse=True)
    patches, slots = len(keys), micro * micro
    cell = patch_size / micro
    base = keys.to(cells.dtype) * patch_size + cells.new_tensor([float(value) for value in origin_xy])
    steps = torch.arange(micro, device=cells.device, dtype=cells.dtype)
    x0 = base[:, 0][:, None] + steps[None, :] * cell
    y0 = base[:, 1][:, None] + steps[None, :] * cell
    x_low = x0[:, None, :].expand(patches, micro, micro)
    y_low = y0[:, :, None].expand(patches, micro, micro)
    boxes = torch.stack((x_low, y_low, x_low + cell, y_low + cell), -1).reshape(patches, slots, 4)
    visible = torch.where(keep[:, None], features, features.new_zeros(()))
    micro_area = torch.as_tensor(cell * cell, dtype=features.dtype, device=features.device)
    support = features.new_zeros((patches, slots))
    counts = torch.zeros((patches, slots), dtype=torch.long, device=features.device)
    total = features.new_zeros((patches, slots, width))
    for slot in range(slots):
        target = boxes[:, slot][inverse]
        overlap = (torch.minimum(cells[:, 2:], target[:, 2:]) - torch.maximum(cells[:, :2], target[:, :2])).clamp_min(0).prod(-1)
        overlap = torch.where(keep, overlap, overlap.new_zeros(())).to(features.dtype)
        total[:, slot].index_add_(0, inverse, overlap[:, None] * visible)
        support[:, slot].index_add_(0, inverse, overlap)
        counts[:, slot].index_add_(0, inverse, (overlap > 0).to(torch.long))
    support = support / micro_area
    total = total / (support * micro_area).clamp_min(torch.finfo(features.dtype).tiny)[:, :, None]
    return dict(patch=keys, U=total, support=support, count=counts)

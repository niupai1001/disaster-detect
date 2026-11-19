"""Resident scenes and one vectorized loss reduction per native tile."""
from functools import lru_cache

import torch

from .whole_scene import tile_bounds, tiles


@lru_cache(maxsize=64)
def band_slots(layout):
    widths = torch.tensor([h*w for _, h, w in layout])
    return tuple(widths.tolist()), torch.repeat_interleave(torch.arange(len(layout)), widths)


def prepared_backward(model, cpu_scene, scene, views, backward_scale, *, reconstruct_all=False):
    """Use the original masks, tiles and parent/band weights with fewer device synchronizations."""
    widths, band_index = band_slots(scene.layout)
    bounds = tile_bounds(scene)
    losses = []
    for task_index, (keep, bands) in enumerate(views):
        if reconstruct_all:
            hidden = True
        elif task_index == 0:
            hidden = ~keep[:, None] if keep.ndim == 1 else ~keep
        else:
            hidden = ~bands[band_index][None]
        targets = cpu_scene.qa & hidden
        counts = torch.stack([q.sum() for q in targets.split(widths, -1)])
        active = int((counts > 0).sum())
        if not active:
            raise ValueError('No valid native targets')
        device = scene.values.device
        weights = (counts.clamp_min(1).float().reciprocal() / active)[band_index].to(device)
        bands = bands.to(device)
        keep = keep.to(device)
        target_tiles = targets.any(-1).reshape(scene.grid)
        total = scene.values.new_zeros(())
        for (tile, visible, _), (r0, r1, c0, c1) in zip(tiles(scene, keep), bounds, strict=True):
            if not target_tiles[r0:r1, c0:c1].any():
                continue
            structural = tile.qa.any(-1)
            if reconstruct_all:
                hidden = True
            elif task_index == 0:
                hidden = ~visible[:, None] if visible.ndim == 1 else ~visible
            else:
                hidden = ~bands[band_index.to(device)][None]
            support = (tile.qa & hidden)[structural]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(tile, visible, bands)
            error = torch.where(support, prediction.float() - tile.values[structural], 0.)
            loss = (error.square() * weights).sum()
            if backward_scale is not None:
                (loss * backward_scale).backward()
            total += loss.detach()
        losses.append(total)
    return torch.stack(losses)

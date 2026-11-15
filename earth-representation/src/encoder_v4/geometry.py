"""Scene-relative work grids and exact rectangular footprint intersections."""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from types import SimpleNamespace

import numpy as np
from rasterio.crs import CRS
import torch

from encoder_v3.hierarchical.geometry import Partition, area_map


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int]
    affine: tuple[float, ...]
    crs: str

    @property
    def spacing(self):
        return self.affine[0], -self.affine[4]


@dataclass(frozen=True)
class WorkGrid:
    shape: tuple[int, int]
    origin: tuple[float, float]
    spacing: tuple[float, float]
    valid: torch.Tensor

    def merge(self):
        h, w = self.shape
        hh, ww = (h + 1) // 2, (w + 1) // 2
        valid = torch.nn.functional.pad(self.valid, (0, w % 2, 0, h % 2))
        valid = valid.reshape(hh, 2, ww, 2).any(dim=(1, 3))
        return WorkGrid((hh, ww), self.origin, tuple(2 * d for d in self.spacing), valid)


@lru_cache(maxsize=64)
def footprint_map(grid, shape, origin, spacing, device):
    h, w = shape
    dx, dy = spacing
    x, y = np.arange(w + 1) * dx, np.arange(h + 1) * dy
    part = Partition(x, y, None, np.full(h * w, dx * dy), None)
    return area_map(grid, SimpleNamespace(origin=origin), part, device)


@lru_cache(maxsize=64)
def build_work_grid(grids, stride, device):
    crs = CRS.from_user_input(grids[0].crs)
    if not crs.is_projected or abs(crs.linear_units_factor[1] - 1) > 1e-9:
        raise ValueError('Expected a projected metre CRS')
    for grid in grids:
        a, b, _, d, e, _ = grid.affine
        if (CRS.from_user_input(grid.crs) != crs or b != 0 or d != 0 or a <= 0 or e >= 0
                or not np.isfinite(grid.affine).all() or min(grid.shape) < 1):
            raise ValueError('Expected north-up axis-aligned grids with a common metre CRS')
    dx, dy = (min(g.spacing[i] for g in grids) * stride for i in range(2))
    left = min(g.affine[2] for g in grids)
    bottom = min(g.affine[5] - g.spacing[1] * g.shape[0] for g in grids)
    right = max(g.affine[2] + g.spacing[0] * g.shape[1] for g in grids)
    top = max(g.affine[5] for g in grids)
    h, w = max(1, ceil((top - bottom) / dy)), max(1, ceil((right - left) / dx))
    origin = (left, bottom)
    maps = {grid: footprint_map(grid, (h, w), origin, (dx, dy), device)
            for grid in dict.fromkeys(grids)}
    valid = torch.zeros(h * w, dtype=torch.bool, device=device)
    for mapping in maps.values():
        valid[mapping.destination[mapping.area > 0]] = True
    return WorkGrid((h, w), origin, (dx, dy), valid.reshape(h, w)), maps

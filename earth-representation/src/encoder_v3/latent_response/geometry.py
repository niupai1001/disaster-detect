"""Immutable complete-observation geometry; masks never change its frame."""
from dataclasses import dataclass
import math
import torch
from .native_patch import subpatch_size


@dataclass(frozen=True)
class FullGeometry:
    grids: tuple
    origin: tuple
    extent: tuple
    patch_size: float
    shape: tuple

    @classmethod
    def from_channels(cls, channels):
        if not channels:
            raise ValueError('Complete loaded-channel geometry required')
        grids = tuple(c.grid for c in channels)
        from rasterio.crs import CRS
        if len({CRS.from_user_input(g.crs) for g in grids}) != 1:
            raise ValueError('Observation channels must share one metric CRS')
        boxes = [(g.affine[2], g.affine[5]+g.affine[4]*g.shape[0],
                  g.affine[2]+g.affine[0]*g.shape[1], g.affine[5]) for g in grids]
        if max(b[0] for b in boxes) >= min(b[2] for b in boxes) or max(b[1] for b in boxes) >= min(b[3] for b in boxes):
            raise ValueError('Channel supports have no common observation footprint')
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        w, h = max(b[2] for b in boxes)-x0, max(b[3] for b in boxes)-y0
        side = subpatch_size(grids)
        return cls(grids, (x0, y0), (w, h), side, (math.ceil(h/side), math.ceil(w/side)))

    def check_channels(self, channels):
        if any(c.grid not in self.grids for c in channels):
            raise ValueError('Input geometry absent from full_geometry')

    def patch_centers(self, device):
        y, x = torch.meshgrid(torch.arange(self.shape[0], device=device),
                              torch.arange(self.shape[1], device=device), indexing='ij')
        index = torch.stack((x.flatten(), y.flatten()), -1)
        return index, (index.double()+.5)*self.patch_size + index.new_tensor(self.origin, dtype=torch.float64)


def native_cells(grid, device, begin=0, end=None):
    end = grid.shape[0] if end is None else end
    y, x = torch.meshgrid(torch.arange(begin, end, device=device, dtype=torch.float64),
                          torch.arange(grid.shape[1], device=device, dtype=torch.float64), indexing='ij')
    a, _, x0, _, e, y0 = grid.affine
    left, top = x0+x*a, y0+y*e
    return torch.stack((left, top+e, left+a, top), -1).reshape(-1, 4)


def position_features(points, geometry):
    d = points.double()-points.new_tensor(geometry.origin, dtype=torch.float64)
    periods = 2.**torch.arange(12, device=points.device, dtype=torch.float64)
    phase = 2*math.pi*d[..., :, None]/periods
    return torch.cat((phase.sin().flatten(-2), phase.cos().flatten(-2),
                      d/d.new_tensor(geometry.extent)), -1).float()


def fourier2d(points, bands=4):
    phase = 2*math.pi*points[..., :, None]*(2.**torch.arange(bands, device=points.device))
    return torch.cat((phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)

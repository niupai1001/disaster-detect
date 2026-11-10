"""Nested metric partitions and exact native area maps, independent of QA/views."""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
import torch
from scipy.spatial import cKDTree
from encoder_v3.shared_mae.geometry import metric_geometry


@dataclass
class Partition:
    x: np.ndarray
    y: np.ndarray
    centers: np.ndarray
    area: np.ndarray
    patch: np.ndarray
    parent: np.ndarray | None = None

    def __len__(self):
        return len(self.area)


def edges(extent, patch, step):
    parts = []
    for lo in np.arange(0., extent, patch):
        hi = min(extent, lo + patch)
        parts.extend(np.arange(lo, hi - 1e-8, step).tolist())
    return np.array(parts + [extent], dtype=np.float64)


def locate(points, x, y):
    ix = np.searchsorted(x, points[:, 0], side='right') - 1
    iy = np.searchsorted(y, points[:, 1], side='right') - 1
    return iy * (len(x) - 1) + ix


@lru_cache(maxsize=64)
def partitions(grids, length):
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Explicit positive L required')
    channels = [type('GeometryChannel', (), {'grid': g})() for g in grids]
    geo = metric_geometry(channels, length)
    # Use finest loaded native support on each axis; channel order and masks do not matter.
    gx, gy = (min(g.gsd_xy[i] for g in grids) for i in (0, 1))
    result = []
    for factor in (2., 4., None):
        x = edges(geo.extent[0], length, length if factor is None else gx * factor)
        y = edges(geo.extent[1], length, length if factor is None else gy * factor)
        xx, yy = np.meshgrid((x[:-1]+x[1:])/2, (y[:-1]+y[1:])/2)
        centers = np.stack((xx.ravel(), yy.ravel()), -1)
        area = (np.diff(y)[:, None]*np.diff(x)[None]).ravel()
        patch = (centers[:, 1] // length).astype(np.int64)*geo.shape[1]+(centers[:, 0] // length).astype(np.int64)
        result.append(Partition(x, y, centers, area, patch))
    for child, parent in zip(result, result[1:]):
        child.parent = locate(child.centers, parent.x, parent.y)
    return geo, tuple(result), (gx, gy)


@dataclass
class AreaMap:
    source: torch.Tensor
    destination: torch.Tensor
    area: torch.Tensor
    region_area: torch.Tensor
    pixels: int
    pixel_area: float

    def aggregate(self, feature, keep):
        if feature.ndim==2:
            values,ratio=self.aggregate(feature[None],keep.flatten()[None])
            return values[0],ratio[0]
        weight = self.area[None] * keep.flatten(1)[:,self.source]
        total = feature.new_zeros(feature.shape[0],len(self.region_area),feature.shape[-1])
        total.index_add_(1,self.destination,feature[:,self.source]*weight[...,None])
        support=weight.new_zeros(feature.shape[0],len(self.region_area)).index_add_(1,self.destination,weight)
        return total/support.clamp_min(1e-12)[...,None],support/self.region_area

    def support(self,keep):
        weight=self.area*keep.flatten()[self.source]
        return weight.new_zeros(len(self.region_area)).index_add_(0,self.destination,weight)/self.region_area

    def native(self, field):
        return field.new_zeros(self.pixels).index_add_(0, self.source, field[self.destination]*self.area/self.pixel_area)

    def hidden(self, region_mask):
        count = torch.zeros(self.pixels, device=self.source.device, dtype=torch.int32)
        return count.index_add_(0, self.source, region_mask[self.destination].int()) > 0


def area_map(grid, geo, part, device):
    # Build in scene-relative float64 coordinates to preserve fractional overlaps.
    h, w = grid.shape
    a, _, ox, _, e, oy = grid.affine
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    left = (ox-geo.origin[0]+xx*a).ravel()
    bottom = (oy-geo.origin[1]+(yy+1)*e).ravel()
    right, top = left+a, bottom-e
    lowx = np.searchsorted(part.x, left+1e-8, side='right')-1
    highx = np.searchsorted(part.x, right-1e-8, side='left')-1
    lowy = np.searchsorted(part.y, bottom+1e-8, side='right')-1
    highy = np.searchsorted(part.y, top-1e-8, side='left')-1
    nx, ny = highx-lowx+1, highy-lowy+1
    lengths = nx*ny
    if np.any(lengths < 1) or lengths.max() > 4096:
        raise ValueError('Unsupported footprint expansion')
    source = np.repeat(np.arange(h*w), lengths)
    offset = np.arange(lengths.sum())-np.repeat(np.cumsum(lengths)-lengths, lengths)
    ix, iy = lowx[source]+offset % nx[source], lowy[source]+offset // nx[source]
    if min(ix.min(), iy.min()) < 0 or ix.max() >= len(part.x)-1 or iy.max() >= len(part.y)-1:
        raise ValueError('Native support outside immutable partition')
    area = np.maximum(0., np.minimum(right[source],part.x[ix+1])-np.maximum(left[source],part.x[ix]))
    area *= np.maximum(0., np.minimum(top[source],part.y[iy+1])-np.maximum(bottom[source],part.y[iy]))
    destination = iy*(len(part.x)-1)+ix
    tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)
    return AreaMap(tensor(source,torch.long),tensor(destination,torch.long),tensor(area,torch.float32),
                   tensor(part.area,torch.float32),h*w,a*(-e))


def neighborhood(part, radius_xy, device):
    """Sparse geometric neighbors, never a dense fine-token attention matrix."""
    points = part.centers / np.array(radius_xy)
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, 1., p=np.inf, return_sorted=True)
    lengths = np.fromiter(map(len, neighbors), dtype=np.int64)
    q = np.repeat(np.arange(len(part)), lengths)
    k = np.concatenate(neighbors).astype(np.int64)
    return sparse_edges(np.stack((q,k)),len(part),device)


def sparse_edges(index, queries, device):
    order=np.argsort(index[0],kind='stable')
    index=index[:,order]
    chunks=[]
    for begin in range(0,queries,4096):
        end=min(queries,begin+4096)
        lo,hi=np.searchsorted(index[0],[begin,end])
        local=index[:,lo:hi].copy()
        local[0]-=begin
        counts=np.bincount(local[0],minlength=end-begin)
        capacity=max(1,int(counts.max()))
        if queries>100000:
            chunks.append((begin,end,torch.tensor(local,device=device),None,capacity))
            continue
        neighbors=np.zeros((end-begin,capacity),dtype=np.int64)
        present=np.zeros_like(neighbors,dtype=bool)
        if len(local[0]):
            slot=np.arange(len(local[0]))-np.repeat(np.cumsum(counts)-counts,counts)
            neighbors[local[0],slot]=local[1]
            present[local[0],slot]=True
        # The CPU edge list is provenance; padded GPU neighbors avoid ragged mask synchronization.
        chunks.append((begin,end,torch.tensor(local),torch.tensor(neighbors,device=device),torch.tensor(present,device=device)))
    return chunks

"""Mask-independent native supports and globally anchored Equal Earth cells."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from pyproj import CRS, Transformer, proj_version_str
import shapely


@lru_cache(maxsize=1)
def geometry_version():
    return (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            proj_version_str, shapely.geos_version_string)


def persistent_geometry(kind, identity, cls, build):
    directory = os.environ.get("V3_H_R_GEOMETRY_CACHE")
    if not directory:
        return build()
    digest = hashlib.sha256(repr((geometry_version(), kind, identity)).encode()).hexdigest()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root/(digest + ".npz")
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            return cls(**{key: payload[key] for key in payload.files})
    value = build()
    with tempfile.NamedTemporaryFile(dir=root, suffix=".npz", delete=False) as stream:
        temp = Path(stream.name)
        try:
            np.savez(stream, **vars(value))
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        if not path.exists():
            os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return value


def expand(counts):
    """Return row and within-row index for variable-length contiguous ranges."""
    ends = counts.cumsum()
    rows = np.repeat(np.arange(len(counts)), counts)
    return rows, np.arange(len(rows)) - (ends - counts)[rows]


@dataclass(frozen=True)
class CSR:
    offsets: np.ndarray
    keys: np.ndarray

    @classmethod
    def from_pairs(cls, query, keys, rows):
        order = np.argsort(query, kind="stable")
        offsets = np.r_[0, np.bincount(query, minlength=rows).cumsum()]
        return cls(offsets, keys[order].astype(np.int32, copy=False))

    def take(self, rows):
        starts, stops = self.offsets[rows], self.offsets[rows + 1]
        query, slot = expand(stops - starts)
        return query, self.keys[starts[query] + slot]


@dataclass(frozen=True)
class Grid:
    crs: str
    affine: tuple
    shape: tuple

    @classmethod
    def from_native(cls, grid):
        return cls(CRS.from_user_input(grid.crs).to_string(), tuple(grid.affine)[:6], tuple(grid.shape))

    @property
    def size(self):
        return math.prod(self.shape)

    def xy(self, x, y):
        a, b, c, d, e, f = self.affine
        return a*x + b*y + c, d*x + e*y + f


@lru_cache(maxsize=64)
def transformer(src, dst):
    return Transformer.from_crs(src, dst, always_xy=True)


def stencil(ids, grid):
    y, x = np.divmod(ids, grid.shape[1])
    dy, dx = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    yy, xx = y[:, None] + dy.ravel(), x[:, None] + dx.ravel()
    valid = (yy >= 0) & (yy < grid.shape[0]) & (xx >= 0) & (xx < grid.shape[1])
    q, slot = np.nonzero(valid)
    return q, yy[q, slot]*grid.shape[1] + xx[q, slot]


def boundary(grid, ids, divisions):
    u = np.arange(divisions)/divisions
    x = np.concatenate((u, np.ones_like(u), 1-u, np.zeros_like(u)))
    y = np.concatenate((np.zeros_like(u), u, np.ones_like(u), 1-u))
    row, col = np.divmod(ids, grid.shape[1])
    return grid.xy(col[:, None] + x, row[:, None] + y)


def projected_parts(grid, ids, divisions):
    """Return projected polygons and their input rows, splitting dateline crossings."""
    x, y = boundary(grid, ids, divisions)
    lon, lat = transformer(grid.crs, "EPSG:4326").transform(x, y)
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("Nonfinite native support transform")
    lon = (lon + 180) % 360 - 180
    ordinary = np.ptp(lon, axis=1) <= 180
    rows = np.flatnonzero(ordinary)
    project = transformer("EPSG:4326", "EPSG:8857")
    px, py = project.transform(lon[ordinary], lat[ordinary])
    pieces = [shapely.polygons(np.stack((px, py), axis=-1))]
    owners = [rows]
    for row in np.flatnonzero(~ordinary):
        unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon[row])))
        coordinates = np.column_stack((unwrapped, lat[row]))
        turns = round((unwrapped[-1]-unwrapped[0])/360)
        if turns:
            # A winding ring encloses a pole. Close through the polar limit
            # before clipping the ring into longitude slabs.
            close = unwrapped[0]+360*turns
            pole = math.copysign(90., float(lat[row].mean()))
            coordinates = np.vstack((coordinates, (close, lat[row, 0]),
                                      (close, pole), (unwrapped[0], pole)))
        polygon = shapely.make_valid(shapely.Polygon(coordinates))
        first, last = np.floor((np.array(polygon.bounds)[[0, 2]] + 180)/360).astype(int)
        for wrap in range(first, last + 1):
            cut = shapely.intersection(polygon, shapely.box(-180 + 360*wrap, -90, 180 + 360*wrap, 90))
            for part in shapely.get_parts(cut):
                if part.geom_type != "Polygon" or part.area <= 0:
                    continue
                coords = np.asarray(part.exterior.coords)
                px, py = project.transform(coords[:, 0] - 360*wrap, coords[:, 1])
                pieces.append(np.array([shapely.Polygon(np.column_stack((px, py)))]))
                owners.append(np.array([row]))
    return np.concatenate(pieces), np.concatenate(owners)


def projected_support(grid, ids):
    """Refine until per-pixel relative area change is below 1e-5."""
    previous = None
    for divisions in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        pieces, owners = projected_parts(grid, ids, divisions)
        areas = np.bincount(owners, weights=shapely.area(pieces), minlength=len(ids))
        if previous is not None:
            error = float(np.max(np.abs(areas - previous)/np.maximum(areas, 1e-12)))
            if error <= 1e-5 and np.all(areas > 0) and shapely.is_valid(pieces).all():
                return pieces, owners, error
        previous = areas
    raise ValueError(f"Equal Earth support did not converge: relative area change {error:g}")


@dataclass(frozen=True)
class MiddleMap:
    cells: np.ndarray
    pixels: np.ndarray
    cell_index: np.ndarray
    area: np.ndarray
    positions: np.ndarray
    pixel_areas: np.ndarray
    area_error: float


@lru_cache(maxsize=24)
def middle_map(grid, step):
    return persistent_geometry("middle", (grid, step), MiddleMap, lambda: build_middle_map(grid, step))


def build_middle_map(grid, step):
    pixels, cells, weights, positions = [], [], [], []
    worst = 0.
    for start in range(0, grid.size, 65536):
        ids = np.arange(start, min(grid.size, start + 65536))
        row, col = np.divmod(ids, grid.shape[1])
        lon, lat = transformer(grid.crs, "EPSG:4326").transform(*grid.xy(col+.5, row+.5))
        positions.append(np.column_stack((lon, lat)).astype(np.float32))
        pieces, owners, error = projected_support(grid, ids)
        worst = max(worst, error)
        bounds = shapely.bounds(pieces)
        low = np.floor(bounds[:, :2]/step).astype(np.int64)
        high = np.maximum(low, np.ceil(bounds[:, 2:]/step).astype(np.int64)-1)
        width = high[:, 0]-low[:, 0]+1
        rows, slots = expand(width*(high[:, 1]-low[:, 1]+1))
        xy = low[rows] + np.column_stack((slots % width[rows], slots // width[rows]))
        area = shapely.area(pieces)[rows]
        crossed = (low != high).any(axis=1)[rows]
        if crossed.any():
            lo = xy[crossed]*step
            hi = lo + step
            box = shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1])
            area[crossed] = shapely.area(shapely.intersection(pieces[rows[crossed]], box))
        valid = area > 0
        pixels.append(ids[owners[rows[valid]]].astype(np.int32))
        cells.append(xy[valid])
        weights.append(area[valid].astype(np.float32))
    coordinates, index = np.unique(np.concatenate(cells), axis=0, return_inverse=True)
    native = np.concatenate(pixels)
    area = np.concatenate(weights)
    pixel_areas = np.bincount(native, weights=area, minlength=grid.size).astype(np.float32)
    return MiddleMap(coordinates, native, index.astype(np.int32),
                     area, np.concatenate(positions), pixel_areas, worst)


def rect_pair(query, key, ids):
    """Exact positive-area intersection for north-up grids in the same CRS."""
    a, _, c, _, e, f = query.affine
    b, _, g, _, h, i = key.affine
    row, col = np.divmod(ids, query.shape[1])
    x0, x1 = (c-g + a*col)/b, (c-g + a*(col+1))/b
    y0, y1 = (f-i + e*row)/h, (f-i + e*(row+1))/h
    x0, x1, y0, y1 = [np.where(np.abs(v-np.rint(v)) < 1e-10, np.rint(v), v)
                       for v in (x0, x1, y0, y1)]
    lx, hx = np.maximum(0, np.floor(x0).astype(int)), np.minimum(key.shape[1], np.ceil(x1).astype(int))
    ly, hy = np.maximum(0, np.floor(y0).astype(int)), np.minimum(key.shape[0], np.ceil(y1).astype(int))
    widths = np.maximum(hx-lx, 0)
    q, slots = expand(widths*np.maximum(hy-ly, 0))
    return q, (ly[q]+slots//widths[q])*key.shape[1]+lx[q]+slots%widths[q]


def north_up(grid):
    a, b, _, d, e, _ = grid.affine
    return a > 0 and e < 0 and b == d == 0


@lru_cache(maxsize=24)
def overlap_map(query, key):
    if query == key:
        return None
    return persistent_geometry("cross", (query, key), CSR, lambda: build_overlap_map(query, key))


def build_overlap_map(query, key):
    if query.crs == key.crs and north_up(query) and north_up(key) and CRS(query.crs).is_projected:
        pairs = [rect_pair(query, key, np.arange(i, min(i+65536, query.size)))
                 for i in range(0, query.size, 65536)]
        q = np.concatenate([rows+i*65536 for i, (rows, _) in enumerate(pairs)])
        k = np.concatenate([cols for _, cols in pairs])
    else:
        same_plane = query.crs == key.crs and CRS(query.crs).is_projected
        def supports(grid, ids):
            if same_plane:
                # An affine pixel is exactly a parallelogram in the shared native plane.
                x, y = boundary(grid, ids, 1)
                return shapely.polygons(np.stack((x, y), axis=-1)), np.arange(len(ids))
            return projected_support(grid, ids)[:2]
        kp, ko = supports(key, np.arange(key.size))
        tree = shapely.STRtree(kp)
        rows, cols = [], []
        for start in range(0, query.size, 65536):
            qp, qo = supports(query, np.arange(start, min(start+65536, query.size)))
            qi, ki = tree.query(qp, predicate="intersects")
            positive = shapely.area(shapely.intersection(qp[qi], kp[ki])) > 0
            rows.append(start+qo[qi[positive]])
            cols.append(ko[ki[positive]])
        q, k = np.concatenate(rows), np.concatenate(cols)
        unique = np.unique(np.column_stack((q, k)), axis=0)
        q, k = unique.T
    return CSR.from_pairs(q, k, query.size)


def overlaps(query, key, ids):
    mapping = overlap_map(query, key)
    return (np.arange(len(ids)), ids) if mapping is None else mapping.take(ids)


def regional_ids(cells, per_side):
    return np.floor_divide(cells, per_side)

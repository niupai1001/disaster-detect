import numpy as np
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.transform import array_bounds

from src.data.adapters._base import patch_grids

CRS_UTM = CRS.from_epsg(32653)
SIZE, GSD = 512, 10


def _bounds(width, height, transform):
    return BoundingBox(*array_bounds(width, height, transform))


def test_grid_is_aligned_and_10m():
    scene_bounds = _bounds(10980, 10980, __import__("rasterio").transform.from_origin(
        300000, 3900000, 10, 10))
    grids = patch_grids([132.4, 34.0, 133.0, 34.6], scene_bounds, CRS_UTM,
                        size=SIZE, gsd=GSD)
    assert grids
    step = SIZE * GSD
    for t in grids:
        left, bottom, right, top = _bounds(SIZE, SIZE, t)
        assert abs(left % step) < 1e-6 and abs((top) % step) < 1e-6
        assert abs((right - left) - step) < 1e-6
        assert abs(t.a - GSD) < 1e-9


def test_grid_stays_inside_scene_footprint():
    scene = __import__("rasterio").transform.from_origin(300000, 3900000, 10, 10)
    bounds = _bounds(10980, 10980, scene)
    grids = patch_grids([132.0, 33.0, 134.0, 35.0], bounds, CRS_UTM)
    for t in grids:
        left, bottom, right, top = _bounds(SIZE, SIZE, t)
        assert left >= bounds.left - 1e-6 and right <= bounds.right + 1e-6
        assert bottom >= bounds.bottom - 1e-6 and top <= bounds.top + 1e-6


def test_no_overlap_returns_no_grid():
    scene = __import__("rasterio").transform.from_origin(300000, 3900000, 10, 10)
    bounds = _bounds(10980, 10980, scene)
    # AOI far to the west of the scene
    assert patch_grids([100.0, 20.0, 101.0, 21.0], bounds, CRS_UTM) == []


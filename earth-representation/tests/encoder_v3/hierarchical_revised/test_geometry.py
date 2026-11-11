from types import SimpleNamespace

import numpy as np
import torch
import shapely

from encoder_v3.hierarchical_revised.geometry import (Grid, middle_map, overlaps, projected_support,
                                                       regional_ids)
from encoder_v3.hierarchical_revised.routes import NativeIndex, native_graph


def test_exact_positive_cross_resolution_overlap():
    query = Grid("EPSG:32632", (20., 0., 500005., 0., -20., 5000020.), (1, 2))
    key = Grid("EPSG:32632", (10., 0., 500000., 0., -10., 5000020.), (2, 3))
    q, k = overlaps(query, key, np.array([0, 1]))
    assert set(zip(q, k)) == {(0, i) for i in range(6)} | {(1, 2), (1, 5)}
    far = Grid(key.crs, (10., 0., 600000., 0., -10., 5000020.), key.shape)
    assert not len(overlaps(query, far, np.array([0, 1]))[0])


def test_rotated_support_global_ids_and_cache():
    grid = Grid("EPSG:32736", (20., 4., 499993., 3., -20., 6000001.), (3, 5))
    mapping = middle_map(grid, 128)
    assert mapping is middle_map(grid, 128)
    assert np.any(mapping.cells[:, 1] < 0)
    assert np.array_equal(regional_ids(np.array([[-1, -4], [-5, 4], [3, 0]]), 4),
                          np.array([[-1, -1], [-2, 1], [0, 0]]))
    # The same physical pixel in a different crop keeps the same cell IDs.
    x, y = grid.xy(2, 1)
    crop = Grid(grid.crs, (20., 4., x, 3., -20., y), (1, 1))
    original = mapping.cells[mapping.cell_index[mapping.pixels == 7]]
    assert set(map(tuple, original)) == set(map(tuple, middle_map(crop, 128).cells))
    q, k = overlaps(crop, grid, np.array([0]))
    assert set(zip(q, k)) == {(0, 7)}


def test_dateline_split_and_polar_support():
    grid = Grid("EPSG:4326", (.02, 0., 179.99, 0., -.01, 30.), (1, 1))
    parts, owners, error = projected_support(grid, np.array([0]))
    assert len(parts) == 2 and set(owners) == {0}
    assert min(shapely.bounds(parts)[:, 0]) < 0 < max(shapely.bounds(parts)[:, 2])
    assert error <= 1e-5
    polar = Grid("EPSG:3413", (30., 0., 30000., 0., -30., 30000.), (2, 2))
    assert np.all(middle_map(polar, 128).area > 0)
    contains_pole = Grid("EPSG:3413", (30., 0., -15., 0., -30., 15.), (1, 1))
    parts, owners, error = projected_support(contains_pole, np.array([0]))
    assert set(owners) == {0} and error <= 1e-5
    assert 900 < shapely.area(parts).sum() < 1000


def test_native_pixel_at_middle_grid_corner_keeps_all_four_supports():
    grid = Grid("EPSG:8857", (20., 0., 118., 0., -20., 138.), (1, 1))
    mapping = middle_map(grid, 128)
    assert set(map(tuple, mapping.cells)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    np.testing.assert_allclose(mapping.area, 100., rtol=1e-5)
    np.testing.assert_allclose(mapping.pixel_areas, [400.], rtol=1e-5)


def test_persistent_maps_survive_eviction(tmp_path, monkeypatch):
    from encoder_v3.hierarchical_revised import geometry

    monkeypatch.setenv("V3_H_R_GEOMETRY_CACHE", str(tmp_path))
    grid = Grid("EPSG:32632", (20., 0., 500006., 0., -20., 5000020.), (3, 4))
    key = Grid(grid.crs, (10., 0., 500001., 0., -10., 5000020.), (4, 6))
    expected = geometry.middle_map(grid, 128)
    overlap = geometry.overlap_map(grid, key)
    geometry.middle_map.cache_clear()
    geometry.overlap_map.cache_clear()

    def unexpected(*args):
        raise AssertionError("Geometry was rebuilt instead of loaded")

    monkeypatch.setattr(geometry, "build_middle_map", unexpected)
    monkeypatch.setattr(geometry, "build_overlap_map", unexpected)
    actual = geometry.middle_map(grid, 128)
    np.testing.assert_array_equal(actual.cell_index, expected.cell_index)
    np.testing.assert_array_equal(actual.area, expected.area)
    np.testing.assert_array_equal(geometry.overlap_map(grid, key).keys, overlap.keys)


def test_union_visibility_pruning_matches_channel_by_channel_search():
    from encoder_v3.hierarchical_revised.routes import priority
    grid = Grid("EPSG:32632", (10., 0., 500000., 0., -10., 5000020.), (3, 4))
    coarse = Grid(grid.crs, (20., 0., 500000., 0., -20., 5000020.), (2, 2))
    grids = [grid, grid, coarse, coarse]
    generator = torch.Generator().manual_seed(24)
    visible = NativeIndex.build(grids, [torch.rand(g.shape, generator=generator) > .6 for g in grids])
    query = NativeIndex.build(grids, [torch.ones(g.shape, dtype=torch.bool) for g in grids])
    specs = [SimpleNamespace(modality="optical", center_um=.4+i*.1, source=str(i)) for i in range(4)]
    graph = native_graph(query, visible, specs, 2, "cross")
    rows, keys = graph(0, int(query.offsets[-1]))
    expected = set()
    active = [c for c in priority(specs) if len(visible.ids[c])]
    for channel, g in enumerate(grids):
        for pixel in range(g.size):
            used = 0
            for other in active:
                if channel == other:
                    continue
                _, ids = overlaps(g, grids[other], np.array([pixel]))
                packed = visible.packed[other][ids]
                packed = packed[packed >= 0]
                if len(packed):
                    expected.update((int(query.offsets[channel])+pixel,int(k)) for k in packed)
                    used += 1
                if used == 2:
                    break
    assert set(zip(rows, keys)) == expected


def test_224_channel_cap_and_metadata_order_without_values():
    grid = Grid("EPSG:32632", (20., 0., 500000., 0., -20., 5000020.), (1, 1))
    n = 224
    index = NativeIndex.build([grid]*n, [torch.ones((1, 1), dtype=torch.bool)]*n)
    specs = [SimpleNamespace(modality="optical", center_um=.4+i*.004, source=str(i)) for i in range(n)]
    q, k = native_graph(index, index, specs, 8, "cross")(0, n)
    assert np.all(np.bincount(q) == 8) and not np.any(q == k)
    other = list(reversed(specs))
    qr, kr = native_graph(index, index, other, 8, "cross")(0, n)
    assert set(zip(q, k)) == set(zip(n-1-qr, n-1-kr))

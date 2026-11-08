"""Exact Sub-Patch/micro-cell geometry: area conservation, support, masks, no hidden values."""
import unittest
import torch
from encoder_v3.latent_response import NativeGrid, reference_gsd, subpatch_size
from encoder_v3.latent_response.native_patch import MICRO, aggregate_microcells, microcell_index


def grid_cells(shape, affine):
    height, width = shape
    a, b, c, d, e, f = affine
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    x, y = x.flatten().double(), y.flatten().double()
    return torch.stack((c + x * a, f + (y + 1) * e, c + (x + 1) * a, f + y * e), -1)


def box(x0, y0, x1, y1):
    return [x0, y0, x1, y1]


def key_slot(result, key):
    index = int(torch.nonzero((result['patch'] == torch.tensor(key)).all(-1))[0, 0])
    return index


class ReferenceScaleTest(unittest.TestCase):
    def test_reference_gsd_uses_all_native_grids(self):
        grids = [NativeGrid((10, 0, 0, 0, -10, 0), (4, 4), 'EPSG:32733'),
                 NativeGrid((20, 0, 0, 0, -20, 0), (2, 2), 'EPSG:32733')]
        self.assertEqual(reference_gsd(grids), 10.0)
        self.assertEqual(subpatch_size(grids), 80.0)
        landsat = [NativeGrid((30, 0, 0, 0, -30, 0), (3, 3), 'EPSG:32639')]
        self.assertEqual(reference_gsd(landsat), 30.0)
        self.assertEqual(subpatch_size(landsat), 240.0)

    def test_empty_or_invalid_grid_rejected(self):
        with self.assertRaises(ValueError):
            reference_gsd([])
        with self.assertRaises(ValueError):
            NativeGrid((0, 0, 0, 0, -10, 0), (2, 2), 'EPSG:32733')
        with self.assertRaises(ValueError):
            NativeGrid((10, 0, 0, 0, 10, 0), (2, 2), 'EPSG:32733')

    def test_geographic_imperial_and_unknown_crs_rejected(self):
        for crs in ('EPSG:4326', 'EPSG:2263', 'not-a-crs', ''):
            with self.subTest(crs=crs), self.assertRaises(ValueError):
                NativeGrid((10, 0, 0, 0, -10, 0), (2, 2), crs)
        NativeGrid((10, 0, 0, 0, -10, 0), (2, 2), 'EPSG:32733')


class AggregateTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32)
        self.keep = torch.ones(4, dtype=torch.bool)

    def test_four_ten_metre_cells_fill_one_microcell(self):
        cells = grid_cells((2, 2), (10, 0, 0, 0, -10, 20))
        result = aggregate_microcells(self.features, cells, self.keep, origin_xy=(0, 0), patch_size=80.0)
        self.assertTrue(torch.equal(result['patch'], torch.tensor([[0, 0]])))
        self.assertAlmostEqual(float(result['support'].sum()), 1.0, places=6)
        self.assertEqual(int(result['count'].sum()), 4)
        self.assertTrue(torch.allclose(result['U'][0, 0], self.features.mean(0), atol=1e-3, rtol=1e-6))
        empty = result['support'][0] == 0
        self.assertTrue(torch.equal(result['U'][0][empty], torch.zeros_like(result['U'][0][empty])))

    def test_one_twenty_metre_cell_weights_by_true_overlap(self):
        cells = torch.tensor([box(10, 10, 30, 30)], dtype=torch.float64)
        result = aggregate_microcells(torch.ones(1, 32), cells, torch.ones(1, dtype=torch.bool),
                                      origin_xy=(0, 0), patch_size=80.0)
        self.assertEqual(int(result['count'].sum()), 4)
        expected = torch.zeros(MICRO * MICRO)
        expected[torch.tensor([0, 1, 4, 5])] = 0.25
        self.assertTrue(torch.allclose(result['support'][0], expected, atol=1e-6))

    def test_hidden_cell_contributes_nothing(self):
        cells = torch.tensor([box(0, 0, 10, 10), box(10, 0, 20, 10), box(0, 10, 10, 20),
                              box(10, 10, 20, 20)], dtype=torch.float64)
        features = self.features.clone()
        features[3] = float('nan')
        keep = torch.tensor([True, True, True, False])
        result = aggregate_microcells(features, cells, keep, origin_xy=(0, 0), patch_size=80.0)
        self.assertAlmostEqual(float(result['support'][0, 0]), .75, places=6)
        self.assertEqual(int(result['count'][0, 0]), 3)
        self.assertTrue(torch.isfinite(result['U']).all())
        self.assertTrue(torch.allclose(result['U'][0, 0], self.features[:3].mean(0), atol=1e-3, rtol=1e-6))

    def test_land_cover_of_one_microcell_by_four_thirty_metre_cells(self):
        cells = grid_cells((2, 2), (30, 0, 0, 0, -30, 60))
        result = aggregate_microcells(torch.ones(4, 32), cells, torch.ones(4, dtype=torch.bool),
                                      origin_xy=(0, 0), patch_size=240.0)
        self.assertAlmostEqual(float(result['support'][0, 0]), 1.0, places=6)
        self.assertEqual(int(result['count'][0, 0]), 4)

    def test_cell_crossing_a_sub_patch_boundary_contributes_to_both(self):
        # Review counterexample: area 4 spans patch 0 and patch 1.
        cells = torch.tensor([box(7, 0, 9, 2)], dtype=torch.float64)
        feature = torch.arange(32, dtype=torch.float32)[None]
        result = aggregate_microcells(feature, cells, torch.ones(1, dtype=torch.bool),
                                      origin_xy=(0, 0), patch_size=8.0)
        self.assertEqual({tuple(item.tolist()) for item in result['patch']}, {(0, 0), (1, 0)})
        self.assertAlmostEqual(float(result['support'].sum()), 1.0, places=6)
        self.assertEqual(int(result['count'].sum()), 2)
        for key in ((0, 0), (1, 0)):
            slot = key_slot(result, key)
            occupied = result['support'][slot] > 0
            self.assertTrue(torch.allclose(result['U'][slot][occupied], feature[0].expand(int(occupied.sum()), -1),
                                           atol=1e-6, rtol=0))

    def test_large_cell_covers_sixteen_patches(self):
        cells = torch.tensor([box(-1, -1, 17, 17)], dtype=torch.float64)
        result = aggregate_microcells(torch.ones(1, 32), cells, torch.ones(1, dtype=torch.bool),
                                      origin_xy=(0, 0), patch_size=8.0)
        self.assertEqual(len(result['patch']), 16)
        self.assertAlmostEqual(float(result['support'].sum()) * (8.0 / MICRO) ** 2, 18.0 * 18.0, places=6)
        self.assertEqual(int(result['count'].sum()), 100)

    def test_cells_split_across_patches_and_negative_indices(self):
        cells = torch.tensor([box(95, 95, 105, 105), box(85, 85, 95, 95)], dtype=torch.float64)
        result = aggregate_microcells(torch.ones(2, 32), cells, torch.ones(2, dtype=torch.bool),
                                      origin_xy=(100, 100), patch_size=80.0)
        self.assertEqual({tuple(item.tolist()) for item in result['patch']},
                         {(-1, -1), (-1, 0), (0, -1), (0, 0)})
        self.assertAlmostEqual(float(result['support'].sum()), 0.5, places=6)
        self.assertEqual(int(result['count'].sum()), 5)
        shared = key_slot(result, (-1, -1))
        self.assertEqual(int(result['count'][shared, 15]), 2)

    def test_area_conservation_on_random_geometry(self):
        generator = torch.Generator().manual_seed(20261108)
        for trial in range(50):
            count = int(torch.randint(1, 12, (1,), generator=generator))
            widths = torch.rand(count, generator=generator, dtype=torch.float64) * 24 + 1
            heights = torch.rand(count, generator=generator, dtype=torch.float64) * 24 + 1
            x0 = torch.rand(count, generator=generator, dtype=torch.float64) * 60 - 30
            y0 = torch.rand(count, generator=generator, dtype=torch.float64) * 60 - 30
            cells = torch.stack((x0, y0, x0 + widths, y0 + heights), -1)
            patch_size = 40.0
            result = aggregate_microcells(torch.ones(count, 4), cells, torch.ones(count, dtype=torch.bool),
                                          origin_xy=(-13.5, 7.25), patch_size=patch_size)
            recovered = float(result['support'].sum()) * (patch_size / MICRO) ** 2
            expected = float((widths * heights).sum())
            self.assertLess(abs(recovered / expected - 1.0), 1e-5, msg=f'trial {trial}')

    def test_feature_integral_and_gradient_follow_overlap_area(self):
        cells = torch.tensor([box(0, 0, 10, 10), box(0, 10, 10, 20)], dtype=torch.float64)
        features = torch.tensor([[2.0], [6.0]], dtype=torch.float32, requires_grad=True)
        result = aggregate_microcells(features, cells, torch.ones(2, dtype=torch.bool),
                                      origin_xy=(0, 0), patch_size=80.0)
        self.assertAlmostEqual(float(result['U'][0, 0, 0].detach()), 4.0, places=6)
        result['U'][0, 0, 0].backward()
        self.assertTrue(torch.allclose(features.grad[:, 0], torch.tensor([0.5, 0.5]), atol=1e-6))

    def test_masked_feature_integral_ignores_hidden_values(self):
        cells = torch.tensor([box(0, 0, 10, 10), box(10, 0, 20, 10)], dtype=torch.float64)
        features = torch.tensor([[1.0], [float('nan')]], dtype=torch.float32)
        result = aggregate_microcells(features, cells, torch.tensor([True, False]),
                                      origin_xy=(0, 0), patch_size=80.0)
        self.assertAlmostEqual(float(result['support'][0, 0]), 0.25, places=6)
        self.assertEqual(int(result['count'][0, 0]), 1)
        self.assertAlmostEqual(float(result['U'][0, 0, 0]), 1.0, places=6)

    def test_microcell_assignment_is_bounded(self):
        centers = torch.tensor([[0.0, 0.0], [79.9, 79.9], [80.0, 0.0], [-0.1, 0.0]], dtype=torch.float64)
        patch, micro = microcell_index(centers, origin_xy=(0, 0), patch_size=80.0)
        self.assertTrue(torch.equal(patch, torch.tensor([[0, 0], [0, 0], [1, 0], [-1, 0]])))
        self.assertTrue(torch.equal(micro, torch.tensor([[0, 0], [3, 3], [0, 0], [3, 0]])))

    def test_no_visible_cell_and_nonfinite_visible_rejected(self):
        cells = torch.tensor([box(0, 0, 10, 10)], dtype=torch.float64)
        with self.assertRaises(ValueError):
            aggregate_microcells(torch.ones(1, 32), cells, torch.zeros(1, dtype=torch.bool),
                                 origin_xy=(0, 0), patch_size=80.0)
        bad = torch.full((1, 32), float('nan'))
        with self.assertRaises(ValueError):
            aggregate_microcells(bad, cells, torch.ones(1, dtype=torch.bool),
                                 origin_xy=(0, 0), patch_size=80.0)

    def test_invalid_cell_rectangles_rejected(self):
        features = torch.ones(1, 32)
        keep = torch.ones(1, dtype=torch.bool)
        for cells in (torch.tensor([box(1, 0, 1, 2)], dtype=torch.float64),
                      torch.tensor([box(0, 2, 2, 1)], dtype=torch.float64),
                      torch.tensor([box(0, 0, float('nan'), 2)], dtype=torch.float64),
                      torch.tensor([box(0, 0, float('inf'), 2)], dtype=torch.float64)):
            with self.subTest(cells=cells.tolist()), self.assertRaises(ValueError):
                aggregate_microcells(features, cells, keep, origin_xy=(0, 0), patch_size=80.0)

    def test_unbounded_geometry_expansion_rejected(self):
        cells = torch.tensor([box(0, 0, 1000, 1000)], dtype=torch.float64)
        with self.assertRaises(ValueError):
            aggregate_microcells(torch.ones(1, 32), cells, torch.ones(1, dtype=torch.bool),
                                 origin_xy=(0, 0), patch_size=80.0)


if __name__ == '__main__':
    unittest.main()

"""Exact Sub-Patch/micro-cell geometry: area weights, support, counts, no hidden values."""
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
        self.assertTrue(torch.allclose(result['support'][0, 0], torch.tensor(1.0), atol=1e-6))
        empty = result['support'][0] == 0
        self.assertTrue(torch.equal(result['U'][0][empty], torch.zeros_like(result['U'][0][empty])))

    def test_one_twenty_metre_cell_weights_by_true_overlap(self):
        cells = torch.tensor([box(10, 10, 30, 30)], dtype=torch.float64)
        features = torch.ones(1, 32)
        result = aggregate_microcells(features, cells, torch.ones(1, dtype=torch.bool),
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
        features = torch.ones(4, 32)
        result = aggregate_microcells(features, cells, torch.ones(4, dtype=torch.bool),
                                      origin_xy=(0, 0), patch_size=240.0)
        self.assertAlmostEqual(float(result['support'][0, 0]), 1.0, places=6)
        self.assertEqual(int(result['count'][0, 0]), 4)

    def test_cells_split_across_patches_and_negative_indices(self):
        cells = torch.tensor([box(95, 95, 105, 105), box(85, 85, 95, 95)], dtype=torch.float64)
        result = aggregate_microcells(torch.ones(2, 32), cells, torch.ones(2, dtype=torch.bool),
                                      origin_xy=(100, 100), patch_size=80.0)
        keys = {tuple(item.tolist()) for item in result['patch']}
        self.assertEqual(keys, {(-1, -1), (0, 0)})
        self.assertEqual(result['U'].shape, (2, MICRO * MICRO, 32))

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


if __name__ == '__main__':
    unittest.main()

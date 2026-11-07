import unittest
from dataclasses import replace
import numpy as np
import torch
from .model import BandObservation
from .pair_spatial_holdout import spatial_holdout


class SpatialHoldoutChecks(unittest.TestCase):
    def test_known_shift_selected_without_heldout_fit(self):
        values=torch.from_numpy(np.random.default_rng(4).random((16,18))).float()
        source=BandObservation(values,torch.ones_like(values,dtype=torch.bool),torch.tensor([400.,500.]),torch.ones(2),(1.,0.,0.,0.,-1.,16.),'EPSG:32650')
        target=replace(source,reflectance=values.roll(1,1));result=spatial_holdout(source,target,[-1.,0.,1.])
        self.assertEqual(result['fixed_target_pixels_before_buffer'],14*16);self.assertLess(result['fixed_target_pixels'],14*16);self.assertEqual(result['internal_boundary_margin_m'],[2.,2.])
        for row in result['rows']:
            self.assertEqual(row['selected_source_shift_m'],(1.,0.));self.assertAlmostEqual(row['heldout_selected']['correlation'],1.,places=12)
            self.assertGreater(row['heldout_correlation_gain'],0.);self.assertEqual(row['fit_pixels']+row['heldout_pixels'],result['fixed_target_pixels'])

    def test_no_texture_and_missing_zero(self):
        values=torch.ones(8,8)
        source=BandObservation(values,torch.ones_like(values,dtype=torch.bool),torch.tensor([400.,500.]),torch.ones(2),(1.,0.,0.,0.,-1.,8.),'EPSG:32650')
        result=spatial_holdout(source,source,[-1.,0.,1.])
        self.assertTrue(all(row['selected_source_shift_m'] is None for row in result['rows']))
        with self.assertRaisesRegex(ValueError,'including zero'):spatial_holdout(source,source,[-1.,1.])

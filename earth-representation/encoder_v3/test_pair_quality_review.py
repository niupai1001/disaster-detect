import unittest
from dataclasses import replace
import numpy as np
import torch
import shapely
from .model import BandObservation
from .pair_quality_review import fixed_candidate_comparison,categorical_area_profile,band_summary


class PairQualityChecks(unittest.TestCase):
    def test_known_shift_identical_support_and_zero_residual(self):
        values=torch.from_numpy(np.random.default_rng(4).random((16,18))).float();source=BandObservation(values,torch.ones_like(values,dtype=torch.bool),torch.tensor([400.,500.]),torch.ones(2),(1.,0.,0.,0.,-1.,16.),'EPSG:32650');target=replace(source,reflectance=values.roll(1,1));row,p0,pc,truth,full=fixed_candidate_comparison(source,target,[-1.,0.,1.],(1.,0.))
        self.assertEqual(row['fixed_pixels'],14*16);self.assertAlmostEqual(row['candidate']['correlation'],1.,places=12);self.assertEqual(row['candidate']['rmse'],0.);self.assertGreater(row['zero']['rmse'],0.);self.assertEqual(row['zero']['pixels'],row['candidate']['pixels'])
        with self.assertRaisesRegex(ValueError,'existing diagnostic grid'):fixed_candidate_comparison(source,target,[-1.,0.,1.],(2.,0.))

    def test_categorical_area_preserves_native_partial_cells(self):
        values=np.array([[2,4],[7,4]],dtype=np.uint8);region=shapely.box(0,0,3,4);profile=categorical_area_profile(values,(2.,0.,0.,0.,-2.,4.),np.ones_like(values,dtype=bool),region);areas={x['value']:x['valid_overlap_m2'] for x in profile['values']}
        self.assertEqual(areas,{2:4.,4:4.,7:4.});self.assertEqual(profile['reference_valid_area_m2'],12.);self.assertEqual(profile['QA_covered_area_m2'],12.)

    def test_empty_reflectance_is_absent_not_zero_statistics(self):
        source=BandObservation(torch.zeros(2,3),torch.zeros(2,3,dtype=torch.bool),torch.tensor([400.,500.]),torch.ones(2),(1.,0.,0.,0.,-1.,2.),'EPSG:32650');row=band_summary(source);self.assertEqual(row['valid_pixels'],0);self.assertIsNone(row['minimum']);self.assertIsNone(row['quantiles'])

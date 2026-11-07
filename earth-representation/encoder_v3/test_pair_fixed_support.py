import unittest
import numpy as np
import torch
from .test_adaptive_window import source
from .pair_fixed_support import fixed_support_scores


class FixedSupportChecks(unittest.TestCase):
    def test_same_pixels_and_mass_at_all_offsets(self):
        b=source(1.,1.);b.valid_mask[5,5]=False;b.reflectance[5,5]=float('nan')
        result=fixed_support_scores(b,b,[-1.,0.,1.]);rows=result['offsets']
        self.assertGreater(result['fixed_target_pixels'],0);self.assertLess(result['fixed_target_pixels'],int(b.keep().sum()))
        self.assertEqual(len({row['pixels'] for row in rows}),1);self.assertEqual(len({row['support_m2'] for row in rows}),1)
        center=next(row for row in rows if row['dx_m']==0 and row['dy_m']==0);self.assertAlmostEqual(center['correlation'],1.)

    def test_no_full_support_is_reported_without_fallback(self):
        b=source(1.,1.);result=fixed_support_scores(b,b,[-1000.,1000.])
        self.assertEqual(result['fixed_target_pixels'],0);self.assertIsNone(result['best_measured_correlation']);self.assertTrue(all(row['correlation'] is None for row in result['offsets']))
        with self.assertRaises(ValueError):fixed_support_scores(b,b,[])

import unittest
from dataclasses import replace
import numpy as np
import torch
from tests.encoder_v3.test_adaptive_window import source
from encoder_v3.pair_registration import area_projection,registration_score


class RegistrationChecks(unittest.TestCase):
    def test_exact_native_overlap_and_hidden_values(self):
        b=source(1.,1.);p,area=area_projection(b,b,dx=0.,dy=0.);np.testing.assert_allclose(p,b.reflectance.numpy());np.testing.assert_allclose(area,1.)
        b.visible=b.valid_mask.clone();b.visible[2,2]=False;b.reflectance[2,2]=float('nan')
        p,area=area_projection(b,b,dx=0.,dy=0.);self.assertEqual(area[2,2],0);self.assertTrue(np.isnan(p[2,2]));self.assertAlmostEqual(registration_score(p,area,b)['correlation'],1.)
        target=replace(b,affine=(1.,0.,101.13,0.,-1.,201.27))
        p,area=area_projection(b,target,dx=1.,dy=0.);np.testing.assert_allclose(p,b.reflectance.numpy(),equal_nan=True)

    def test_fractional_support_and_constant_scene(self):
        b=source(1.,1.);b.reflectance=torch.ones_like(b.reflectance)
        p,area=area_projection(b,b,dx=.25,dy=.5);np.testing.assert_allclose(p[area>0],1.);self.assertTrue(bool(((area>0)&(area<1)).any()));self.assertIsNone(registration_score(p,area,b)['correlation'])

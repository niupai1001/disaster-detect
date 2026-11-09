import unittest
import numpy as np
from encoder_v3.pair_support_audit import mask_region


class PairSupportChecks(unittest.TestCase):
    def test_native_holes_noninteger_cells_and_partial_overlap(self):
        mask=np.array([[1,0,1],[1,1,0]],dtype=bool)
        region=mask_region(mask,(.37,0,10,0,-1.21,20))
        self.assertAlmostEqual(region.area,4*.37*1.21)
        other=mask_region([[1]],(.37,0,10+.37/2,0,-1.21,20))
        self.assertAlmostEqual(region.intersection(other).area,.37*1.21/2)
        self.assertTrue(mask_region(np.zeros((2,3)),(.37,0,10,0,-1.21,20)).is_empty)
        with self.assertRaises(ValueError):mask_region(mask,(1,.1,0,0,-1,0))

import unittest
import numpy as np
from encoder_v3.audit_train_statistics import merge_moments


class MomentChecks(unittest.TestCase):
    def test_streaming_merge_matches_full_values_and_empty(self):
        state=dict(count=0,mean=0.,m2=0.,min=float('inf'),max=float('-inf'));values=np.array([-.2,0.,.1,.9,1.2],dtype=np.float64)
        for part in (values[:2],np.array([]),values[2:]):merge_moments(state,part)
        self.assertEqual(state['count'],5);self.assertAlmostEqual(state['mean'],values.mean(),places=14);self.assertAlmostEqual(state['m2']/5,values.var(),places=14)
        self.assertEqual(state['min'],-.2);self.assertEqual(state['max'],1.2)
        with self.assertRaises(ValueError):merge_moments(state,np.array([float('nan')]))

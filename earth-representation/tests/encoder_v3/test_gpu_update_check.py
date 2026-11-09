import unittest
import torch
from encoder_v3.gpu_update_check import cpu_tree,difference


class UpdateMeasurementChecks(unittest.TestCase):
    def test_snapshot_and_difference_do_not_mutate_parameters(self):
        original=dict(parameter=torch.tensor([1.,2.],requires_grad=True),state={'step':1},groups=[(.04,3e-6)])
        snapshot=cpu_tree(original)
        self.assertFalse(snapshot['parameter'].requires_grad)
        self.assertTrue(difference(original,snapshot)['bitwise_equal'])
        snapshot['parameter'][0]=1.01
        measured=difference(original,snapshot)
        self.assertGreater(measured['max_absolute'],.009)
        self.assertFalse(measured['bitwise_equal']);self.assertEqual(original['parameter'][0],1.)
        snapshot['state']['step']=2
        self.assertIn('root/state/step',difference(original,snapshot)['scalar_mismatches'])
        with self.assertRaisesRegex(ValueError,'layout'):difference(torch.zeros(2),torch.zeros(3))
        self.assertFalse(difference(torch.ones(1),torch.ones(1,dtype=torch.float64))['bitwise_equal'])

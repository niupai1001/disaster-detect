import unittest
from dataclasses import replace
import numpy as np
import torch
from encoder_v3.participation_layout import participation_layout,AreaEquivalentHillRate
from tests.encoder_v3.test_adaptive_window import source


class ParticipationLayoutChecks(unittest.TestCase):
    def curve(self):return AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)

    def test_unique_membership_large_groups_and_rate_counts(self):
        for g in (.5,1.21,10.,20.):
            b=replace(source(g,g),reflectance=torch.zeros(64,64),valid_mask=torch.ones(64,64,dtype=torch.bool))
            layout=participation_layout(b,rate_curve=self.curve(),selection_seed=0,origin_xy=(100.13,201.27-64*g));stats=layout['participation_audit'];indices=layout['indices']
            self.assertEqual(len(indices),4096);self.assertEqual(len(set(indices)),4096);self.assertEqual(sum(n for n in layout['lengths'] if n>1),stats['merged_participants']);self.assertEqual(stats['tokens'],len(layout['lengths']))
            self.assertAlmostEqual(stats['candidate_span_pixels'],1/(1-stats['target_rate']));self.assertLess(abs(stats['actual_rate']-stats['target_rate']),.01)
            if g==.5:self.assertGreater(stats['group_size_max'],4)
            second=participation_layout(b,rate_curve=self.curve(),selection_seed=0,origin_xy=(100.13,201.27-64*g));np.testing.assert_array_equal(layout['indices'],second['indices'])

    def test_masked_noninteger_anisotropic_and_empty(self):
        b=source(.5,1.21);b.valid_mask[0,0]=False;b.visible=b.valid_mask.clone();b.visible[3:7,3:7]=False
        layout=participation_layout(b,rate_curve=self.curve(),selection_seed=3,origin_xy=(99.8,180.));self.assertEqual(sum(layout['lengths']),int(b.keep().sum()))
        keep=b.keep().flatten().numpy();self.assertTrue(keep[layout['indices']].all());self.assertEqual(sum(len(x) for x in layout['visible']),int(b.keep().sum()))
        allowed=b.cells()[b.keep().flatten()].numpy()
        for c in layout['receptive']:self.assertTrue(np.any(np.all(allowed==c,axis=1)))
        empty=replace(b,valid_mask=torch.zeros_like(b.valid_mask));self.assertIsNone(participation_layout(empty,rate_curve=self.curve(),selection_seed=3,origin_xy=(99.8,180.)))

import unittest
import torch
from .adaptive_census import band_census,spectral_panel
from .adaptive_window import adaptive_layout
from .test_adaptive_window import source
from dataclasses import replace


class CensusChecks(unittest.TestCase):
    def test_counts_match_executed_layout(self):
        for gx,gy in ((.5,1.21),(10.,30.)):
            b=source(gx,gy);b.valid_mask[2:4,2:4]=False;b.visible=b.valid_mask.clone();b.visible[5:8,5:8]=False
            origin=(97.3,151.2);layout=adaptive_layout(b,exponent=.5,origin_xy=origin);row=band_census(b,exponent=.5,origin_xy=origin)
            self.assertEqual(row['tokens'],len(layout['lengths']));self.assertEqual(row['visible_pixels'],sum(layout['lengths']));self.assertEqual(row['local_sequence_max'],max(layout['lengths']))

    def test_spectral_coverage_and_hidden_counts(self):
        for count in (2,4,7,10):
            bands=[replace(source(),record_id=str(i),wavelengths_nm=torch.tensor([450.,500.,550.])+i*200) for i in range(count)]
            panel=spectral_panel(bands);groups=panel['hidden_band_combinations'];names={str(i) for i in range(count)}
            self.assertEqual(set(map(len,groups)),set(range(1,count)))
            self.assertEqual({g[0] for g in groups if len(g)==1},names)
            self.assertTrue(all(0<len(g)<count and set(g)<names for g in groups));self.assertEqual(len(groups),len({tuple(g) for g in groups}))
            for size in range(1,count):self.assertEqual(set().union(*(set(g) for g in groups if len(g)==size)),names)

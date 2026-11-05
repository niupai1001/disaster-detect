import unittest
import numpy as np
import torch
from .token_census import token_count
from .model import BandObservation
from .layout import native_layout


class TokenCensusChecks(unittest.TestCase):
    def test_noninteger_grid_holes_and_offsets_match_full_layout(self):
        mask=np.random.default_rng(3).random((13,17))>.4
        band=BandObservation(torch.zeros(13,17),torch.from_numpy(mask),torch.tensor([400.,500.]),torch.ones(2),(.37,0,10.21,0,-1.21,42.7),'EPSG:32650')
        for patch in (1.7,3.3):
            for origin in ((9.,20.),(10.21,26.97)):
                layout=native_layout(band,patch,origin)
                self.assertEqual(token_count(mask,band.affine,patch,origin),len(layout['lengths']))
        self.assertEqual(token_count(np.zeros_like(mask),band.affine,1.7,(0,0)),0)

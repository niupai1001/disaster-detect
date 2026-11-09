import unittest
from dataclasses import replace
import numpy as np
import shapely
import torch
from encoder_v3.data import Observation
from encoder_v3.model import ReflectanceDecoder
from encoder_v3.participation_encoder import ParticipationEncoder, prepare_participation_task
from encoder_v3.participation_layout import AreaEquivalentHillRate
from encoder_v3.pair_geometry import PairQueryFrame
from encoder_v3.jepa import _regions
from encoder_v3.tasks import spatial_task
from tests.encoder_v3.test_adaptive_window import source


class FormalSemanticsChecks(unittest.TestCase):
    def test_pair_query_offset_is_applied_once(self):
        frame = PairQueryFrame("s", "t", "EPSG:32631", (10.0, -5.0), "a" * 64)
        boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        shifted = frame.boxes(boxes)
        self.assertEqual(float(shifted[0, 0]), 10.0)
        self.assertEqual(float(shifted[0, 2]), 14.0)
        self.assertEqual(float(shifted[0, 1]), -5.0)
        region = _regions([boxes.double()], torch.tensor([True]), offset_xy=(10.0, -5.0))[0]
        self.assertEqual(tuple(round(v, 9) for v in region.bounds), (10.0, -5.0, 14.0, -1.0))

    def test_dense_A_uses_represented_boundary_rule(self):
        band = replace(source(.5, .5), reflectance=torch.arange(16).reshape(4, 4).float() / 100,
                       valid_mask=torch.ones(4, 4, dtype=torch.bool), affine=(.5, 0., 0., 0., -.5, 2.))
        task = spatial_task(Observation([band], dict(id="generated", role="TRAIN")), (.5, .5, 2.5, 2.5))
        curve = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)
        plan = prepare_participation_task(task, rate_curve=curve, selection_seed=0, origin_xy=(0, 0), chunk_pixels=3, cell_indexed=True)
        model = ParticipationEncoder(rate_curve=curve, selection_seed=0, hyper_hidden=64)
        representation = model(task.visible, origin_xy=(0, 0))
        targets = task.targets[0].band.cells()[plan.bands[0].indices]
        condition = model.srf(task.targets[0].band.wavelengths_nm, task.targets[0].band.response)
        mask = torch.tensor(task.rectangle, dtype=torch.float64)
        indexed = ReflectanceDecoder(indexed=True, cell_indexed=True)
        dense = ReflectanceDecoder(indexed=False)
        dense.load_state_dict(indexed.state_dict())
        a, support_a = indexed(representation, targets, condition, kind="A", mask_bounds=mask)
        b, support_b = dense(representation, targets, condition, kind="A", mask_bounds=mask)
        torch.testing.assert_close(support_a, support_b)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()

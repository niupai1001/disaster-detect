import json,tempfile,unittest
from pathlib import Path
import torch
from encoder_v3.model import ReflectanceDecoder
from encoder_v3.participation_encoder import ParticipationEncoder
from encoder_v3.participation_layout import AreaEquivalentHillRate
from encoder_v3.participation_validation_probe import confirmed_curve,evaluate_case,parameter_hash
from encoder_v3.data import Observation
from tests.encoder_v3.test_adaptive_window import source
from dataclasses import replace


class ParticipationValidationChecks(unittest.TestCase):
    def test_unreviewed_curve_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'curve.json';p.write_text(json.dumps(dict(status='CANDIDATE',curve=dict(effective_gsd='sqrt(gx*gy)'))))
            with self.assertRaisesRegex(ValueError,'Confirmed preflight'):confirmed_curve(p)

    def test_real_confirmation_coefficients_and_anchors(self):
        curve=confirmed_curve((Path(__file__).resolve().parents[2] / 'src/encoder_v3' / 'reports')/'merge_rate_confirmation.json')
        for g,r in ((.5,.75),(10.,.2),(20.,.1)):self.assertAlmostEqual(curve(g,g),r,places=12)
        self.assertEqual(curve(.5,2.),curve(1.,1.))

    def test_forward_panel_preserves_params_and_covers_targets(self):
        torch.set_num_threads(2);torch.manual_seed(4);curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
        encoder=ParticipationEncoder(rate_curve=curve,selection_seed=0,hyper_hidden=64,local_batch_size=64,packed_local_output=True).eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True).eval()
        bands=[replace(source(),record_id=str(i),wavelengths_nm=source().wavelengths_nm+i*100) for i in range(3)]
        obs=Observation(bands,dict(id='diagnostic',sensor='synthetic'));before=parameter_hash((encoder,decoder))
        for hidden in (['0'],['0','2']):
            row=evaluate_case(obs,hidden,encoder=encoder,decoder=decoder,train_stds={str(i):1. for i in range(3)},origin_xy=(97.,151.),chunk_pixels=128)
            self.assertEqual(row['status'],'OK');self.assertEqual({b['band'] for b in row['bands']},set(hidden))
            self.assertTrue(all(b['count']>0 for b in row['bands']));self.assertEqual({b['band'] for b in row['participation_audit']},{str(i) for i in range(3)}-set(hidden))
        self.assertEqual(parameter_hash((encoder,decoder)),before)

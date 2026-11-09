"""Control/commit tests with explicit supplied gradients; no model backward proof."""
from unittest.mock import patch
import torch
from tests.encoder_v3 import test_route_step as base


class RouteTransactionChecks(base.RouteStepChecks):
    def setUp(self):
        super().setUp()
        def gradients(encoder,decoder,views,*,predictor=None,**kwargs):
            # Intentionally substitute only gradient production; test real AdamW,
            # EMA, scheduler, I/O, lineage and restoration separately from graphs.
            for module in [encoder,decoder]+([predictor] if predictor is not None else []):
                for p in module.parameters():p.grad=torch.full_like(p,.001)
            return dict(effective_batch=8,optimizer_updates=0,logical_update=kwargs['logical_update'],weighted_loss=.125,gradient_fixture=True)
        patchers=[patch('encoder_v3.route_step.accumulate_adaptive_joint',side_effect=gradients),
                  patch('torch.cuda.is_available',return_value=False),
                  patch('torch.cuda._lazy_init',side_effect=AssertionError('No CUDA initialization in CPU transaction tests')),
                  patch('torch.autograd.graph._engine_run_backward',side_effect=AssertionError('No autograd engine in transaction tests'))]
        for patcher in patchers:patcher.start();self.addCleanup(patcher.stop)

    def test_JEPA_cannot_change_common_AB_exposure(self):
        import tempfile
        from pathlib import Path
        f=self.fixture(jepa=True,cursor=2000);f['expected_AB_input_signature']='c'*64
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')))
            with self.assertRaisesRegex(ValueError,'common-prefix stream'):self.call(f,folder)
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_teacher_target_must_belong_to_same_frozen_TRAIN_release(self):
        import tempfile
        from pathlib import Path
        f=self.fixture(jepa=True,cursor=2000);f['jepa_views'][0].target.metadata['frozen_source_ids']={'members':'unfrozen_fixture'}
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')))
            with self.assertRaisesRegex(ValueError,'Teacher target source differs'):self.call(f,folder)
            self.assertEqual(list(Path(folder).iterdir()),[])

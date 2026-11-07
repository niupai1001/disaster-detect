"""Control/commit tests with explicit supplied gradients; no model backward proof."""
from unittest.mock import patch
import torch
from . import test_route_step as base


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

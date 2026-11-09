"""Check pending interface arithmetic and the documented TRAIN-mean baseline."""
from pathlib import Path
import re
import unittest
import torch

ROOT = Path(__file__).resolve().parents[4]
DOCS = ROOT / 'src/encoder_v3/latent_response'


class InterfaceContractTest(unittest.TestCase):
    def test_candidate_geometry_and_decoder_widths(self):
        geometry = 2 * 4 * 2 + 2 + 1
        width = 3 * 128 + geometry + 32 + 1 + 1
        self.assertEqual((geometry, width), (19, 437))
        for name in ('interface_revision.md', 'interface_proposal.md'):
            text = (DOCS / name).read_text(encoding='utf-8')
            self.assertIn('Linear(437,128)', text)
            self.assertNotIn('429', text)
            self.assertNotIn('q_geo(11)', text)
            self.assertIn('仅', text)
            self.assertIn('候选', text)

    def test_documented_mean_baseline_equals_direct_scored_mse(self):
        text = (DOCS / 'interface_proposal.md').read_text(encoding='utf-8')
        expression = re.search(r'`NMSE=(Var_target[^`]+)`', text).group(1)
        expression = expression.replace('σ²', 'sigma**2').replace('σ', 'sigma')
        expression = expression.replace('μ', 'mu').replace('−', '-').replace('²', '**2')
        for values, mean, sigma in (([0., 4.], 1., 1.), ([3., 3.], 3., 2.),
                                    ([2.], -1., .5), ([-2., 1., 7.], 0., 3.)):
            with self.subTest(values=values):
                # These are the scored pixels, after masking out other pixels.
                target = torch.tensor(values, dtype=torch.float64)
                terms = dict(Var_target=target.var(correction=0).item(),
                             mean_target=target.mean().item(), mu=mean, sigma=sigma)
                formula = eval(expression, {'__builtins__': {}}, terms)
                direct = ((target - mean).square().mean() / sigma**2).item()
                self.assertAlmostEqual(formula, direct, places=12)
                self.assertAlmostEqual((((target-mean)/sigma)**2).mean().item(), direct, places=12)

"""Shared dynamic kernel bank and hidden-value-isolated native convolution."""
import unittest
import torch
from encoder_v3.latent_response import DynamicKernelBank, masked_native_conv


def bank(seed=0):
    torch.manual_seed(seed)
    return DynamicKernelBank()


class KernelBankTest(unittest.TestCase):
    def test_shapes_and_normalized_coefficients(self):
        model = bank()
        z = torch.randn(4, 32)
        alpha = model.coefficients(z)
        kernels = model.kernels(z)
        self.assertEqual(alpha.shape, (4, 8))
        self.assertTrue(torch.allclose(alpha.sum(-1), torch.ones(4)))
        self.assertEqual(kernels.shape, (4, 32, 1, 3, 3))
        self.assertEqual(tuple(model.base.shape), (32, 1, 3, 3))
        self.assertEqual(tuple(model.basis.shape), (8, 32, 1, 3, 3))
        self.assertEqual(tuple(model.bias.shape), (32,))

    def test_two_channel_conditions_give_different_kernels(self):
        model = bank()
        first = model.kernels(torch.zeros(1, 32))
        second = model.kernels(torch.ones(1, 32))
        self.assertFalse(torch.equal(first, second))

    def test_gradients_reach_bank_and_condition(self):
        model = bank()
        z = torch.randn(2, 32, requires_grad=True)
        model.kernels(z).square().sum().backward()
        self.assertIsNotNone(model.base.grad)
        self.assertIsNotNone(model.basis.grad)
        self.assertIsNotNone(model.coefficients_layer.weight.grad)
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all())


class MaskedConvTest(unittest.TestCase):
    def setUp(self):
        self.model = bank(seed=5)
        self.values = torch.randn(1, 5, 6)
        self.keep = torch.ones(1, 5, 6, dtype=torch.bool)

    def run_conv(self, values, keep, kernels, *, tile_rows=None):
        return masked_native_conv(values, keep, kernels, self.model.bias, tile_rows=tile_rows)

    def test_tiled_execution_is_equivalent(self):
        kernels = self.model.kernels(torch.zeros(1, 32))
        full = self.run_conv(self.values, self.keep, kernels)
        for rows in (1, 2, 3):
            self.assertTrue(torch.allclose(full, self.run_conv(self.values, self.keep, kernels, tile_rows=rows),
                                           atol=1e-6, rtol=0))
        self.assertEqual(full.shape, (1, 32, 5, 6))

    def test_per_sample_kernels_match_individual_calls(self):
        z = torch.stack((torch.zeros(32), torch.ones(32)))
        values = torch.stack((self.values[0], self.values[0]))
        keep = torch.stack((self.keep[0], self.keep[0]))
        kernels = self.model.kernels(z)
        grouped = self.run_conv(values, keep, kernels)
        for index in range(2):
            single = self.run_conv(values[index:index + 1], keep[index:index + 1], kernels[index:index + 1])
            self.assertTrue(torch.allclose(grouped[index:index + 1], single, atol=1e-6, rtol=0))

    def test_hidden_values_cannot_change_visible_output(self):
        kernels = self.model.kernels(torch.zeros(1, 32))
        keep = self.keep.clone()
        keep[0, 2, 2] = False
        clean = self.values.clone()
        poisoned = clean.clone()
        poisoned[0, 2, 2] = float('nan')
        reference = self.run_conv(clean, keep, kernels)
        self.assertTrue(torch.equal(reference, self.run_conv(poisoned, keep, kernels)))
        poisoned[0, 2, 2] = 1e9
        self.assertTrue(torch.equal(reference, self.run_conv(poisoned, keep, kernels)))

    def test_gradient_at_hidden_position_is_zero(self):
        kernels = self.model.kernels(torch.zeros(1, 32))
        keep = self.keep.clone()
        keep[0, 2, 2] = False
        poisoned = self.values.clone()
        poisoned[0, 2, 2] = float('nan')
        poisoned.requires_grad_(True)
        output = self.run_conv(poisoned, keep, kernels)
        output.sum().backward()
        self.assertEqual(float(poisoned.grad[0, 2, 2]), 0.0)
        self.assertTrue(torch.isfinite(poisoned.grad[keep]).all())

    def test_visible_neighbour_count_normalization(self):
        model = DynamicKernelBank(bases=8).eval()
        values = torch.zeros(1, 5, 5)
        values[0, 2, 2] = 2.0
        keep = torch.zeros(1, 5, 5, dtype=torch.bool)
        keep[0, 2, 2] = True
        kernels = model.kernels(torch.zeros(1, 32))
        output = masked_native_conv(values, keep, kernels, model.bias)
        expected = torch.nn.functional.gelu(9.0 * kernels[0, :, 0, 1, 1] * 2.0 + model.bias)
        self.assertTrue(torch.allclose(output[0, :, 2, 2], expected, atol=1e-6, rtol=0))
        self.assertTrue(torch.equal(output[0, :, 0, 0], torch.nn.functional.gelu(model.bias)))

    def test_invalid_execution_arguments(self):
        kernels = self.model.kernels(torch.zeros(1, 32))
        with self.assertRaises(ValueError):
            masked_native_conv(self.values, self.values > 0, kernels, self.model.bias, tile_rows=0)
        with self.assertRaises(ValueError):
            masked_native_conv(self.values, self.keep[0], kernels, self.model.bias)
        with self.assertRaises(ValueError):
            masked_native_conv(torch.stack((self.values[0], self.values[0])),
                               torch.stack((self.keep[0], self.keep[0])), kernels, self.model.bias)


if __name__ == '__main__':
    unittest.main()

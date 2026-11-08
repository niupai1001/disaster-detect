"""Shared module queries: static channel metadata only, identical across input subsets."""
import unittest
import torch
from encoder_v3.latent_response import ChannelMeta, ModuleQuery
from encoder_v3.latent_response.module_query import KINDS, fourier_features

# Frequency set is a test fixture only: I2 has not frozen the Fourier band set.
FREQUENCIES = (1.0, 2.0, 4.0)


class ModuleQueryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.query = ModuleQuery(frequencies=FREQUENCIES)
        self.b04 = ChannelMeta('B04', 2, 10)
        self.b08 = ChannelMeta('B08', 6, 10)

    def test_query_shape_and_encode_width(self):
        value = self.query('conv', self.b04, (10.0, 10.0))
        self.assertEqual(value.shape, (32,))
        self.assertEqual(self.query.encode(self.b04, (10.0, 10.0)).shape, (2 * len(FREQUENCIES) + 2,))

    def test_same_channel_identity_across_input_subsets(self):
        four_band = self.query('conv', self.b04, (10.0, 10.0))
        ten_band = self.query('conv', self.b04, (10.0, 10.0))
        self.assertTrue(torch.equal(four_band, ten_band))
        # Query depends on static metadata, not on a sensor-specific table.
        other_sensor = self.query('conv', ChannelMeta('B04', 2, 10), (10.0, 10.0))
        self.assertTrue(torch.equal(four_band, other_sensor))

    def test_two_target_bands_are_distinguishable(self):
        self.assertFalse(torch.equal(self.query('decoder', self.b04, (10.0, 10.0)),
                                     self.query('decoder', self.b08, (10.0, 10.0))))

    def test_three_module_kinds_are_distinguishable(self):
        values = [self.query(kind, self.b04, (10.0, 10.0)) for kind in KINDS]
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                self.assertFalse(torch.equal(values[left], values[right]))

    def test_native_gsd_enters_the_query(self):
        coarse = self.query('channel', self.b04, (20.0, 20.0))
        fine = self.query('channel', self.b04, (10.0, 10.0))
        self.assertFalse(torch.equal(coarse, fine))

    def test_gradient_reaches_seed_and_projection(self):
        total = sum(self.query(kind, self.b04, (10.0, 10.0)).sum() for kind in KINDS)
        total.backward()
        for kind in KINDS:
            seed = self.query.seed[kind]
            self.assertIsNotNone(seed.grad)
            self.assertTrue(torch.isfinite(seed.grad).all())
            weight = self.query.project[kind][0].weight
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertTrue(bool(weight.grad.abs().sum() > 0))

    def test_untouched_module_query_stays_empty(self):
        self.query('conv', self.b04, (10.0, 10.0)).square().sum().backward()
        self.assertIsNotNone(self.query.seed['conv'].grad)
        self.assertIsNone(self.query.seed['decoder'].grad)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            ModuleQuery(frequencies=())
        with self.assertRaises(ValueError):
            ModuleQuery(frequencies=(0.0,))
        with self.assertRaises(ValueError):
            self.query('other', self.b04, (10.0, 10.0))
        with self.assertRaises(ValueError):
            self.query('conv', self.b04, (0.0, 10.0))
        with self.assertRaises(ValueError):
            self.query('conv', 'B04', (10.0, 10.0))
        with self.assertRaises(ValueError):
            fourier_features(torch.zeros(2), self.query.frequencies)


if __name__ == '__main__':
    unittest.main()

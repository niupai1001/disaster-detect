"""Shared memory read plus the P32 -> KV -> channel-query -> read component chain."""
import unittest
import torch
from encoder_v3.latent_response import (SensorKey, ChannelCatalog, SensorRegistry,
                                        SensorMemory, ModuleQuery)
from encoder_v3.latent_response.memory_read import read_memory

S2 = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']


class MemoryReadTest(unittest.TestCase):
    def test_attention_is_normalized_and_not_constant(self):
        torch.manual_seed(3)
        keys, values = torch.randn(8, 32), torch.randn(8, 32)
        first = torch.randn(32)
        second = torch.randn(32)
        weights_first, read_first = read_memory(first, keys, values)
        weights_second, read_second = read_memory(second, keys, values)
        self.assertAlmostEqual(float(weights_first.sum()), 1.0, places=6)
        self.assertTrue(torch.isfinite(weights_first).all())
        self.assertGreater(float(weights_first.max() - weights_first.min()), 1e-3)
        self.assertFalse(torch.equal(weights_first, weights_second))
        self.assertFalse(torch.equal(read_first, read_second))

    def test_gradients_reach_query_keys_values(self):
        torch.manual_seed(4)
        query = torch.randn(32, requires_grad=True)
        keys = torch.randn(8, 32, requires_grad=True)
        values = torch.randn(8, 32, requires_grad=True)
        _, read = read_memory(query, keys, values)
        read.sum().backward()
        for tensor in (query, keys, values):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0)

    def test_empty_or_invalid_memory_rejected(self):
        with self.assertRaises(ValueError):
            read_memory(torch.zeros(32), torch.zeros(0, 32), torch.zeros(0, 32))
        with self.assertRaises(ValueError):
            read_memory(torch.zeros(32), torch.zeros(8, 32), torch.zeros(7, 32))
        with self.assertRaises(ValueError):
            read_memory(torch.zeros(31), torch.zeros(8, 32), torch.zeros(8, 32))
        with self.assertRaises(ValueError):
            read_memory(torch.full((32,), float('nan')), torch.zeros(8, 32), torch.zeros(8, 32))


class ComponentChainTest(unittest.TestCase):
    def test_two_channels_of_one_descriptor_reach_different_reads(self):
        registry = SensorRegistry({SensorKey('S2', 'S2A'): ChannelCatalog.from_ordered(S2)}, seed=7)
        memory = SensorMemory()
        query = ModuleQuery(frequencies=(1.0, 2.0, 4.0))
        p = registry.get(SensorKey('S2', 'S2A'))
        keys, values = memory(p)
        b04 = query('decoder', registry.meta(SensorKey('S2', 'S2A'), 'B04'), (10.0, 10.0))
        b08 = query('decoder', registry.meta(SensorKey('S2', 'S2A'), 'B08'), (10.0, 10.0))
        weights_b04, read_b04 = read_memory(b04, keys, values)
        weights_b08, read_b08 = read_memory(b08, keys, values)
        self.assertFalse(torch.equal(weights_b04, weights_b08))
        self.assertFalse(torch.equal(read_b04, read_b08))
        read_b04.sum().backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_same_channel_metadata_reads_the_same_condition(self):
        registry = SensorRegistry({SensorKey('S2', 'S2A'): ChannelCatalog.from_ordered(S2)}, seed=7)
        memory, query = SensorMemory(), ModuleQuery(frequencies=(1.0, 2.0, 4.0))
        keys, values = memory(registry.get(SensorKey('S2', 'S2A')))
        meta = registry.meta(SensorKey('S2', 'S2A'), 'B04')
        first = read_memory(query('conv', meta, (10.0, 10.0)), keys, values)
        second = read_memory(query('conv', meta, (10.0, 10.0)), keys, values)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))


if __name__ == '__main__':
    unittest.main()

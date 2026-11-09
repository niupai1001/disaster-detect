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


class BatchedMemoryReadTest(unittest.TestCase):
    def test_batch_shapes_loop_and_gradients(self):
        torch.manual_seed(13)
        for batch in (1, 2):
            for count in (None, 3):
                with self.subTest(batch=batch, queries=count):
                    q = torch.randn((batch, 5) if count is None else (batch, count, 5),
                                    dtype=torch.float64, requires_grad=True)
                    k = torch.randn(batch, 4, 5, dtype=torch.float64, requires_grad=True)
                    v = torch.randn(batch, 4, 7, dtype=torch.float64, requires_grad=True)
                    w, read = read_memory(q, k, v)
                    self.assertEqual(w.shape, q.shape[:-1] + (4,))
                    self.assertEqual(read.shape, q.shape[:-1] + (7,))
                    loop = [read_memory(q[i], k[i], v[i]) for i in range(batch)]
                    expected_w = torch.stack([x[0] for x in loop])
                    expected = torch.stack([x[1] for x in loop])
                    torch.testing.assert_close(w, expected_w)
                    torch.testing.assert_close(read, expected)
                    grads = torch.autograd.grad(read.square().sum(), (q, k, v), retain_graph=True)
                    reference = torch.autograd.grad(expected.square().sum(), (q, k, v))
                    for actual, wanted in zip(grads, reference):
                        torch.testing.assert_close(actual, wanted)

    def test_cross_sample_isolation(self):
        torch.manual_seed(14)
        for count in (None, 3):
            q = torch.randn((2, 5) if count is None else (2, count, 5))
            k, v = torch.randn(2, 4, 5), torch.randn(2, 4, 7)
            w, read = read_memory(q, k, v)
            q[1] += 100; k[1] *= -10; v[1] += 100
            changed_w, changed = read_memory(q, k, v)
            torch.testing.assert_close(w[0], changed_w[0], rtol=0, atol=0)
            torch.testing.assert_close(read[0], changed[0], rtol=0, atol=0)

    def test_many_queries_shared_memory(self):
        q, k, v = torch.randn(2, 3, 5), torch.randn(4, 5), torch.randn(4, 7)
        w, read = read_memory(q, k, v, scale=.3)
        torch.testing.assert_close(w, torch.softmax(q @ k.T * .3, -1))
        torch.testing.assert_close(read, torch.stack([read_memory(x, k, v, scale=.3)[1] for x in q]))

    def test_nested_batch(self):
        q, k, v = torch.randn(2, 3, 5), torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 7)
        _, read = read_memory(q, k, v)
        self.assertEqual(read.shape, (2, 3, 7))
        for i in range(2):
            for j in range(3):
                torch.testing.assert_close(read[i, j], read_memory(q[i, j], k[i, j], v[i, j])[1])

    def test_broadcast_shapes_rejected(self):
        for shape in ((5,), (1, 5), (3, 5), (1, 3, 5), (2, 1, 3, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                read_memory(torch.randn(shape), torch.randn(2, 4, 5), torch.randn(2, 4, 7))
        with self.assertRaises(ValueError):
            read_memory(torch.randn(2, 5), torch.randn(2, 4, 5), torch.randn(1, 4, 7))
        with self.assertRaises(ValueError):
            read_memory(torch.randn(5), torch.randn(4, 5).double(), torch.randn(4, 7))
        with self.assertRaises(ValueError):
            read_memory(torch.empty(0), torch.empty(4, 0), torch.randn(4, 7))


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

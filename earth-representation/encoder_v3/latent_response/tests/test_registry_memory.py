"""Registry (one P32 per SensorKey) and shared multi-slot memory."""
import unittest
import torch
from encoder_v3.latent_response import (SensorKey, ChannelCatalog, SensorRegistry, SensorMemory,
                                        ChannelMeta)

S2A, S2B, S2C = SensorKey('S2', 'S2A'), SensorKey('S2', 'S2B'), SensorKey('S2', 'S2C')
L8, L9 = SensorKey('L8', 'LC08'), SensorKey('L9', 'LC09')
# Fixtures from the observed release metadata only: they are NOT an admitted catalog (I5 pending).
S2_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
CATALOGS = {S2A: ChannelCatalog.from_ordered(S2_BANDS), S2B: ChannelCatalog.from_ordered(S2_BANDS),
            S2C: ChannelCatalog.from_ordered(S2_BANDS),
            L8: ChannelCatalog.from_ordered([f'SR_B{i}' for i in range(1, 8)]),
            L9: ChannelCatalog.from_ordered([f'SR_B{i}' for i in range(2, 8)])}


class RegistryTest(unittest.TestCase):
    def test_one_descriptor_per_sensor_key(self):
        registry = SensorRegistry(CATALOGS, seed=7)
        self.assertEqual(len(registry.p), len(CATALOGS))
        self.assertEqual(len(list(registry.parameters())), len(CATALOGS))
        self.assertEqual(sum(p.numel() for p in registry.parameters()), 32 * len(CATALOGS))
        self.assertIs(registry.get(S2A), registry.get(S2A))
        self.assertIsNot(registry.get(S2A), registry.get(S2B))

    def test_224_channels_still_one_descriptor(self):
        many = SensorKey('SYN', 'HYPER224')
        registry = SensorRegistry({many: ChannelCatalog.from_ordered([f'C{i:03d}' for i in range(224)])}, seed=3)
        self.assertEqual(len(registry.p), 1)
        self.assertEqual(registry.get(many).shape, (32,))
        self.assertEqual(registry.meta(many, 'C223'), ChannelMeta('C223', 223, 224))

    def test_channel_identity_is_catalog_metadata_not_tensor_index(self):
        registry = SensorRegistry(CATALOGS, seed=7)
        # Four-band and ten-band inputs of one instrument share this metadata object.
        self.assertEqual(registry.meta(S2A, 'B04'), ChannelMeta('B04', 2, 10))
        self.assertEqual(registry.meta(S2B, 'B04'), ChannelMeta('B04', 2, 10))
        self.assertEqual(registry.meta(L8, 'SR_B1'), ChannelMeta('SR_B1', 0, 7))
        with self.assertRaises(KeyError):
            registry.meta(S2A, 'B01')

    def test_incomplete_catalog_changes_identity(self):
        full = SensorRegistry({S2A: ChannelCatalog.from_ordered(S2_BANDS)}, seed=7)
        partial = SensorRegistry({S2A: ChannelCatalog.from_ordered(S2_BANDS[:4])}, seed=7)
        self.assertNotEqual(full.identity_sha256(), partial.identity_sha256())
        self.assertEqual(partial.meta(S2A, 'B04'), ChannelMeta('B04', 2, 4))

    def test_unknown_sensor_fails_closed(self):
        registry = SensorRegistry(CATALOGS, seed=7)
        with self.assertRaises(KeyError):
            registry.get(SensorKey('S2', 'S2D'))
        with self.assertRaises(KeyError):
            registry.meta(SensorKey('S2', 'S2D'), 'B04')

    def test_deterministic_init_and_manifest_round_trip(self):
        first = SensorRegistry(CATALOGS, seed=7)
        second = SensorRegistry(CATALOGS, seed=7)
        other = SensorRegistry(CATALOGS, seed=8)
        self.assertEqual(first.parameter_digest(), second.parameter_digest())
        self.assertNotEqual(first.parameter_digest(), other.parameter_digest())
        rebuilt = SensorRegistry.from_manifest(first.manifest())
        self.assertEqual(rebuilt.identity_sha256(), first.identity_sha256())
        self.assertEqual(rebuilt.parameter_digest(), first.parameter_digest())
        broken = first.manifest()
        broken['schema'] = 'unknown'
        with self.assertRaises(ValueError):
            SensorRegistry.from_manifest(broken)

    def test_unused_descriptor_keeps_no_gradient_and_no_adamw_update(self):
        """CPU synthetic single step: separate from any real-data update ledger."""
        registry = SensorRegistry({S2A: ChannelCatalog.from_ordered(S2_BANDS),
                                   L8: ChannelCatalog.from_ordered([f'SR_B{i}' for i in range(1, 8)])}, seed=7)
        untouched = registry.get(L8).detach().clone()
        optimizer = torch.optim.AdamW(registry.parameters(), lr=1e-2, weight_decay=0.04)
        loss = registry.get(S2A).square().sum()
        loss.backward()
        self.assertIsNone(registry.get(L8).grad)
        changed = registry.get(S2A).detach().clone()
        optimizer.step()
        self.assertTrue(torch.equal(registry.get(L8).detach(), untouched))
        self.assertFalse(torch.equal(registry.get(S2A).detach(), changed))
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(registry.get(S2A).grad)


class MemoryTest(unittest.TestCase):
    def test_multi_slot_kv_from_one_descriptor(self):
        memory = SensorMemory()
        p = torch.randn(32, generator=torch.Generator().manual_seed(0), requires_grad=True)
        keys, values = memory(p)
        self.assertEqual(keys.shape, (8, 32))
        self.assertEqual(values.shape, (8, 32))
        self.assertTrue(torch.isfinite(keys).all() and torch.isfinite(values).all())
        self.assertFalse(torch.equal(keys, values))
        keys.sum().backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_descriptor_identity_changes_memory(self):
        memory = SensorMemory()
        first = memory(torch.zeros(32))[0]
        second = memory(torch.ones(32))[0]
        self.assertFalse(torch.equal(first, second))

    def test_batch_descriptors(self):
        memory = SensorMemory()
        keys, values = memory(torch.zeros(3, 32))
        self.assertEqual(keys.shape, (3, 8, 32))
        self.assertEqual(values.shape, (3, 8, 32))

    def test_invalid_descriptor_rejected(self):
        memory = SensorMemory()
        with self.assertRaises(ValueError):
            memory(torch.zeros(31))
        with self.assertRaises(ValueError):
            memory(torch.tensor([float('nan')] * 32))
        with self.assertRaises(ValueError):
            SensorMemory(slots=0)


if __name__ == '__main__':
    unittest.main()

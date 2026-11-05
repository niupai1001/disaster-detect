from dataclasses import replace
import unittest
import numpy as np
import torch
from .geometry_cache import NativeGeometryCache
from .test_model import band
from .model import Encoder
from . import layout,streaming


class GeometryCacheChecks(unittest.TestCase):
    def test_key_invalidation_copy_isolation_and_restore(self):
        b=band();cache=NativeGeometryCache(data_version={'source':'frozen'})
        first=cache(b,3.,(0,0));original=first['indices'].copy();first['indices'][:]=-99
        np.testing.assert_array_equal(cache(b,3.,(0,0))['indices'],original)
        self.assertEqual(cache.hits,1)
        key=cache.key(b,3.,(0,0))
        self.assertEqual(key,cache.key(replace(b,reflectance=b.reflectance+1),3.,(0,0)))
        self.assertNotEqual(key,NativeGeometryCache(data_version={'source':'other'}).key(b,3.,(0,0)))
        for changed,p,origin in [(replace(b,affine=(.37,0,0,0,-1.21,6)),3.,(0,0)),(replace(b,crs='EPSG:32649'),3.,(0,0)),(replace(b,valid_mask=~b.valid_mask),3.,(0,0)),(replace(b,visible=~b.valid_mask),3.,(0,0)),(b,3.1,(0,0)),(b,3.,(.1,0))]:
            self.assertNotEqual(key,cache.key(changed,p,origin))
        original_function=layout.native_layout
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with cache.installed():
                self.assertIs(streaming.native_layout,cache)
                raise RuntimeError('deliberate')
        self.assertIs(layout.native_layout,original_function);self.assertIs(streaming.native_layout,original_function)
        tiny=NativeGeometryCache(data_version='test',max_bytes=1);tiny(b,3.,(0,0))
        self.assertEqual(tiny.summary()['entries'],0)
        empty=replace(b,valid_mask=torch.zeros_like(b.valid_mask))
        self.assertIsNone(cache(empty,3.,(0,0)));self.assertIsNone(cache(empty,3.,(0,0)))
        bounded=NativeGeometryCache(data_version='test',max_bytes=cache.entries[key][1]+1024)
        for offset in (0.,.1,.2,.3):bounded(b,3.,(offset,0))
        self.assertLessEqual(bounded.bytes,bounded.max_bytes);self.assertGreater(bounded.evictions,0)

    def test_geometry_reuse_preserves_encoder_outputs_and_all_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(5);b=band()
        encoder=Encoder(hyper_hidden=64,local_batch_size=8,native_mode='grouped',layout_mode='indexed').eval()
        def run(source):
            encoder.zero_grad(set_to_none=True);z=encoder([source],patch_m=3.,origin_xy=(0,0)).tokens;z.square().mean().backward()
            return z.detach().clone(),[p.grad.clone() for p in encoder.parameters()]
        output,gradients=run(b);cache=NativeGeometryCache(data_version='frozen')
        with cache.installed():
            for _ in range(2):
                actual,grads=run(b);torch.testing.assert_close(actual,output,atol=0,rtol=0)
                for a,g in zip(grads,gradients):torch.testing.assert_close(a,g,atol=0,rtol=0)
            changed,_=run(replace(b,reflectance=b.reflectance+.3))
            self.assertFalse(torch.equal(changed,output))
        self.assertGreater(cache.hits,0)

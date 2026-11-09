import json
from pathlib import Path
import unittest
import torch
from encoder_v3.model import BandObservation, Encoder, SRFEncoder, ReflectanceDecoder, spatial_hide
from encoder_v3.gates import require_training


def band():
    return BandObservation(torch.rand(6,6),torch.ones(6,6,dtype=torch.bool),torch.tensor([450.,500.,550.]),torch.tensor([0.,1.,0.]),(10.,0.,0.,0.,-10.,60.),'EPSG:32631')


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gate(self):
        config=json.loads((Path(__file__).resolve().parents[2] / 'src/encoder_v3' / 'config.json').read_text())
        with self.assertRaises(ValueError): require_training(config)
        with self.assertRaises(ValueError): require_training({})

    def test_srf(self):
        encoder=SRFEncoder()
        x=torch.linspace(450,550,101); y=1-(x-500).abs()/50
        a=encoder(x,y); b=encoder(x[::2],y[::2])
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
        a.sum().backward()
        self.assertTrue(all(p.grad is not None for p in encoder.parameters()))
        with self.assertRaises(ValueError):encoder(x,torch.zeros_like(y))

    def test_hidden_nan_and_gradient(self):
        model=Encoder(hyper_hidden=64).eval(); source=band()
        hidden,target=spatial_hide(source,[0,0,30,60])
        a=model([hidden],patch_m=30,origin_xy=(0,0))
        for cells in a.receptive_cells:
            overlap=(torch.minimum(cells[:,2:],cells.new_tensor([30,60]))-torch.maximum(cells[:,:2],cells.new_tensor([0,0]))).gt(0).all(1)
            self.assertFalse(bool(overlap.any()))
        hidden.reflectance=source.reflectance.clone(); hidden.reflectance[~hidden.keep()]=torch.nan
        b=model([hidden],patch_m=30,origin_xy=(0,0))
        torch.testing.assert_close(a.tokens,b.tokens)
        self.assertEqual(a.tokens.shape,(2,128)); self.assertEqual(int(target.sum()),18)
        a.tokens.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.hyper.parameters()))

    def test_decoder_and_empty(self):
        model=Encoder(hyper_hidden=64).eval(); source=band()
        z=model([source],patch_m=30,origin_xy=(0,0))
        decoder=ReflectanceDecoder(); cells=source.cells()
        condition=model.srf(source.wavelengths_nm,source.response)
        out,valid=decoder(z,cells,condition,kind='B1')
        self.assertTrue(bool(valid.all())); self.assertTrue(bool(torch.isfinite(out).all()))
        out.mean().backward()
        self.assertIsNotNone(decoder.mlp[0].weight.grad)
        out,valid=decoder(z,cells+1000,condition,kind='B2')
        self.assertFalse(bool(valid.any())); self.assertTrue(bool(torch.isnan(out).all()))

    def test_native_noninteger_unique_assignment(self):
        source=band(); source.affine=(0.37,0,0,0,-1.21,10)
        z=Encoder(hyper_hidden=64).eval()([source],patch_m=1.5,origin_xy=(0,0))
        self.assertEqual(sum(len(c) for c in z.cells),source.reflectance.numel())


if __name__=='__main__': unittest.main()

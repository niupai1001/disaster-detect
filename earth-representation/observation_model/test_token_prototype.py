"""Critical native alignment/dependency checks; no optimizer or training."""
import unittest
import torch
from .model import BandRaster, NativeBandTokenizer
from .token_prototype import IndependentTokenizer, PatchModel, SIZE, patches, unpatch


class NativeTokenTests(unittest.TestCase):
    def test_regions_and_old_dependency(self):
        torch.manual_seed(0);torch.set_num_threads(2)
        old=NativeBandTokenizer();new=IndependentTokenizer();new.load_state_dict(old.state_dict())
        self.old_counts={}
        for g in [10,20,30]:
            x=torch.zeros(1,SIZE//g,SIZE//g);valid=torch.ones_like(x,dtype=torch.bool)
            a=BandRaster('test',x,valid,800,30,g,(g/2,g/2));b=BandRaster('test',x.clone(),valid,800,30,g,(g/2,g/2));b.value[0,0,0]=1
            with torch.no_grad():
                original=(old(a,SIZE)[0]-old(b,SIZE)[0]).abs().amax(-1)>1e-6
                changed=(new(a,SIZE)[0]-new(b,SIZE)[0]).abs().amax(-1)>1e-6
            self.assertEqual(int(original.sum()),4);self.assertEqual(int(changed.sum()),1)
            numbered=torch.arange(x.numel()).reshape_as(x)
            self.assertTrue(torch.equal(unpatch(patches(numbered,g),g),numbered))

    def test_reject_partial_twenty_meter_support(self):
        h=torch.zeros(1,8,8,dtype=torch.bool);h[0,2,2]=True
        x=torch.zeros(1,SIZE//20,SIZE//20)
        band=BandRaster('test',x,torch.ones_like(x,dtype=torch.bool),800,30,20,(10,10))
        with self.assertRaisesRegex(ValueError,'whole60m'):
            PatchModel()([band],h)

    def test_a_masks_only_target_band(self):
        torch.manual_seed(0);model=PatchModel().eval()
        items=[BandRaster(str(g),torch.randn(1,SIZE//g,SIZE//g),torch.ones(1,SIZE//g,SIZE//g,dtype=torch.bool),800+g,30,g,(g/2,g/2)) for g in [10,20]]
        h=torch.zeros(1,2,8,8,dtype=torch.bool);h[:,0,2:4,2:4]=True
        with torch.no_grad():
            original=model(items,h)[0]
            items[0].value[:,6:12,6:12]+=10000
            self.assertTrue(torch.equal(original,model(items,h)[0]))
            items[1].value[:,3:6,3:6]+=.7
            self.assertGreater(float((original-model(items,h)[0]).abs().max()),1e-7)


if __name__=='__main__':unittest.main()

import copy
import unittest
import torch
from torch import nn
from .complete_attention import query_chunked_transformer


def network():
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',batch_first=True,norm_first=True),4,enable_nested_tensor=False)


class CompleteAttentionChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(23)

    def test_all_outputs_inputs_and_parameters_keep_gradients(self):
        reference=network().double().train();candidate=copy.deepcopy(reference)
        x=torch.randn(19,128,dtype=torch.float64,requires_grad=True);weights=torch.randn_like(x)
        expected=reference(x[None])[0];(expected*weights).sum().backward()
        for chunk in (1,7,30):
            candidate.zero_grad(set_to_none=True);z=x.detach().clone().requires_grad_();rng_before=torch.get_rng_state().clone();actual=query_chunked_transformer(candidate,z,query_chunk_size=chunk);(actual*weights).sum().backward()
            torch.testing.assert_close(actual,expected,atol=2e-13,rtol=2e-13)
            torch.testing.assert_close(z.grad,x.grad,atol=4e-12,rtol=4e-12)
            for a,b in zip(reference.parameters(),candidate.parameters(),strict=True):torch.testing.assert_close(a.grad,b.grad,atol=4e-12,rtol=4e-12)
            self.assertTrue(torch.equal(rng_before,torch.get_rng_state()))

    def test_full_interaction_and_observation_isolation(self):
        model=network().double().eval();x=torch.randn(17,128,dtype=torch.float64)
        with torch.no_grad():
            a=query_chunked_transformer(model,x,query_chunk_size=3);changed=x.clone();changed[-1]+=torch.linspace(-3,3,128);b=query_chunked_transformer(model,changed,query_chunk_size=3)
        self.assertGreater(float((a[0]-b[0]).abs().max()),1e-6)
        with self.assertRaises(ValueError):query_chunked_transformer(model,x[None].expand(2,-1,-1),query_chunk_size=3)
        with self.assertRaises(ValueError):query_chunked_transformer(model,x,query_chunk_size=0)

    def test_activation_storage_excludes_quadratic_scores(self):
        model=network().double().train();x=torch.randn(71,128,dtype=torch.float64,requires_grad=True);shapes=[]
        def pack(t):shapes.append(tuple(t.shape));return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            out=query_chunked_transformer(model,x,query_chunk_size=9)
        self.assertFalse(any(len(shape)>=3 and shape[-1]==71 and (shape[-2]==71 or 0<shape[-2]<=9) for shape in shapes));out.square().sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_adaptive_encoder_integration_has_same_parameters(self):
        from dataclasses import replace
        from .adaptive_window import AdaptiveWindowEncoder
        from .test_adaptive_window import source
        b=source();b=replace(b,reflectance=b.reflectance.double(),wavelengths_nm=b.wavelengths_nm.double(),response=b.response.double())
        a=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64).double().train();c=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64,global_query_chunk_size=7).double().train();c.load_state_dict(a.state_dict())
        x=a([b],origin_xy=(100.,180.));y=c([b],origin_xy=(100.,180.));torch.testing.assert_close(x.tokens,y.tokens,atol=3e-12,rtol=3e-12)
        weights=torch.randn_like(x.tokens);(x.tokens*weights).sum().backward();(y.tokens*weights).sum().backward()
        self.assertEqual(list(a.state_dict()),list(c.state_dict()))
        for p,q in zip(a.parameters(),c.parameters(),strict=True):torch.testing.assert_close(p.grad,q.grad,atol=4e-11,rtol=4e-11)

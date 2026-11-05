import unittest
import torch
from .data import Observation
from .model import Encoder,ReflectanceDecoder
from .tasks import spatial_task,spectral_task
from .streaming import prepare_task
from .joint import PlannedView,accumulate_joint,layout
from .test_model import band


class JointChecks(unittest.TestCase):
    def views(self,dtype=torch.float32):
        result=[]
        for index in range(8):
            values=[band(),band()]
            for i,value in enumerate(values):
                value.record_id=str(i)
                value.reflectance=value.reflectance.to(dtype);value.wavelengths_nm=value.wavelengths_nm.to(dtype);value.response=value.response.to(dtype)
            observation=Observation(values,dict(id=str(index),role='TRAIN'))
            for task in (spatial_task(observation,(0,0,20,60)),spectral_task(observation,seed=0,update=0)):
                plan=prepare_task(task,patch_m=20,origin_xy=(0,0),chunk_pixels=5)
                result.append(PlannedView(task,plan,{'0':.1,'1':.2},'S2' if index<6 else 'Landsat'))
        return result

    def test_batch_source_weights_and_order_invariance(self):
        torch.set_num_threads(2);views=self.views();_,weights=layout(views)
        self.assertAlmostEqual(weights[id(views[0])],1/12)
        self.assertAlmostEqual(weights[id(views[-1])],1/4)
        a=Encoder(hyper_hidden=64,local_batch_size=8,layout_mode='indexed').eval();head=ReflectanceDecoder(indexed=True)
        b=Encoder(hyper_hidden=64,local_batch_size=8,layout_mode='indexed').eval();other=ReflectanceDecoder(indexed=True)
        b.load_state_dict(a.state_dict());other.load_state_dict(head.state_dict())
        x=accumulate_joint(a,head,views,logical_update=0,chunk_pixels=5,precision='fp32',clip_norm=None)
        y=accumulate_joint(b,other,list(reversed(views)),logical_update=0,chunk_pixels=19,precision='fp32',clip_norm=None)
        self.assertAlmostEqual(x['weighted_loss'],x['A_loss']+.25*x['B_loss'],places=5)
        self.assertAlmostEqual(x['weighted_loss'],y['weighted_loss'],places=4)
        self.assertIsNone(x['losses_by_kind']['B2']);self.assertEqual(x['effective_batch'],8)
        for p,q in zip(list(a.parameters())+list(head.parameters()),list(b.parameters())+list(other.parameters())):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=3e-4,rtol=3e-4)

    def test_incomplete_effective_batch_rejected(self):
        with self.assertRaisesRegex(ValueError,'eight distinct'):layout(self.views()[:-2])

    def test_microbatch_loss_gradient_and_update_equivalence(self):
        # Float64 isolates algebraic equivalence from FP32 Adam's sensitivity near zero.
        torch.set_num_threads(2);torch.manual_seed(23);views=self.views(torch.float64)
        reference=Encoder(hyper_hidden=64,local_batch_size=8,native_mode='grouped',layout_mode='indexed').double().eval()
        decoder=ReflectanceDecoder(indexed=True).double()
        initial={k:v.clone() for k,v in reference.state_dict().items()};head={k:v.clone() for k,v in decoder.state_dict().items()}
        baseline=accumulate_joint(reference,decoder,views,logical_update=123,chunk_pixels=5,precision='fp32',clip_norm=3.)
        parameters=list(reference.parameters())+list(decoder.parameters())
        gradients=[p.grad.clone() for p in parameters]
        torch.optim.AdamW(parameters,lr=3e-4).step()
        for microbatch,cell_indexed in [(m,c) for m in (2,4,8) for c in (False,True)]:
            with self.subTest(microbatch=microbatch,cell_indexed=cell_indexed):
                model=Encoder(hyper_hidden=64,local_batch_size=8,native_mode='grouped',layout_mode='indexed').double().eval();model.load_state_dict(initial)
                other=ReflectanceDecoder(indexed=True,cell_indexed=cell_indexed).double();other.load_state_dict(head)
                result=accumulate_joint(model,other,views,logical_update=123,chunk_pixels=19,precision='fp32',clip_norm=3.,microbatch_observations=microbatch)
                self.assertAlmostEqual(result['weighted_loss'],baseline['weighted_loss'],places=10)
                self.assertEqual(result['effective_batch'],8);self.assertEqual(result['execution_microbatch_observations'],microbatch)
                self.assertEqual(result['global_tokens'],baseline['global_tokens'])
                self.assertGreaterEqual(result['global_padding_fraction'],0)
                actual=list(model.parameters())+list(other.parameters())
                for p,g in zip(actual,gradients):torch.testing.assert_close(p.grad,g,atol=1e-10,rtol=1e-8)
                torch.optim.AdamW(actual,lr=3e-4).step()
                for p,q in zip(actual,parameters):torch.testing.assert_close(p,q,atol=1e-10,rtol=1e-8)

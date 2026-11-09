import copy,unittest
from dataclasses import replace
import torch
from tests.encoder_v3.test_adaptive_window import source
from encoder_v3.participation_encoder import ParticipationEncoder
from encoder_v3.participation_layout import AreaEquivalentHillRate
from tests.encoder_v3 import test_participation_joint as fixture_module
from encoder_v3.adaptive_joint import accumulate_adaptive_joint
from encoder_v3.model import ReflectanceDecoder
from encoder_v3.training import JEPAPredictor,make_teacher


def encoder():return ParticipationEncoder(rate_curve=AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556),selection_seed=0,hyper_hidden=64,local_batch_size=64)


def flat_grads(modules):
    return torch.cat([torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten() for module in modules for p in module.parameters()])


class ParticipationBatchChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(191)

    def observations(self):
        result=[]
        for i,(g,h,w) in enumerate(((.5,8,9),(.5,8,9),(1.21,9,7),(20.,7,8))):
            b=replace(source(g,g),reflectance=torch.rand(h,w),valid_mask=torch.ones(h,w,dtype=torch.bool),crs='EPSG:32631' if i%2==0 else 'EPSG:32632');b.valid_mask[0,0]=False
            result.append(([b],(100.13,201.27-h*g)))
        return result

    def test_variable_lengths_metadata_outputs_and_gradient(self):
        inputs=self.observations();a=encoder().eval();b=copy.deepcopy(a);reference=[a(bands,origin_xy=origin) for bands,origin in inputs];actual=b.forward_batch(inputs)
        for want,got in zip(reference,actual,strict=True):
            torch.testing.assert_close(got.tokens,want.tokens,atol=2e-6,rtol=2e-6)
            self.assertEqual(got.participation_audit,want.participation_audit);self.assertEqual(got.group_ids,want.group_ids);self.assertEqual(got.crs,want.crs)
            for name in ('cells','coverage_cells','visible_cells','receptive_cells'):
                for x,y in zip(getattr(got,name),getattr(want,name),strict=True):torch.testing.assert_close(x,y,atol=0,rtol=0)
        self.assertGreater(len({len(x.tokens) for x in actual}),1)
        target=torch.linspace(-.5,.5,128)
        sum((i+1)*(x.tokens.sin()*target).mean() for i,x in enumerate(reference)).backward();sum((i+1)*(x.tokens.sin()*target).mean() for i,x in enumerate(actual)).backward()
        x,y=flat_grads([a]),flat_grads([b]);self.assertLess(float((x-y).norm()/x.norm()),2e-6)
        self.assertEqual(a.get_extra_state(),b.get_extra_state())

    def test_no_cross_observation_attention_or_input_gradient(self):
        inputs=self.observations();model=encoder().eval();reference=model.forward_batch(inputs)
        changed=copy.deepcopy(inputs);changed[1][0][0].reflectance.add_(3.);actual=model.forward_batch(changed)
        for i in (0,2,3):torch.testing.assert_close(actual[i].tokens,reference[i].tokens,atol=0,rtol=0)
        for bands,_ in inputs:
            for band in bands:band.reflectance.requires_grad_(True)
        output=model.forward_batch(inputs);output[0].tokens.sin().mean().backward()
        self.assertGreater(int(torch.count_nonzero(inputs[0][0][0].reflectance.grad)),0)
        for bands,_ in inputs[1:]:self.assertEqual(int(torch.count_nonzero(bands[0].reflectance.grad)),0)
        with self.assertRaises(ValueError):model.forward_batch([])
        with self.assertRaises(ValueError):model.forward_batch([(inputs[0][0],120.,inputs[0][1])])
        bad=copy.deepcopy(inputs);bad[0][0][0].valid_mask.zero_()
        with self.assertRaisesRegex(ValueError,'No valid visible'):model.forward_batch(bad)

    def test_joint_batch_8_source_normalization_and_jepa_isolation(self):
        fixtures=fixture_module.ParticipationJointChecks();views,paired=fixtures.fixtures();model=encoder();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True);predictor=JEPAPredictor();teacher=make_teacher(model)
        options=dict(logical_update=2000,chunk_pixels=7,clip_norm=None,jepa_views=paired,teacher=teacher,predictor=predictor,jepa_weight=.05,latent_loss_fn=lambda p,t,b:(p-t).square().mean(),curriculum_end_update=2000,ab_stable=True)
        reference=accumulate_adaptive_joint(model,decoder,views,**options);expected=flat_grads([model,decoder,predictor]).clone()
        for size in (2,4,8):
            actual=accumulate_adaptive_joint(model,decoder,views,**options,microbatch_observations=size);gradient=flat_grads([model,decoder,predictor])
            self.assertEqual(actual['effective_batch'],8);self.assertEqual(actual['execution_microbatch_observations'],size);self.assertEqual(actual['teacher_target_ids'],reference['teacher_target_ids'])
            self.assertAlmostEqual(actual['weighted_loss'],reference['weighted_loss'],places=5);self.assertLess(float((gradient-expected).norm()/expected.norm()),2e-6)
            self.assertEqual([x['exposure'] for x in actual['AB']],[x['exposure'] for x in reference['AB']]);self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_microbatch_8_joint_update_ema_and_exact_resume(self):
        from functools import partial
        from unittest.mock import patch
        from tests.encoder_v3 import test_adaptive_joint as reference_tests
        with patch.object(reference_tests,'accumulate_adaptive_joint',partial(accumulate_adaptive_joint,microbatch_observations=8)):
            fixture_module.ParticipationJointChecks().test_full_joint_update_ema_and_next_step_restore()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_microbatch_outputs_and_no_cross_input_gradient(self):
        inputs=[]
        for bands,origin in self.observations():
            moved=[replace(b,reflectance=b.reflectance.cuda().requires_grad_(True),valid_mask=b.valid_mask.cuda(),wavelengths_nm=b.wavelengths_nm.cuda(),response=b.response.cuda()) for b in bands]
            inputs.append((moved,origin))
        model=encoder().cuda().eval();reference=[model(bands,origin_xy=origin) for bands,origin in inputs];actual=model.forward_batch(inputs)
        for want,got in zip(reference,actual,strict=True):torch.testing.assert_close(got.tokens,want.tokens,atol=2e-6,rtol=2e-6)
        actual[0].tokens.sin().mean().backward()
        self.assertGreater(int(torch.count_nonzero(inputs[0][0][0].reflectance.grad)),0)
        for bands,_ in inputs[1:]:self.assertEqual(int(torch.count_nonzero(bands[0].reflectance.grad)),0)

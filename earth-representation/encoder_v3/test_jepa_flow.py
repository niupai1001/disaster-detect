import unittest
from dataclasses import replace
import numpy as np
import torch
from .adaptive_window import AdaptiveRepresentation,AdaptiveWindowEncoder
from .jepa import teacher_support_queries,predict_teacher_targets,cross_source_latent_loss
from .training import JEPAPredictor,make_teacher,update_teacher
from .model import ReflectanceDecoder
from .data import Observation
from .test_adaptive_window import source


def representation(cells,visible=None):
    cells=[torch.tensor(c,dtype=torch.float64).reshape(-1,4) for c in cells];n=len(cells)
    bounds=torch.stack([torch.cat((c[:,:2].amin(0),c[:,2:].amax(0))) for c in cells]);centers=(bounds[:,:2]+bounds[:,2:])/2
    return AdaptiveRepresentation(tokens=torch.randn(n,128,requires_grad=True),cells=cells,centers=centers,bounds=bounds,valid=torch.ones(n,dtype=torch.bool),crs='EPSG:32650',gsd_xy=torch.ones(n,2),srf_condition=torch.zeros(n,32),source_bands=tuple(str(i) for i in range(n)),origin_xy=(0.,0.),visible_cells=cells if visible is None else visible)


class JepaFlowChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(29)

    def test_exact_visible_holes_duplicate_bands_and_no_index_pairing(self):
        support=[[[0,0,1,1],[2,0,3,1]]]
        a=representation(support*2)
        b=representation([[[1,0,2,1]],[[0,0,1,1]],[[2,0,4,1]],[[4,0,5,1]]])
        q=teacher_support_queries(a,b);self.assertEqual(q.indices.tolist(),[1,2]);np.testing.assert_allclose(q.overlap_m2.numpy(),[1.,1.]);np.testing.assert_allclose(q.overlap_fraction.numpy(),[1.,.5])
        a.visible_cells=[torch.empty(0,4),torch.empty(0,4)];q=teacher_support_queries(a,b);self.assertEqual(len(q.indices),0)

    def test_teacher_latent_values_do_not_construct_queries(self):
        a=representation([[[0,0,2,2]]]);b=representation([[[0,0,1,1]],[[1,0,2,1]]]);predictor=JEPAPredictor().eval()
        x,y,q=predict_teacher_targets(a,b,predictor);b.tokens=b.tokens*100+10;xx,yy,qq=predict_teacher_targets(a,b,predictor)
        torch.testing.assert_close(x,xx,atol=0,rtol=0);self.assertFalse(torch.equal(y,yy));self.assertFalse(yy.requires_grad)

    def test_joint_gradient_step_and_ema_isolation_on_synthetic_fixture(self):
        # This one update uses only generated tensors, never the stopped real pilot.
        b=source(.5,.5);other=source(1.21,1.21)
        source_obs=Observation([b],dict(id='synthetic_source',role='TRAIN'));target_obs=Observation([other],dict(id='synthetic_target',role='TRAIN'))
        student=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64);teacher=make_teacher(student);predictor=JEPAPredictor();decoder=ReflectanceDecoder()
        admission=dict(source_id='synthetic_source',target_id='synthetic_target',checks={key:True for key in ('space','registration','time_difference','surface_change','qa','srf','role_isolation')})
        before=[p.detach().clone() for p in student.parameters()];teacher_before=[p.detach().clone() for p in teacher.parameters()];decoder_before=[p.detach().clone() for p in decoder.parameters()]
        loss,report=cross_source_latent_loss(student,teacher,predictor,source_obs,target_obs,admission=admission,source_encoder_kwargs=dict(origin_xy=(100.,180.)),target_encoder_kwargs=dict(origin_xy=(100.,180.)),loss_fn=lambda p,t,bands:(p-t).square().mean(),logical_update=2000,curriculum_end_update=2000,ab_stable=True,weight=.05)
        self.assertGreater(report['supported_queries'],0);loss.backward()
        self.assertTrue(all(p.grad is not None for p in student.parameters()));self.assertTrue(all(p.grad is not None for p in predictor.parameters()));self.assertTrue(all(p.grad is None for p in teacher.parameters()));self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        opt=torch.optim.AdamW(list(student.parameters())+list(predictor.parameters())+list(decoder.parameters()),lr=1e-4);opt.step();update_teacher(teacher,student,.996)
        self.assertTrue(any(not torch.equal(p,b) for p,b in zip(student.parameters(),before)));self.assertTrue(all(torch.equal(p,b) for p,b in zip(decoder.parameters(),decoder_before)))
        for old,new,actual in zip(teacher_before,student.parameters(),teacher.parameters(),strict=True):torch.testing.assert_close(actual,old*.996+new*.004,atol=1e-7,rtol=1e-6)

    def test_start_admission_and_loss_are_fail_closed(self):
        b=source();obs=Observation([b],dict(id='fixture',role='TRAIN'));student=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64);teacher=make_teacher(student);predictor=JEPAPredictor()
        kwargs=dict(admission={},source_encoder_kwargs={},target_encoder_kwargs={},loss_fn=None,logical_update=1999,curriculum_end_update=2000,ab_stable=True,weight=.05)
        with self.assertRaisesRegex(ValueError,'completed A/B'):cross_source_latent_loss(student,teacher,predictor,obs,obs,**kwargs)
        kwargs.update(logical_update=2000)
        with self.assertRaisesRegex(ValueError,'reviewed latent loss'):cross_source_latent_loss(student,teacher,predictor,obs,obs,**kwargs)
        kwargs.update(loss_fn=lambda p,t,bands:(p-t).square().mean())
        with self.assertRaisesRegex(ValueError,'admission identity'):cross_source_latent_loss(student,teacher,predictor,obs,obs,**kwargs)

    def test_metric_translation_preserves_query_subpixel_precision(self):
        a=representation([[[0.,0.,2.,2.]]]);b=representation([[[.123,.234,1.333,1.444]]])
        before=teacher_support_queries(a,b);offset=torch.tensor([500000.123,7000000.321],dtype=torch.float64)
        for rep in (a,b):
            rep.centers=rep.centers+offset;rep.bounds=rep.bounds+offset.repeat(2);rep.cells=[c+offset.repeat(2) for c in rep.cells];rep.visible_cells=[c+offset.repeat(2) for c in rep.visible_cells];rep.origin_xy=tuple(offset.tolist())
        after=teacher_support_queries(a,b)
        torch.testing.assert_close(before.relative_xy,after.relative_xy,atol=1e-7,rtol=0);torch.testing.assert_close(before.support_xy,after.support_xy,atol=1e-7,rtol=0)

    def test_absent_support_excluded_and_teacher_config_checked(self):
        from unittest.mock import patch
        b=source();far=replace(source(),affine=(.5,0.,5000.,0.,-.5,6000.))
        a=Observation([b],dict(id='a',role='TRAIN'));t=Observation([far],dict(id='b',role='TRAIN'));student=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64);teacher=make_teacher(student);predictor=JEPAPredictor()
        admission=dict(source_id='a',target_id='b',checks={key:True for key in ('space','registration','time_difference','surface_change','qa','srf','role_isolation')})
        kwargs=dict(admission=admission,source_encoder_kwargs=dict(origin_xy=(100.,180.)),target_encoder_kwargs=dict(origin_xy=(100.,180.)),loss_fn=lambda p,t,bands:(p-t).square().mean(),logical_update=2000,curriculum_end_update=2000,ab_stable=True,weight=.05)
        with patch.object(predictor,'forward',side_effect=AssertionError('Unsupported query reached Predictor')):
            loss,report=cross_source_latent_loss(student,teacher,predictor,a,t,**kwargs)
        self.assertIsNone(loss);self.assertIsNone(report['loss']);self.assertEqual(report['supported_queries'],0)
        teacher.exponent=.4
        with self.assertRaisesRegex(ValueError,'configuration mismatch'):cross_source_latent_loss(student,teacher,predictor,a,t,**kwargs)

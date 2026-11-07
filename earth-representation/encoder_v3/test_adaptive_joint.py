import unittest,copy
from dataclasses import replace
from collections import Counter
import torch
from .model import ReflectanceDecoder
from .adaptive_window import AdaptiveWindowEncoder
from .adaptive_streaming import prepare_adaptive_task
from .adaptive_joint import PairedLatentView,accumulate_adaptive_joint
from .joint import PlannedView
from .tasks import spatial_task,spectral_task
from .data import Observation
from .training import make_teacher,JEPAPredictor
from .jepa import cross_source_latent_loss
from .test_model import band


class AdaptiveJointChecks(unittest.TestCase):
    def encoder(self):return AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64)
    def plan(self,task):return prepare_adaptive_task(task,exponent=.5,origin_xy=(0,0),chunk_pixels=7,cell_indexed=True)

    def fixtures(self):
        views=[];paired=[]
        for i in range(8):
            bands=[replace(band(),reflectance=torch.rand(3,4),valid_mask=torch.ones(3,4,dtype=torch.bool),record_id=str(j),affine=(10.,0.,0.,0.,-10.,30.)) for j in range(2)]
            obs=Observation(bands,dict(id=str(i),role='TRAIN'));label='S2' if i<3 else 'Landsat'
            for task in (spatial_task(obs,(0,0,10,20)),spectral_task(obs,seed=0,update=2000)):
                plan=self.plan(task);views.append(PlannedView(task,plan,{'0':.2,'1':.3},label))
            target=Observation([replace(bands[0],record_id='teacher')],dict(id='teacher_'+str(i),role='TRAIN'))
            admission=dict(source_id=obs.identifier,target_id=target.identifier,checks={k:True for k in ('space','registration','time_difference','surface_change','qa','srf','role_isolation')})
            paired.append(PairedLatentView(obs,target,admission,dict(origin_xy=(0,0)),dict(origin_xy=(0,0)),label))
        # One unmatched target is excluded; unequal remaining source counts are 3/4.
        paired[-1].target.bands=[replace(paired[-1].target.bands[0],affine=(10.,0.,1000.,0.,-10.,1030.))]
        return views,paired

    def test_joint_gradient_equals_explicit_source_balanced_sum(self):
        torch.set_num_threads(2);torch.manual_seed(51);views,paired=self.fixtures()
        model=self.encoder();head=ReflectanceDecoder(indexed=True,cell_indexed=True);predictor=JEPAPredictor();teacher=make_teacher(model)
        options=dict(logical_update=2000,chunk_pixels=7,clip_norm=None)
        baseline=accumulate_adaptive_joint(model,head,views,**options)
        decoder_grad=[p.grad.clone() for p in head.parameters()]
        counts=Counter(v.loss_source for v in paired[:-1]);criterion=lambda p,t,bands:(p-t).square().mean()
        for view in paired[:-1]:
            loss,_=cross_source_latent_loss(model,teacher,predictor,view.source,view.target,admission=view.admission,source_encoder_kwargs=view.source_encoder_kwargs,target_encoder_kwargs=view.target_encoder_kwargs,loss_fn=criterion,logical_update=2000,curriculum_end_update=2000,ab_stable=True,weight=.05)
            (loss/(len(counts)*counts[view.loss_source])).backward()
        expected=[None if p.grad is None else p.grad.clone() for p in list(model.parameters())+list(head.parameters())+list(predictor.parameters())]
        result=accumulate_adaptive_joint(model,head,views,**options,jepa_views=paired,teacher=teacher,predictor=predictor,jepa_weight=.05,latent_loss_fn=criterion,curriculum_end_update=2000,ab_stable=True)
        self.assertEqual(result['effective_batch'],8);self.assertIsNone(result['JEPA'][-1]['result']['loss']);self.assertAlmostEqual(result['AB_weighted_loss'],baseline['AB_weighted_loss'],places=5)
        for actual,want in zip(list(model.parameters())+list(head.parameters())+list(predictor.parameters()),expected,strict=True):
            if want is None:self.assertIsNone(actual.grad)
            else:torch.testing.assert_close(actual.grad,want,atol=3e-5,rtol=3e-5)
        for p,want in zip(head.parameters(),decoder_grad):torch.testing.assert_close(p.grad,want,atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()));self.assertTrue(all(len(row['exposure']['task_signature'])==64 for row in result['AB']));self.assertTrue(all(len(row['pair_signature'])==64 for row in result['JEPA']))

    def test_batch_and_start_conditions_fail_before_computation(self):
        torch.set_num_threads(2);views,paired=self.fixtures();model=self.encoder();head=ReflectanceDecoder()
        with self.assertRaisesRegex(ValueError,'eight distinct'):accumulate_adaptive_joint(model,head,views[:-2],logical_update=0,chunk_pixels=7,clip_norm=None)
        with self.assertRaisesRegex(ValueError,'reviewed loss'):accumulate_adaptive_joint(model,head,views,logical_update=0,chunk_pixels=7,clip_norm=None,jepa_views=paired)

    def test_full_joint_update_ema_and_next_step_restore(self):
        import tempfile
        from pathlib import Path
        from .training import update_teacher,save_checkpoint,restore_checkpoint,CheckedPlateau
        torch.set_num_threads(2);torch.manual_seed(61);views,paired=self.fixtures()
        model=self.encoder();head=ReflectanceDecoder(indexed=True,cell_indexed=True);predictor=JEPAPredictor();teacher=make_teacher(model)
        modules=dict(encoder=model,decoder=head,predictor=predictor,teacher=teacher);parameters=list(model.parameters())+list(head.parameters())+list(predictor.parameters());optimizer=torch.optim.AdamW(parameters,lr=1e-4);scheduler=CheckedPlateau(optimizer,threshold=0.)
        options=dict(chunk_pixels=7,clip_norm=3.,jepa_views=paired,teacher=teacher,predictor=predictor,jepa_weight=.05,latent_loss_fn=lambda p,t,b:(p-t).square().mean(),curriculum_end_update=2000,ab_stable=True)
        def step(update):
            metrics=accumulate_adaptive_joint(model,head,views,logical_update=update,**options);optimizer.step();update_teacher(teacher,model,.996);return metrics
        step(2000)
        frozen=dict(fixture='generated_data_only',members=[str(i) for i in range(8)],merge=model.get_extra_state())
        sampler=dict(next_update=2001,members=[str(i) for i in range(8)],hidden=[v.task.metadata.get('hidden_band_ids') for v in views])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'joint.pt';save_checkpoint(path,modules=modules,optimizer=optimizer,scheduler=scheduler,logical_update=2001,sampler=sampler,frozen_ids=frozen)
            rng_expected=torch.rand(3);expected_metrics=step(2001);expected={name:copy.deepcopy(module.state_dict()) for name,module in modules.items()};moments=copy.deepcopy(optimizer.state_dict())
            state=restore_checkpoint(path,modules=modules,optimizer=optimizer,scheduler=scheduler,frozen_ids=frozen)
            self.assertEqual(state['sampler'],sampler);torch.testing.assert_close(torch.rand(3),rng_expected,atol=0,rtol=0);actual_metrics=step(2001)
            self.assertEqual(expected_metrics['weighted_loss'],actual_metrics['weighted_loss'])
            for name,module in modules.items():
                for key,value in module.state_dict().items():
                    if isinstance(value,torch.Tensor):torch.testing.assert_close(value,expected[name][key],atol=0,rtol=0)
                    else:self.assertEqual(value,expected[name][key])
            for key,value in optimizer.state_dict()['state'].items():
                for field,tensor in value.items():
                    if isinstance(tensor,torch.Tensor):torch.testing.assert_close(tensor,moments['state'][key][field],atol=0,rtol=0)
                    else:self.assertEqual(tensor,moments['state'][key][field])
        with self.assertRaisesRegex(ValueError,'2000-update'):
            accumulate_adaptive_joint(model,head,views,logical_update=1000,**dict(options,curriculum_end_update=1000))

    def test_cuda_joint_forward_backward_without_optimizer_step(self):
        from .cuda_probe import probe_cuda
        from .data import to_device
        from .tasks import Target
        if not probe_cuda()['available']:self.skipTest('CUDA unavailable: joint device integration unverified')
        torch.set_num_threads(2);torch.manual_seed(71);views,paired=self.fixtures();device='cuda:0'
        moved=[]
        for v in views:
            visible=to_device(Observation(v.task.visible,v.task.metadata),device).bands
            targets=[Target(to_device(Observation([t.band],v.task.metadata),device).bands[0],t.mask.to(device)) for t in v.task.targets]
            moved.append(replace(v,task=replace(v.task,visible=visible,targets=targets)))
        paired=[replace(v,source=to_device(v.source,device),target=to_device(v.target,device)) for v in paired]
        model=self.encoder().to(device);head=ReflectanceDecoder(indexed=True,cell_indexed=True).to(device);predictor=JEPAPredictor().to(device);teacher=make_teacher(model)
        result=accumulate_adaptive_joint(model,head,moved,logical_update=2000,chunk_pixels=7,clip_norm=3.,jepa_views=paired,teacher=teacher,predictor=predictor,jepa_weight=.05,latent_loss_fn=lambda p,t,b:(p-t).square().mean(),curriculum_end_update=2000,ab_stable=True)
        self.assertEqual(result['effective_batch'],8);self.assertEqual(result['optimizer_updates'],0);self.assertEqual(sum(row['result']['loss'] is not None for row in result['JEPA']),7)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for module in (model,head,predictor) for p in module.parameters()));self.assertTrue(all(p.grad is None for p in teacher.parameters()))

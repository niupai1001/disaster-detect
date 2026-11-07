import copy,unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from .data import Observation,to_device
from .model import ReflectanceDecoder
from .tasks import spatial_task,fixed_spectral_task
from .participation_encoder import ParticipationEncoder,prepare_participation_task
from .participation_layout import AreaEquivalentHillRate
from .adaptive_streaming import stream_adaptive_task
from .test_adaptive_window import source
from .test_participation_batch import flat_grads
from . import joint


def curve():return AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
def model():return ParticipationEncoder(rate_curve=curve(),selection_seed=0,hyper_hidden=64,packed_local_output=True)
def prepare(task,retain=True,patch_m=60.):return prepare_participation_task(task,rate_curve=curve(),selection_seed=0,origin_xy=(99.8,180.),chunk_pixels=29,cell_indexed=True,retain_layouts=retain,physical_patch_m=patch_m)


class RetainedGeometryChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(33)
    def observation(self):
        bands=[replace(source(.5,1.21),record_id=str(i),wavelengths_nm=source().wavelengths_nm+i*100) for i in range(3)]
        bands[0].valid_mask[0,0]=False
        return Observation(bands,dict(id='synthetic',role='TRAIN',sensor='fixture',geometry_data_version='explicit-fixture-v1'))

    def test_single_outputs_support_and_gradients_exact_without_recomputation(self):
        obs=self.observation()
        for task in (spatial_task(obs,(102.,191.,105.,196.)),fixed_spectral_task(obs,['1','2'])):
            cached=prepare(task);reference=prepare(task,False);encoder=model();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True)
            options=dict(train_stds={str(i):1. for i in range(3)},chunk_pixels=31,loss_weight=1.,backward=True)
            expected=stream_adaptive_task(encoder,decoder,task,reference,**options);grads=flat_grads([encoder,decoder]).clone()
            encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
            with patch('encoder_v3.participation_encoder.participation_layout',side_effect=AssertionError('Unexpected geometry recomputation')):actual=stream_adaptive_task(encoder,decoder,task,cached,**options)
            self.assertEqual(actual,expected);torch.testing.assert_close(flat_grads([encoder,decoder]),grads,atol=0,rtol=0)
            a=encoder(task.visible,origin_xy=reference.origin_xy);b=encoder(task.visible,origin_xy=cached.origin_xy,geometry_plan=cached)
            torch.testing.assert_close(a.tokens,b.tokens,atol=0,rtol=0);self.assertEqual(a.participation_audit,b.participation_audit);self.assertEqual(a.group_ids,b.group_ids)
            for key in ('cells','coverage_cells','visible_cells','receptive_cells'):
                for x,y in zip(getattr(a,key),getattr(b,key),strict=True):torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_versions_masks_grids_origin_curve_and_patch_invalidate(self):
        task=fixed_spectral_task(self.observation(),['2']);plan=prepare(task);encoder=model();plan.validate_retained(task)
        for key,value in (('geometry_data_version','v2'),('cache_sha256','changed')):
            changed=replace(task,metadata=dict(task.metadata,**{key:value}))
            with self.assertRaisesRegex(ValueError,'data version'):plan.validate_retained(changed)
        changed=copy.deepcopy(plan);changed.physical_patch_m=120.
        with self.assertRaisesRegex(ValueError,'physical Patch'):changed.validate_retained(task)
        for field in ('affine','crs','valid_mask','visible','wavelengths_nm'):
            bands=copy.deepcopy(task.visible);band=bands[0]
            if field=='affine':band.affine=(band.affine[0]*1.01,*band.affine[1:])
            elif field=='crs':band.crs='EPSG:32631'
            elif field=='valid_mask':band.valid_mask[1,1]=~band.valid_mask[1,1]
            elif field=='visible':band.visible=band.valid_mask.clone();band.visible[1,1]=False
            else:band.wavelengths_nm=band.wavelengths_nm+1
            with self.assertRaisesRegex(ValueError,'Retained Merge geometry'):encoder(bands,origin_xy=plan.origin_xy,geometry_plan=plan)
        with self.assertRaises(ValueError):encoder(task.visible,origin_xy=(99.9,180.),geometry_plan=plan)
        changed_model=model();changed_model.selection_seed=1
        with self.assertRaises(ValueError):changed_model(task.visible,origin_xy=plan.origin_xy,geometry_plan=plan)
        missing=replace(task,metadata={k:v for k,v in task.metadata.items() if k!='geometry_data_version'})
        with self.assertRaisesRegex(ValueError,'data version required'):prepare(missing)
        p60,p120=prepare(task,patch_m=60.),prepare(task,patch_m=120.)
        self.assertNotEqual(p60.metadata_signature,p120.metadata_signature)
        for left,right in zip(p60.retained_layouts,p120.retained_layouts,strict=True):
            self.assertEqual(left['lengths'],right['lengths']);self.assertEqual(left['group_ids'],right['group_ids'])
            torch.testing.assert_close(torch.from_numpy(left['indices']),torch.from_numpy(right['indices']),atol=0,rtol=0)

    def test_no_learned_tensors_cached_and_synthetic_update_reuses_only_geometry(self):
        task=fixed_spectral_task(self.observation(),['2']);plan=prepare(task)
        def check(value):
            self.assertNotIsInstance(value,torch.Tensor)
            if isinstance(value,dict):
                for v in value.values():check(v)
            elif isinstance(value,(tuple,list)):
                for v in value:check(v)
        check(plan.retained_layouts)
        a=model();b=copy.deepcopy(a);da=ReflectanceDecoder(indexed=True,cell_indexed=True);db=copy.deepcopy(da)
        oa=torch.optim.AdamW(list(a.parameters())+list(da.parameters()),lr=1e-4);ob=torch.optim.AdamW(list(b.parameters())+list(db.parameters()),lr=1e-4)
        options=dict(train_stds={str(i):1. for i in range(3)},chunk_pixels=29,loss_weight=1.,backward=True)
        before=b(task.visible,origin_xy=plan.origin_xy,geometry_plan=plan).tokens.detach().clone()
        for _ in range(2):
            oa.zero_grad();ob.zero_grad();x=stream_adaptive_task(a,da,task,prepare(task,False),**options);y=stream_adaptive_task(b,db,task,plan,**options)
            self.assertEqual(x,y);torch.testing.assert_close(flat_grads([a,da]),flat_grads([b,db]),atol=0,rtol=0);oa.step();ob.step()
            for pa,pb in zip(list(a.parameters())+list(da.parameters()),list(b.parameters())+list(db.parameters()),strict=True):torch.testing.assert_close(pa,pb,atol=0,rtol=0)
        self.assertFalse(torch.equal(before,b(task.visible,origin_xy=plan.origin_xy,geometry_plan=plan).tokens))

    def test_joint_microbatches_preserve_hierarchical_losses_and_gradients(self):
        from .test_participation_joint import ParticipationJointChecks
        views,_=ParticipationJointChecks().fixtures();encoder=model();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True)
        for view in views:view.task.metadata=dict(view.task.metadata,geometry_data_version='joint-fixture-v1')
        reference=joint.accumulate_joint(encoder,decoder,views,logical_update=2000,chunk_pixels=11,precision='fp32',clip_norm=None);expected=flat_grads([encoder,decoder]).clone()
        for view in views:view.plan=prepare_participation_task(view.task,rate_curve=curve(),selection_seed=0,origin_xy=view.plan.origin_xy,chunk_pixels=11,cell_indexed=True,retain_layouts=True,physical_patch_m=120.)
        for size in (1,2,4,8):
            actual=joint.accumulate_joint(encoder,decoder,views,logical_update=2000,chunk_pixels=11,precision='fp32',clip_norm=None,microbatch_observations=size)
            self.assertAlmostEqual(actual['weighted_loss'],reference['weighted_loss'],places=6);self.assertEqual(actual['global_tokens'],reference['global_tokens']);self.assertEqual(actual['global_padding_fraction'],0)
            exposure=lambda report:{(r['observation'],r['kind']):r['exposure'] for r in report['rows']}
            self.assertEqual(exposure(actual),exposure(reference))
            self.assertLess(float((flat_grads([encoder,decoder])-expected).norm()/expected.norm()),2e-6)

    def test_synthetic_joint_optimizer_ema_and_exact_next_step_resume(self):
        from .test_participation_joint import ParticipationJointChecks
        class RetainedJointFixture(ParticipationJointChecks):
            def plan(self,task):
                task.metadata=dict(task.metadata,geometry_data_version='synthetic-resume-v1')
                return prepare_participation_task(task,rate_curve=self.curve(),selection_seed=0,origin_xy=(0,0),chunk_pixels=7,cell_indexed=True,retain_layouts=True,physical_patch_m=120.)
        RetainedJointFixture().test_full_joint_update_ema_and_next_step_restore()

    def test_empty_and_large_band_count_noninteger_gsd(self):
        obs=self.observation();empty=replace(obs.bands[0],valid_mask=torch.zeros_like(obs.bands[0].valid_mask));obs.bands=[empty]+[replace(source(1.21,1.21),record_id=str(i)) for i in range(1,13)];task=fixed_spectral_task(obs,['12']);plan=prepare(task);encoder=model();a=encoder(task.visible,origin_xy=plan.origin_xy);b=encoder(task.visible,origin_xy=plan.origin_xy,geometry_plan=plan)
        torch.testing.assert_close(a.tokens,b.tokens,atol=0,rtol=0);self.assertEqual(b.participation_audit[0]['status'],'NO_VISIBLE_INPUT')
        obs.bands=[replace(band,valid_mask=torch.zeros_like(band.valid_mask)) for band in obs.bands]
        self.assertIsNone(prepare(fixed_spectral_task(obs,['12'])))

    def test_hyperspectral_band_count_keeps_all_visible_members(self):
        base=replace(source(1.21,1.21),reflectance=torch.rand(2,3),valid_mask=torch.ones(2,3,dtype=torch.bool))
        obs=Observation([replace(base,record_id=str(i)) for i in range(165)],dict(id='hyper-fixture',geometry_data_version='synthetic-165',sensor='fixture'))
        task=fixed_spectral_task(obs,['164']);plan=prepare(task);encoder=model();a=encoder(task.visible,origin_xy=plan.origin_xy);b=encoder(task.visible,origin_xy=plan.origin_xy,geometry_plan=plan)
        self.assertEqual(len(plan.retained_layouts),164);self.assertEqual(len(b.participation_audit),164);torch.testing.assert_close(a.tokens,b.tokens,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_outputs_gradients_and_hidden_isolation(self):
        obs=self.observation();task=spatial_task(obs,(102.,191.,105.,196.));plan=prepare(task);moved=to_device(Observation(task.visible,task.metadata),'cuda:0');encoder=model().cuda();a=encoder(moved.bands,origin_xy=plan.origin_xy);a.tokens.sin().mean().backward();expected=flat_grads([encoder]).clone();encoder.zero_grad(set_to_none=True)
        changed=[replace(band,reflectance=band.reflectance.clone()) for band in moved.bands]
        for band in changed:band.reflectance[~band.keep()]=float('nan')
        with patch('encoder_v3.participation_encoder.participation_layout',side_effect=AssertionError('Unexpected recomputation')):b=encoder(changed,origin_xy=plan.origin_xy,geometry_plan=plan)
        b.tokens.sin().mean().backward();torch.testing.assert_close(a.tokens,b.tokens,atol=0,rtol=0);torch.testing.assert_close(flat_grads([encoder]),expected,atol=0,rtol=0)

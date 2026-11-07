import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from dataclasses import replace
import torch
from .data import Observation,to_device
from .tasks import spatial_task,cross_source_task
from .pair_geometry import reviewed_pair_task,frame_for_task,observation_version,canonical_sha,QueryFrameDecoder
from .participation_encoder import ParticipationEncoder,prepare_participation_task
from .participation_layout import AreaEquivalentHillRate
from .adaptive_streaming import stream_adaptive_task
from .adaptive_joint import PairedLatentView,accumulate_adaptive_joint
from .model import ReflectanceDecoder
from .training import make_teacher,JEPAPredictor
from .jepa import cross_source_latent_loss,teacher_support_queries
from .test_model import band
from .test_participation_batch import flat_grads
from .joint import PlannedView


def curve():return AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
def encoder():return ParticipationEncoder(rate_curve=curve(),selection_seed=0,hyper_hidden=64,packed_local_output=True)
def origin(obs):return (obs.bands[0].affine[2],0.)
def plan(task):return prepare_participation_task(task,rate_curve=curve(),selection_seed=0,origin_xy=(task.visible[0].affine[2],0.),chunk_pixels=7,cell_indexed=True,retain_layouts=True,physical_patch_m=60.)


def fixture(root,label='0',reverse=False):
    s_id,t_id=('target_'+label,label) if reverse else (label,'target_'+label)
    def obs(identifier,left,sensor):
        bands=[replace(band(),record_id=str(j),reflectance=torch.rand(3,4),valid_mask=torch.ones(3,4,dtype=torch.bool),affine=(10.,0.,left,0.,-10.,30.)) for j in range(2)]
        return Observation(bands,dict(id=identifier,role='TRAIN',sensor=sensor,geometry_data_version='synthetic-'+identifier,frozen_source_ids=dict(fixture='generated-only')))
    s=obs(s_id,0.,'S2');l=obs(t_id,20.,'Landsat')
    payload=dict(mechanism='query_offset_native_encoder_v1',S2_id=s.identifier,Landsat_id=l.identifier,translation_m=[20.,0.],crs='EPSG:32650',role='TRAIN',versions={s.identifier:observation_version(s),l.identifier:observation_version(l)})
    # Use the fixture's actual metric CRS; the source values are entirely generated.
    payload['crs']=s.bands[0].crs
    path=Path(root)/('review_'+label+'.json');path.write_text(json.dumps(dict(status='USER_CONFIRMED_PAIR_OFFSET_VALUES',record_sha256=canonical_sha(payload))))
    record=dict(payload,offset_review_file=str(path),offset_review_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source,target=(l,s) if reverse else (s,l)
    admission=dict(source_id=source.identifier,target_id=target.identifier,checks={k:True for k in ('space','registration','time_difference','surface_change','qa','srf','role_isolation')},pair_geometry=record)
    return source,target,admission


def joint_fixture(root):
    views=[];pairs=[]
    for i in range(8):
        source,target,admission=fixture(root,str(i),reverse=i>=4);left=source.bands[0].affine[2]
        for task in (spatial_task(source,(left,0.,left+10.,20.)),reviewed_pair_task(source,target,admission)):
            views.append(PlannedView(task,plan(task),{'0':1.,'1':1.},source.metadata['sensor']))
        pairs.append(PairedLatentView(source,target,admission,dict(origin_xy=origin(source)),dict(origin_xy=origin(target)),source.metadata['sensor']))
    return views,pairs


class PairGeometryChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(81);self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def test_bidirectional_query_sign_native_encoder_and_original_targets(self):
        for reverse in (False,True):
            source,target,admission=fixture(self.temp.name,str(reverse),reverse);task=reviewed_pair_task(source,target,admission);p=plan(task);m=encoder();d=ReflectanceDecoder(indexed=True,cell_indexed=True)
            self.assertEqual(p.pair_query_frame.query_offset_xy,(20.,0.) if reverse else (-20.,-0.))
            a=m(source.bands,origin_xy=origin(source));b=m(task.visible,origin_xy=p.origin_xy,geometry_plan=p);torch.testing.assert_close(a.tokens,b.tokens,atol=0,rtol=0);self.assertEqual(a.group_ids,b.group_ids)
            for entry in p.bands:
                self.assertEqual(entry.indices.tolist(),list(range(12)));self.assertEqual(entry.supported_count,12)
                q=entry.target.band.cells();condition=m.srf(entry.target.band.wavelengths_nm,entry.target.band.response)
                actual,support=QueryFrameDecoder(d,p.pair_query_frame)(b,q,condition,kind='B2');expected,other=d(b,p.pair_query_frame.boxes(q),condition,kind='B2')
                torch.testing.assert_close(actual,expected,atol=0,rtol=0);torch.testing.assert_close(support,other);self.assertTrue(bool(support.all()))
            self.assertEqual(source.bands[0].affine[2],20. if reverse else 0.);self.assertEqual(target.bands[0].affine[2],0. if reverse else 20.)

    def test_value_approval_data_versions_context_and_legacy_path_fail_closed(self):
        source,target,admission=fixture(self.temp.name);bad=copy.deepcopy(admission);bad['pair_geometry'].pop('offset_review_file')
        with self.assertRaisesRegex(ValueError,'value review required'):reviewed_pair_task(source,target,bad)
        bad=copy.deepcopy(admission);bad['pair_geometry']['translation_m'][0]+=1
        with self.assertRaisesRegex(ValueError,'does not bind'):reviewed_pair_task(source,target,bad)
        changed=Observation(source.bands,dict(source.metadata,geometry_data_version='changed'))
        with self.assertRaisesRegex(ValueError,'data version'):reviewed_pair_task(changed,target,admission)
        with self.assertRaisesRegex(ValueError,'use reviewed_pair_task'):plan(cross_source_task(source,target,admission))
        task=reviewed_pair_task(source,target,admission);p=plan(task);task.metadata['pair_query_frame']['query_offset_xy']=(0.,0.)
        with self.assertRaisesRegex(ValueError,'context missing or changed'):p.validate_retained(task)
        same=reviewed_pair_task(source,target,admission);invalid=replace(same,kind='B1')
        with self.assertRaisesRegex(ValueError,'only to B2'):frame_for_task(invalid)

    def test_stream_gradients_match_explicit_native_query_reference(self):
        source,target,admission=fixture(self.temp.name);task=reviewed_pair_task(source,target,admission);p=plan(task);m=encoder();d=ReflectanceDecoder(indexed=True,cell_indexed=True)
        result=stream_adaptive_task(m,d,task,p,train_stds={'0':1.,'1':1.},chunk_pixels=7,loss_weight=.7,backward=True);actual=flat_grads([m,d]).clone();m.zero_grad(set_to_none=True);d.zero_grad(set_to_none=True)
        z=m(source.bands,origin_xy=origin(source));loss=[]
        for entry in task.targets:
            condition=m.srf(entry.band.wavelengths_nm,entry.band.response);prediction,valid=d(z,p.pair_query_frame.boxes(entry.band.cells()),condition,kind='B2');truth=entry.band.reflectance.flatten();loss.append((prediction[valid]-truth[valid]).square().mean())
        objective=.7*sum(loss)/len(loss);objective.backward();expected=flat_grads([m,d]);self.assertAlmostEqual(result['weighted_loss'],float(objective.detach()),places=6);self.assertLess(float((actual-expected).norm()/expected.norm()),2e-6)

    def test_teacher_queries_only_translation_and_gradient_isolation(self):
        source,target,admission=fixture(self.temp.name);m=encoder();teacher=make_teacher(m);predictor=JEPAPredictor();decoder=ReflectanceDecoder();task=reviewed_pair_task(source,target,admission);frame=frame_for_task(task)
        sr=m(source.bands,origin_xy=origin(source))
        with torch.no_grad():tr=teacher(target.bands,origin_xy=origin(target))
        center=tr.centers.clone();query=teacher_support_queries(sr,tr,query_offset_xy=frame.query_offset_xy)
        manual=replace(tr,cells=[frame.boxes(c) for c in tr.cells],centers=tr.centers+tr.centers.new_tensor(frame.query_offset_xy),bounds=frame.boxes(tr.bounds))
        expected=teacher_support_queries(sr,manual);torch.testing.assert_close(query.relative_xy,expected.relative_xy,atol=0,rtol=0);torch.testing.assert_close(query.overlap_m2,expected.overlap_m2,atol=0,rtol=0);torch.testing.assert_close(tr.centers,center,atol=0,rtol=0)
        loss,result=cross_source_latent_loss(m,teacher,predictor,source,target,admission=admission,source_encoder_kwargs=dict(origin_xy=origin(source)),target_encoder_kwargs=dict(origin_xy=origin(target)),loss_fn=lambda a,b,c:(a-b).square().mean(),logical_update=2000,curriculum_end_update=2000,ab_stable=True,weight=.05)
        self.assertIsNotNone(loss);loss.backward();self.assertGreater(result['supported_queries'],0);self.assertEqual(result['query_offset_xy'],(-20.,-0.));self.assertTrue(all(p.grad is None for p in teacher.parameters()));self.assertTrue(all(p.grad is None for p in decoder.parameters()));self.assertTrue(any(p.grad is not None for p in m.parameters()));self.assertTrue(any(p.grad is not None for p in predictor.parameters()))

    def test_joint_B2_JEPA_microbatches_preserve_losses_exposure_and_gradients(self):
        views,pairs=joint_fixture(self.temp.name);m=encoder();d=ReflectanceDecoder(indexed=True,cell_indexed=True);predictor=JEPAPredictor();teacher=make_teacher(m)
        options=dict(logical_update=2000,chunk_pixels=7,clip_norm=None,jepa_views=pairs,teacher=teacher,predictor=predictor,jepa_weight=.05,latent_loss_fn=lambda a,b,c:(a-b).square().mean(),curriculum_end_update=2000,ab_stable=True)
        reference=accumulate_adaptive_joint(m,d,views,**options);gradient=flat_grads([m,d,predictor]).clone()
        for size in (2,4,8):
            actual=accumulate_adaptive_joint(m,d,views,**options,microbatch_observations=size);self.assertAlmostEqual(actual['weighted_loss'],reference['weighted_loss'],places=6);self.assertLess(float((flat_grads([m,d,predictor])-gradient).norm()/gradient.norm()),2e-6)
            self.assertEqual([r['exposure'] for r in actual['AB']],[r['exposure'] for r in reference['AB']]);self.assertEqual(actual['teacher_target_ids'],reference['teacher_target_ids'])

    def test_noninteger_grids_fractional_queries_QA_holes_and_no_support(self):
        source,target,admission=fixture(self.temp.name);record=admission['pair_geometry']
        source.bands=[replace(b,reflectance=torch.rand(16,16),valid_mask=torch.ones(16,16,dtype=torch.bool),affine=(.5,0.,0.,0.,-.5,8.)) for b in source.bands]
        for b in source.bands:b.valid_mask[5:7,5:7]=False
        target.bands=[replace(b,reflectance=torch.rand(7,7),valid_mask=torch.ones(7,7,dtype=torch.bool),affine=(1.21,0.,.37,0.,-1.21,6.79)) for b in target.bands]
        for b in target.bands:b.valid_mask[2,3]=False
        source.metadata['geometry_data_version']='synthetic-fractional-source';target.metadata['geometry_data_version']='synthetic-fractional-target'
        def approve(delta):
            record['translation_m']=list(delta);record['versions']={source.identifier:observation_version(source),target.identifier:observation_version(target)}
            payload={k:v for k,v in record.items() if k not in ('offset_review_file','offset_review_sha256')};path=Path(record['offset_review_file']);path.write_text(json.dumps(dict(status='USER_CONFIRMED_PAIR_OFFSET_VALUES',record_sha256=canonical_sha(payload))));record['offset_review_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        approve((.37,-1.21));task=reviewed_pair_task(source,target,admission);p=plan(task);m=encoder();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True)
        for entry in p.bands:self.assertEqual(entry.indices.tolist(),torch.where(entry.target.mask.flatten())[0].tolist())
        original=[b.affine for b in source.bands+target.bands];result=stream_adaptive_task(m,decoder,task,p,train_stds={'0':1.,'1':1.},chunk_pixels=7,loss_weight=1.,backward=True)
        self.assertEqual(result['status'],'OK');self.assertEqual([b.affine for b in source.bands+target.bands],original)
        approve((10000.,10000.));empty=reviewed_pair_task(source,target,admission);empty_plan=plan(empty);self.assertTrue(all(b.supported_count==0 for b in empty_plan.bands))
        absent=stream_adaptive_task(m,decoder,empty,empty_plan,train_stds={'0':1.,'1':1.},chunk_pixels=7,loss_weight=1.,backward=True);self.assertEqual(absent['status'],'NO_VALID_SUPPORTED_TARGET');self.assertIsNone(absent['loss'])

    def test_distinct_sensor_band_names_and_target_TRAIN_normalization(self):
        source,target,admission=fixture(self.temp.name);m=encoder();d=ReflectanceDecoder(indexed=True,cell_indexed=True);before=m(source.bands,origin_xy=origin(source)).tokens
        source.bands=[replace(b,record_id=name) for b,name in zip(source.bands,('B03','B04'),strict=True)];target.bands=[replace(b,record_id=name) for b,name in zip(target.bands,('SR_B3','SR_B4'),strict=True)]
        source.metadata['geometry_data_version']='synthetic-S2-named';target.metadata['geometry_data_version']='synthetic-Landsat-named';record=admission['pair_geometry'];record['versions']={source.identifier:observation_version(source),target.identifier:observation_version(target)}
        payload={k:v for k,v in record.items() if k not in ('offset_review_file','offset_review_sha256')};path=Path(record['offset_review_file']);path.write_text(json.dumps(dict(status='USER_CONFIRMED_PAIR_OFFSET_VALUES',record_sha256=canonical_sha(payload))));record['offset_review_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        torch.testing.assert_close(before,m(source.bands,origin_xy=origin(source)).tokens,atol=0,rtol=0)
        task=reviewed_pair_task(source,target,admission);p=plan(task);options=dict(chunk_pixels=7,loss_weight=1.,backward=False)
        plain=stream_adaptive_task(m,d,task,p,train_stds={'SR_B3':1.,'SR_B4':1.},**options);scaled=stream_adaptive_task(m,d,task,p,train_stds={'SR_B3':1.,'SR_B4':2.},**options)
        values={b['band']:b['nmse'] for b in plain['bands']};actual={b['band']:b['nmse'] for b in scaled['bands']};self.assertEqual(actual['SR_B3'],values['SR_B3']);self.assertEqual(actual['SR_B4'],values['SR_B4']/4)
        with self.assertRaises(KeyError):stream_adaptive_task(m,d,task,p,train_stds={'B03':1.,'B04':1.},**options)

    def test_synthetic_joint_update_EMA_and_exact_next_step_resume(self):
        from .test_participation_joint import ParticipationJointChecks
        directory=self.temp.name
        class Fixture(ParticipationJointChecks):
            def fixtures(self):return joint_fixture(directory)
        Fixture().test_full_joint_update_ema_and_next_step_restore()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_pair_B2_and_JEPA_backward_without_real_optimizer_updates(self):
        source,target,admission=fixture(self.temp.name);task=reviewed_pair_task(source,target,admission);p=plan(task)
        from .plan_prefetch import move_views
        from .joint import PlannedView
        view=move_views([PlannedView(task,p,{'0':1.,'1':1.},'fixture')],'cuda:0')[0];m=encoder().cuda();d=ReflectanceDecoder(indexed=True,cell_indexed=True).cuda()
        result=stream_adaptive_task(m,d,view.task,p,train_stds=view.train_stds,chunk_pixels=7,loss_weight=1.,backward=True);self.assertEqual(result['status'],'OK');self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for module in (m,d) for p in module.parameters()))
        teacher=make_teacher(m);predictor=JEPAPredictor().cuda();moved_source=to_device(source,'cuda:0');moved_target=to_device(target,'cuda:0');m.zero_grad(set_to_none=True)
        loss,_=cross_source_latent_loss(m,teacher,predictor,moved_source,moved_target,admission=admission,source_encoder_kwargs=dict(origin_xy=origin(source)),target_encoder_kwargs=dict(origin_xy=origin(target)),loss_fn=lambda a,b,c:(a-b).square().mean(),logical_update=2000,curriculum_end_update=2000,ab_stable=True,weight=.05);loss.backward();self.assertTrue(all(p.grad is None for p in teacher.parameters()))

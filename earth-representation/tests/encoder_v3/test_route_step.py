import copy,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
from encoder_v3.route_control import RouteProgress,frozen_identity,save_route_checkpoint,restore_route_checkpoint,inspect_route_checkpoint
from encoder_v3.route_step import RouteSchedule,execute_route_update,update_signature
from encoder_v3.training import make_teacher,JEPAPredictor
from encoder_v3.pilot_protocol import optimizer_groups
from tests.encoder_v3 import test_participation_joint as joint_fixtures
from encoder_v3.formal_test_fixtures import populate as populate_approvals


class RouteStepChecks(unittest.TestCase):
    def fixture(self,*,jepa=False,cursor=0):
        torch.set_num_threads(2);torch.manual_seed(119);base=joint_fixtures.ParticipationJointChecks();views,paired=base.fixtures();source_ids={'members':'SYNTHETIC_FIXTURE_ONLY'}
        for v in views:v.task.metadata['frozen_source_ids']=source_ids;v.plan=base.plan(v.task)
        for v in paired:v.source.metadata['frozen_source_ids']=source_ids;v.target.metadata['frozen_source_ids']=source_ids
        encoder=base.encoder();modules=dict(encoder=encoder,decoder=__import__('encoder_v3.model',fromlist=['ReflectanceDecoder']).ReflectanceDecoder(indexed=True,cell_indexed=True))
        if jepa:modules.update(predictor=JEPAPredictor(),teacher=make_teacher(encoder))
        real=json.loads(((Path(__file__).resolve().parents[2] / 'src/encoder_v3')/'config.json').read_text());config=dict(required_before_training={k:None for k in real['required_before_training']},required_before_jepa={k:None for k in real['required_before_jepa']},formal_step=dict(precision='fp32',chunk_pixels=7,microbatch_observations=1,clip_norm=3.,peak_lr=1e-4,plateau_threshold=0.,weight_decay=.04,betas=[.9,.999],eps=1e-8,ema_coefficient=.996))
        populate_approvals(config,jepa=True)
        optimizer=torch.optim.AdamW(optimizer_groups([m for k,m in modules.items() if k!='teacher'],.04),lr=1e-4,betas=(.9,.999),eps=1e-8);scheduler=RouteSchedule(optimizer,peak_lr=1e-4,threshold=0.)
        frozen=dict(formal_config_sha256=frozen_identity(config),source_identities=source_ids,merge_recipe=encoder.merge_recipe,fixture='NO_REAL_DATA_OR_OPERATIONAL_AUTHORIZATION')
        progress=RouteProgress('shared',cursor,frozen_identity(frozen))
        if cursor:
            state=scheduler.state_dict();state.update(last_update=cursor-1,last_validation=(cursor//250)*250);state['warmup']['last_update']=99;scheduler.load_state_dict(state)
        if jepa:progress=progress.fork('a'*64,ab_stable=True)['A+B+JEPA']
        latent=paired if jepa else ();signature=update_signature(views,latent)
        return dict(config=config,progress=progress,frozen_ids=frozen,modules=modules,optimizer=optimizer,scheduler=scheduler,views=views,expected_input_signature=signature,expected_AB_input_signature=update_signature(views),jepa_views=latent,jepa_weight=.05 if jepa else None,latent_loss_fn=(lambda p,t,b:(p-t).square().mean()) if jepa else None,ab_stable=jepa)

    def call(self,fixture,directory,**extra):
        return execute_route_update(**fixture,directory=directory,sampler_after=dict(next_update=fixture['progress'].logical_update+1,next_input_signature=fixture['expected_input_signature'],next_AB_input_signature=fixture['expected_AB_input_signature'],fixture_cursor=fixture['progress'].logical_update+1),**extra)

    def compare(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for k in a:self.compare(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.compare(x,y)
        else:self.assertEqual(a,b)

    def test_current_real_configuration_and_cap_stop_before_any_work(self):
        config=json.loads(((Path(__file__).resolve().parents[2] / 'src/encoder_v3')/'config.json').read_text())
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            path=Path(folder)/'never_created'
            with self.assertRaisesRegex(ValueError,'Training blocked'):
                execute_route_update(config=config,progress=None,frozen_ids={},modules=None,optimizer=None,scheduler=None,views=None,expected_input_signature=None,sampler_after={},directory=path)
            self.assertFalse(path.exists())
            f=self.fixture();f['progress']=replace(f['progress'],logical_update=10000)
            with patch.object(f['optimizer'],'step',side_effect=AssertionError('Over-cap optimizer operation')):
                with self.assertRaisesRegex(RuntimeError,'cap reached'):self.call(f,path)
            self.assertFalse(path.exists())
            f=self.fixture();f['expected_input_signature']='c'*64
            with self.assertRaisesRegex(ValueError,'members/masks/targets'):self.call(f,path)
            self.assertFalse(path.exists())

    def test_baseline_and_JEPA_next_complete_update_restore_exactly(self):
        for jepa,cursor in [(False,0),(True,2000)]:
            with self.subTest(jepa=jepa),tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
                self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));folder=Path(folder);f=self.fixture(jepa=jepa,cursor=cursor)
                old_teacher=copy.deepcopy(f['modules']['teacher'].state_dict()) if jepa else None
                first,metrics=self.call(f,folder/'continuous');checkpoint=folder/'continuous'/first.phase/f'u{first.logical_update:06d}.pt'
                self.assertEqual(first.logical_update,cursor+1);self.assertEqual(metrics['result']['effective_batch'],8)
                if jepa:
                    for key,p in f['modules']['teacher'].named_parameters():torch.testing.assert_close(p,torch.lerp(old_teacher[key],dict(f['modules']['encoder'].named_parameters())[key],1-.996),rtol=0,atol=0)
                    self.assertTrue(all(p.grad is None for p in f['modules']['teacher'].parameters()))
                f['progress']=first;expected_rng=torch.rand(3);second,expected=self.call(f,folder/'continuous');weights=copy.deepcopy({k:m.state_dict() for k,m in f['modules'].items()});moments=copy.deepcopy(f['optimizer'].state_dict());schedule=copy.deepcopy(f['scheduler'].state_dict())
                restored,progress=restore_route_checkpoint(checkpoint,modules=f['modules'],optimizer=f['optimizer'],scheduler=f['scheduler'],frozen_ids=f['frozen_ids'],phase=first.phase,prefix_checkpoint_sha256=first.prefix_checkpoint_sha256)
                torch.testing.assert_close(torch.rand(3),expected_rng,rtol=0,atol=0);self.assertEqual(restored['sampler']['fixture_cursor'],cursor+1);f['progress']=progress
                actual_progress,actual=self.call(f,folder/'resumed');self.assertEqual(actual_progress,second);self.assertEqual(actual['result']['weighted_loss'],expected['result']['weighted_loss']);self.assertEqual(actual['input_signature'],expected['input_signature'])
                self.compare({k:m.state_dict() for k,m in f['modules'].items()},weights);self.compare(f['optimizer'].state_dict(),moments);self.compare(f['scheduler'].state_dict(),schedule)

    def test_runner_state_and_optimizer_steps_are_committed_with_the_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            folder=Path(folder);f=self.fixture()
            state=dict(best_validation=1.25,stale_validations=2,validation_evaluations=3)
            progress,metrics=self.call(f,folder/'with_state',runner_state=state)
            checkpoint=folder/'with_state'/progress.phase/f'u{progress.logical_update:06d}.pt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            self.assertEqual(saved['sampler']['runner_state'],state)
            self.assertEqual(metrics['optimizer_steps'],1)
            self.assertEqual(metrics['optimizer_steps_before'],0)
            self.assertEqual(metrics['result']['optimizer_updates'],0)
            with self.assertRaisesRegex(ValueError,'Complete runner early-stop state'):
                self.call(self.fixture(),folder/'bad_state',runner_state=dict(best_validation=1.0))

    def test_default_gate_blocks_and_an_injected_gate_is_used(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            with self.assertRaisesRegex(ValueError,'Missing required_before_training'):
                execute_route_update(config={},progress=None,frozen_ids={},modules=None,optimizer=None,scheduler=None,views=None,expected_input_signature=None,sampler_after={},directory=Path(folder)/'never')
            def refuse(config):raise ValueError('custom E0 gate refused')
            with self.assertRaisesRegex(ValueError,'custom E0 gate refused'):
                execute_route_update(config={},progress=None,frozen_ids={},modules=None,optimizer=None,scheduler=None,views=None,expected_input_signature=None,sampler_after={},directory=Path(folder)/'never',gate=refuse)
            self.assertFalse((Path(folder)/'never').exists())

    def test_plateau_reduction_survives_warmup_exit_and_resume(self):
        model=torch.nn.Linear(1,1);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4);schedule=RouteSchedule(optimizer,peak_lr=3e-4,threshold=.01)
        for index in range(1250):
            schedule.before_update(index)
            if (index+1)%250==0:schedule.validate(1.,completed_update=index+1)
        self.assertEqual(optimizer.param_groups[0]['lr'],1.5e-4);state=copy.deepcopy(schedule.state_dict());schedule.before_update(1250);self.assertEqual(optimizer.param_groups[0]['lr'],1.5e-4)
        schedule.load_state_dict(state);schedule.before_update(1250);self.assertEqual(optimizer.param_groups[0]['lr'],1.5e-4)
        with self.assertRaises(ValueError):schedule.before_update(1250)
        with self.assertRaises(ValueError):RouteSchedule(optimizer,peak_lr=3e-4,threshold=.1).load_state_dict(state)

    def test_post_optimizer_failure_keeps_attempt_and_rejects_replay(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));folder=Path(folder);f=self.fixture();initial=folder/'initial.pt'
            save_route_checkpoint(initial,progress=f['progress'],modules=f['modules'],optimizer=f['optimizer'],scheduler=f['scheduler'],sampler=dict(next_update=0,next_input_signature=f['expected_input_signature']),frozen_ids=f['frozen_ids'])
            with patch('encoder_v3.route_step.save_route_checkpoint',side_effect=OSError('synthetic disk failure')):
                with self.assertRaises(OSError):self.call(f,folder/'run')
            attempt=folder/'run/shared/u000001.attempt.json';record=json.loads(attempt.read_text());self.assertEqual(record['status'],'FAILED_RECONCILE_REQUIRED');self.assertTrue(record['optimizer_may_have_executed']);self.assertFalse((folder/'run/shared/u000001.pt').exists());before=attempt.read_bytes()
            _,f['progress']=restore_route_checkpoint(initial,modules=f['modules'],optimizer=f['optimizer'],scheduler=f['scheduler'],frozen_ids=f['frozen_ids'],phase='shared',prefix_checkpoint_sha256=None)
            with patch.object(f['optimizer'],'step',side_effect=AssertionError('No blind replay')):
                with self.assertRaises(FileExistsError):self.call(f,folder/'run')
            self.assertEqual(attempt.read_bytes(),before);self.assertFalse((folder/'run/shared/.update.lock').exists())

    def test_post_checkpoint_logging_failure_retains_state_and_metrics(self):
        from encoder_v3 import route_step
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')));folder=Path(folder);f=self.fixture();original=route_step._json
            def fail_metrics(path,payload,**kwargs):
                if path.name.endswith('.metrics.json'):raise OSError('synthetic logging failure')
                return original(path,payload,**kwargs)
            with patch('encoder_v3.route_step._json',side_effect=fail_metrics):
                with self.assertRaises(OSError):self.call(f,folder)
            state,progress=inspect_route_checkpoint(folder/'shared/u000001.pt',frozen_ids=f['frozen_ids'],phase='shared',prefix_checkpoint_sha256=None);self.assertEqual(progress.logical_update,1);self.assertIn('result',state['sampler']['_route_last_metrics'])
            record=json.loads((folder/'shared/u000001.attempt.json').read_text());self.assertEqual(record['status'],'FAILED_RECONCILE_REQUIRED');self.assertIsNotNone(record['checkpoint_sha256'])


if __name__=='__main__':unittest.main()

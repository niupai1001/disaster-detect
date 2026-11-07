import copy,hashlib,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import test_route_step as fixtures
from .route_branch import initialize_route_branch
from .route_control import save_route_checkpoint,inspect_route_checkpoint,checkpoint_sha
from .training import JEPAPredictor


class RouteBranchChecks(unittest.TestCase):
    def prefix(self,folder):
        f=fixtures.RouteStepChecks().fixture(cursor=2000);optimizer=f['optimizer']
        for group in optimizer.param_groups:
            group['lr']=5e-5
            for p in group['params']:optimizer.state[p]=dict(step=torch.tensor(2000.),exp_avg=torch.full_like(p,.1),exp_avg_sq=torch.full_like(p,.2))
        f['scheduler'].plateau.best=.25;f['scheduler'].plateau.wait=3
        f['modules']['encoder'].zero_grad(set_to_none=True);f['modules']['decoder'].zero_grad(set_to_none=True)
        path=Path(folder)/'prefix.pt';sampler=dict(next_update=2000,next_input_signature=f['expected_input_signature'],members=['synthetic'+str(i) for i in range(8)],masks='fixture',targets='fixture')
        save_route_checkpoint(path,progress=f['progress'],modules=f['modules'],optimizer=optimizer,scheduler=f['scheduler'],sampler=sampler,frozen_ids=f['frozen_ids'])
        return f,path,sampler

    def branch(self,f,path,phase,**kwargs):
        return initialize_route_branch(config=f['config'],prefix_checkpoint=path,frozen_ids=f['frozen_ids'],phase=phase,encoder_factory=lambda:copy.deepcopy(f['modules']['encoder']),decoder_factory=lambda:copy.deepcopy(f['modules']['decoder']),ab_stable=True,**kwargs)

    def compare(self,a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for k in a:self.compare(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.compare(x,y)
        else:self.assertEqual(a,b)

    def test_two_routes_preserve_full_prefix_and_training_rng(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));f,path,sampler=self.prefix(folder);original=path.read_bytes();a=self.branch(f,path,'A+B');a_random=(random.random(),np.random.rand(),torch.rand(4))
            def predictor():
                random.random();np.random.rand(7);torch.rand(9)
                return JEPAPredictor()
            b=self.branch(f,path,'A+B+JEPA',predictor_factory=predictor,next_AB_input_signature=sampler['next_input_signature'],next_input_signature='e'*64);b_random=(random.random(),np.random.rand(),torch.rand(4));self.compare(a_random,b_random)
            self.assertEqual(path.read_bytes(),original);self.assertEqual(a['progress'].logical_update,2000);self.assertEqual(b['progress'].logical_update,2000);self.assertEqual(a['prefix_checkpoint_sha256'],b['prefix_checkpoint_sha256'])
            for name in ['encoder','decoder']:
                self.compare(a['modules'][name].state_dict(),b['modules'][name].state_dict());self.compare(a['modules'][name].state_dict(),f['modules'][name].state_dict())
            self.compare(a['optimizer'].state_dict(),f['optimizer'].state_dict());self.compare(a['scheduler'].state_dict(),b['scheduler'].state_dict())
            self.compare(a['optimizer'].state_dict()['state'],b['optimizer'].state_dict()['state']);self.assertTrue(all(g['lr']==5e-5 for g in b['optimizer'].param_groups));b['scheduler'].before_update(2000);self.assertTrue(all(g['lr']==5e-5 for g in b['optimizer'].param_groups))
            self.assertEqual(len(b['optimizer'].state),len(a['optimizer'].state));teacher=b['modules']['teacher'];self.compare(teacher.state_dict(),b['modules']['encoder'].state_dict());self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            ids={id(p) for p in teacher.parameters()};self.assertFalse(ids & {id(p) for g in b['optimizer'].param_groups for p in g['params']});self.assertFalse(ids & {id(p) for p in b['modules']['encoder'].parameters()})
            self.assertEqual(a['sampler']['members'],b['sampler']['members']);self.assertEqual(b['sampler']['next_AB_input_signature'],a['sampler']['next_input_signature']);self.assertEqual(b['sampler']['next_input_signature'],'e'*64)

    def test_branch_snapshot_is_complete_and_no_prefix_file_is_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));f,path,sampler=self.prefix(folder);b=self.branch(f,path,'A+B+JEPA',predictor_factory=JEPAPredictor,next_AB_input_signature=sampler['next_input_signature'],next_input_signature='f'*64);target=Path(folder)/'branch.pt'
            save_route_checkpoint(target,progress=b['progress'],modules=b['modules'],optimizer=b['optimizer'],scheduler=b['scheduler'],sampler=b['sampler'],frozen_ids=f['frozen_ids'])
            state,progress=inspect_route_checkpoint(target,frozen_ids=f['frozen_ids'],phase='A+B+JEPA',prefix_checkpoint_sha256=checkpoint_sha(path));self.assertEqual(progress.logical_update,2000);self.assertEqual(set(state['modules']),{'encoder','decoder','predictor','teacher'});self.assertEqual(state['sampler']['next_update'],2000)

    def test_bad_AB_handoff_and_unreviewed_Predictor_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent));f,path,sampler=self.prefix(folder)
            with self.assertRaisesRegex(ValueError,'next A/B batch'):self.branch(f,path,'A+B+JEPA',predictor_factory=JEPAPredictor,next_AB_input_signature='b'*64,next_input_signature='c'*64)
            a=self.branch(f,path,'A+B');expected=(random.random(),np.random.rand(),torch.rand(3))
            def wrong():
                random.random();np.random.rand(3);torch.rand(3)
                return torch.nn.Linear(2,2)
            with self.assertRaisesRegex(ValueError,'Predictor architecture'):self.branch(f,path,'A+B+JEPA',predictor_factory=wrong,next_AB_input_signature=sampler['next_input_signature'],next_input_signature='c'*64)
            self.compare((random.random(),np.random.rand(),torch.rand(3)),expected)
            self.assertEqual(set(Path(folder).iterdir()),{path})

    def test_operational_gate_blocks_initialization_before_factories(self):
        config=json.loads((Path(__file__).parent/'config.json').read_text())
        def forbidden():raise AssertionError('Operational configuration must not initialize models')
        with self.assertRaisesRegex(ValueError,'Training blocked'):
            initialize_route_branch(config=config,prefix_checkpoint='not_a_real_checkpoint',frozen_ids={},phase='A+B',encoder_factory=forbidden,decoder_factory=forbidden,ab_stable=False)

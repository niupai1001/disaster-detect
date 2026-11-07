import copy,hashlib,random,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .route_control import RouteProgress,committed_update_counts,frozen_identity,checkpoint_sha,save_route_checkpoint,inspect_route_checkpoint,fork_route_checkpoint,restore_route_checkpoint
from .training import CheckedPlateau,make_teacher


class RouteControlChecks(unittest.TestCase):
    frozen={'data':'synthetic_fixture','config':'not_a_formal_training_configuration','optimizer':'AdamW'}

    def progress(self,update):return RouteProgress('shared',update,frozen_identity(self.frozen))

    def objects(self,phase='shared'):
        encoder=torch.nn.Linear(2,2);decoder=torch.nn.Linear(2,1);modules=dict(encoder=encoder,decoder=decoder)
        if phase=='A+B+JEPA':modules.update(teacher=make_teacher(encoder),predictor=torch.nn.Linear(2,2))
        parameters=[p for k,m in modules.items() if k!='teacher' for p in m.parameters()]
        optimizer=torch.optim.AdamW(parameters,lr=3e-4)
        # Populate a state fixture without executing any optimizer update.
        for p in parameters:optimizer.state[p]=dict(step=torch.tensor(1.),exp_avg=torch.ones_like(p)*.1,exp_avg_sq=torch.ones_like(p)*.2)
        return modules,optimizer,CheckedPlateau(optimizer,threshold=0.)

    def save(self,path,progress,objects):
        modules,opt,scheduler=objects
        return save_route_checkpoint(path,progress=progress,modules=modules,optimizer=opt,scheduler=scheduler,sampler=dict(next_update=progress.logical_update,next_input_signature='b'*64,members=['fixture'],mask='fixture',targets='fixture'),frozen_ids=self.frozen)

    def test_curriculum_stability_and_exact_two_routes(self):
        for cursor,stable in [(1999,True),(2000,False),(10000,True)]:
            with self.assertRaises(ValueError):self.progress(cursor).fork('a'*64,ab_stable=stable)
        routes=self.progress(2000).fork('a'*64,ab_stable=True)
        self.assertEqual(set(routes),{'A+B','A+B+JEPA'})
        for p in routes.values():
            self.assertEqual(p.logical_update,2000)
            with self.assertRaises(ValueError):p.fork('c'*64,ab_stable=True)
        for value in [True,-1,10001,1.2]:
            with self.assertRaises(ValueError):self.progress(value)
        with self.assertRaises(ValueError):RouteProgress('extra',0,'a'*64)

    def test_shared_compute_counts_once_and_cap_cannot_extend(self):
        routes=self.progress(2000).fork('a'*64,ab_stable=True)
        routes={k:replace(v,logical_update=10000) for k,v in routes.items()}
        self.assertEqual(committed_update_counts(routes)['unique_committed_updates'],18000)
        self.assertEqual(committed_update_counts(routes)['logical_by_route'],{'A+B':10000,'A+B+JEPA':10000})
        for progress in routes.values():
            with self.assertRaises(RuntimeError):progress.next()
        tail=self.progress(9999).fork('a'*64,ab_stable=True);tail={k:v.next() for k,v in tail.items()}
        self.assertEqual(committed_update_counts(tail)['unique_committed_updates'],10001)
        for field,value in [('prefix_checkpoint_sha256','c'*64),('run_identity','d'*64),('prefix_update',2001)]:
            wrong=dict(routes);wrong['A+B+JEPA']=replace(wrong['A+B+JEPA'],**{field:value})
            with self.assertRaises(ValueError):committed_update_counts(wrong)
        with self.assertRaises(ValueError):committed_update_counts({'A+B':routes['A+B']})

    def test_full_prefix_fork_is_read_only_and_cursors_not_reset(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            path=Path(folder)/'prefix.pt';objects=self.objects();digest=self.save(path,self.progress(2000),objects);before=path.read_bytes();rng=torch.get_rng_state().clone()
            routes=fork_route_checkpoint(path,frozen_ids=self.frozen,ab_stable=True)
            self.assertEqual(before,path.read_bytes());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(all(p.prefix_checkpoint_sha256==digest and p.logical_update==2000 for p in routes.values()))
            self.assertEqual(set(Path(folder).iterdir()),{path})
            with self.assertRaises(FileExistsError):self.save(path,self.progress(2000),objects)
            with self.assertRaises(ValueError):fork_route_checkpoint(path,frozen_ids={'wrong':'fixture'},ab_stable=True)

    def test_restore_complete_branch_states_and_rng_without_CUDA(self):
        for phase in ['A+B','A+B+JEPA']:
            with self.subTest(phase=phase),tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
                self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            path=Path(folder)/'branch.pt';progress=self.progress(2000).fork('a'*64,ab_stable=True)[phase].next();objects=self.objects(phase);modules,opt,scheduler=objects
                random.seed(5);np.random.seed(5);torch.manual_seed(5)
                with patch('torch.cuda.is_available',side_effect=AssertionError('Unused CUDA queried')):
                    self.save(path,progress,objects)
                    expected=(random.random(),np.random.rand(),torch.rand(3));weights=copy.deepcopy({k:v.state_dict() for k,v in modules.items()});moments=copy.deepcopy(opt.state_dict());schedule=copy.deepcopy(scheduler.state_dict())
                    for m in modules.values():
                        for p in m.parameters():p.data.zero_()
                    for state in opt.state.values():state['exp_avg'].zero_()
                    opt.param_groups[0]['lr']=1e-5;scheduler.best=999.
                    state,actual=restore_route_checkpoint(path,modules=modules,optimizer=opt,scheduler=scheduler,frozen_ids=self.frozen,phase=phase,prefix_checkpoint_sha256='a'*64)
                self.assertEqual(actual,progress);self.assertEqual(state['sampler']['members'],['fixture']);self.assertEqual(state['sampler']['next_update'],2001);self.assertEqual(state['sampler']['next_input_signature'],'b'*64)
                self.assertEqual(random.random(),expected[0]);self.assertEqual(np.random.rand(),expected[1]);torch.testing.assert_close(torch.rand(3),expected[2],rtol=0,atol=0)
                for name,m in modules.items():
                    for k,v in m.state_dict().items():torch.testing.assert_close(v,weights[name][k],rtol=0,atol=0)
                self.assertEqual(scheduler.state_dict(),schedule);self.assertEqual(opt.state_dict()['param_groups'],moments['param_groups'])
                for i,s in opt.state_dict()['state'].items():
                    for k,v in s.items():torch.testing.assert_close(v,moments['state'][i][k],rtol=0,atol=0)

    def test_reject_wrong_phase_parent_cursor_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            path=Path(folder)/'state.pt';progress=self.progress(2000).fork('a'*64,ab_stable=True)['A+B'];self.save(path,progress,self.objects('A+B'));original=torch.load(path,weights_only=False)
            for phase,parent in [('A+B+JEPA','a'*64),('A+B','c'*64)]:
                with self.assertRaises(ValueError):inspect_route_checkpoint(path,frozen_ids=self.frozen,phase=phase,prefix_checkpoint_sha256=parent)
            for key in ['modules','optimizer','scheduler','sampler','rng']:
                broken=copy.deepcopy(original);del broken[key];torch.save(broken,path)
                with self.assertRaises(ValueError):inspect_route_checkpoint(path,frozen_ids=self.frozen,phase='A+B',prefix_checkpoint_sha256='a'*64)
            broken=copy.deepcopy(original);broken['logical_update']=0;torch.save(broken,path)
            with self.assertRaises(ValueError):inspect_route_checkpoint(path,frozen_ids=self.frozen,phase='A+B',prefix_checkpoint_sha256='a'*64)

    def test_optimizer_parameter_reordering_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            path=Path(folder)/'state.pt';objects=self.objects();self.save(path,self.progress(2000),objects);modules=objects[0]
            params=[p for m in modules.values() for p in m.parameters()];optimizer=torch.optim.AdamW(list(reversed(params)),lr=1e-5);scheduler=CheckedPlateau(optimizer,threshold=0.)
            for p in params:p.data.zero_()
            with self.assertRaisesRegex(ValueError,'parameter order differs'):
                restore_route_checkpoint(path,modules=modules,optimizer=optimizer,scheduler=scheduler,frozen_ids=self.frozen,phase='shared',prefix_checkpoint_sha256=None)
            self.assertTrue(all(torch.count_nonzero(p)==0 for p in params));self.assertEqual(optimizer.param_groups[0]['lr'],1e-5)

    def test_interrupted_write_and_lock_are_not_replayed_or_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            path=Path(folder)/'state.pt';partial=path.with_suffix('.pt.tmp');lock=path.with_suffix('.pt.route.lock');objects=self.objects()
            def interrupted(*args,**kwargs):
                partial.write_bytes(b'uncommitted fixture');raise OSError('simulated interruption')
            with patch('encoder_v3.training.save_checkpoint',side_effect=interrupted):
                with self.assertRaises(OSError):self.save(path,self.progress(2000),objects)
            self.assertFalse(path.exists());self.assertFalse(lock.exists());self.assertEqual(partial.read_bytes(),b'uncommitted fixture')
            with self.assertRaises(FileExistsError):self.save(path,self.progress(2000),objects)
            self.assertEqual(partial.read_bytes(),b'uncommitted fixture');self.assertFalse(lock.exists())
            partial.unlink();lock.write_bytes(b'fixture owner')
            with self.assertRaises(FileExistsError):self.save(path,self.progress(2000),objects)
            self.assertEqual(lock.read_bytes(),b'fixture owner');self.assertFalse(path.exists())

    def test_optimizer_and_teacher_isolation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            path=Path(folder)/'state.pt';progress=self.progress(2000).fork('a'*64,ab_stable=True)['A+B+JEPA'];objects=self.objects(progress.phase);modules,opt,scheduler=objects
            opt.add_param_group({'params':list(modules['teacher'].parameters())})
            with self.assertRaises(ValueError):self.save(path,progress,objects)
            objects=self.objects(progress.phase);objects[0]['teacher'].train()
            with self.assertRaises(ValueError):self.save(path,progress,objects)
            objects=self.objects();objects[0]['decoder']=objects[0]['encoder']
            with self.assertRaises(ValueError):self.save(path,self.progress(2000),objects)
            objects=self.objects(progress.phase);modules=objects[0];modules['teacher']=modules['encoder'];modules['teacher'].eval().requires_grad_(False)
            params=[p for k,m in modules.items() if k!='teacher' for p in m.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=3e-4)
            with self.assertRaisesRegex(ValueError,'independent'):
                self.save(path,progress,(modules,optimizer,CheckedPlateau(optimizer,threshold=0.)))
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()

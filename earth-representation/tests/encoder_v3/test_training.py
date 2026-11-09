import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from encoder_v3.training import CAdamW, CheckedPlateau, save_checkpoint, restore_checkpoint, JEPAPredictor, make_teacher, update_teacher
from encoder_v3.model import Encoder
from tests.encoder_v3.test_model import band


class TrainingChecks(unittest.TestCase):
    def test_cpu_checkpoint_never_initializes_unused_cuda_devices(self):
        model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        scheduler=CheckedPlateau(optimizer,threshold=0.)
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            path=Path(folder)/'cpu.pt'
            with patch('torch.cuda.is_available',side_effect=AssertionError('Unused GPU queried')),patch('torch.cuda.get_rng_state',side_effect=AssertionError('Unused GPU initialized')),patch('torch.cuda.get_rng_state_all',side_effect=AssertionError('All GPUs initialized')):
                save_checkpoint(path,modules={'model':model},optimizer=optimizer,scheduler=scheduler,logical_update=0,sampler={'next':0},frozen_ids={'fixture':True})
                state=restore_checkpoint(path,modules={'model':model},optimizer=optimizer,scheduler=scheduler,frozen_ids={'fixture':True})
            self.assertIsNone(state['rng']['cuda'])

    def test_batching_equivalence(self):
        torch.set_num_threads(2); a=Encoder(hyper_hidden=64).eval(); b=Encoder(hyper_hidden=64,local_batch_size=8).eval()
        b.load_state_dict(a.state_dict()); x=band()
        za=a([x],patch_m=20,origin_xy=(0,0)); zb=b([x],patch_m=20,origin_xy=(0,0))
        torch.testing.assert_close(za.tokens,zb.tokens,atol=2e-6,rtol=2e-5)
        za.tokens.square().mean().backward(); zb.tokens.square().mean().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)

    def test_restore_and_plateau_minimum(self):
        model=torch.nn.Linear(2,1); opt=CAdamW(model.parameters(),lr=3e-4)
        scheduler=CheckedPlateau(opt,threshold=0.,patience=1,minimum=1e-5)
        model(torch.ones(1,2)).sum().backward(); opt.step(); opt.zero_grad()
        scheduler.step(1.)
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')))
            path=folder+'/state.pt'
            save_checkpoint(path,modules={'model':model},optimizer=opt,scheduler=scheduler,logical_update=1,sampler={'next':2},frozen_ids={'data':'test'})
            expected=torch.rand(3); original={k:v.clone() for k,v in model.state_dict().items()}
            scheduler.step(1.); model.weight.data.zero_()
            state=restore_checkpoint(path,modules={'model':model},optimizer=opt,scheduler=scheduler,frozen_ids={'data':'test'})
            torch.testing.assert_close(torch.rand(3),expected)
            self.assertEqual(state['sampler']['next'],2)
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,original[k])
        for _ in range(10): scheduler.step(1.)
        self.assertEqual(opt.param_groups[0]['lr'],1e-5)

    def test_jepa_stop_gradient(self):
        student=torch.nn.Linear(128,128); teacher=make_teacher(student); predictor=JEPAPredictor()
        z=student(torch.randn(3,128))
        pred=predictor(z,relative_xy=torch.zeros(2,2),support_xy=torch.ones(2,2),srf_condition=torch.zeros(2,32),gsd_xy=torch.ones(2,2))
        target=teacher(torch.randn(2,128))
        (pred-target).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is not None for p in student.parameters()))
        update_teacher(teacher,student,.996)

    def test_next_update_after_restore(self):
        # Compare the next complete update, including moments, RNG, LR and EMA.
        for optimizer_type in (torch.optim.AdamW,CAdamW):
            with self.subTest(optimizer=optimizer_type.__name__):
                torch.manual_seed(19)
                model=torch.nn.Linear(2,1); teacher=make_teacher(model)
                opt=optimizer_type(model.parameters(),lr=3e-4,weight_decay=.04)
                scheduler=CheckedPlateau(opt,threshold=0.,patience=1)
                def step():
                    x=torch.randn(4,2);target=torch.randn(4,1)
                    opt.zero_grad(set_to_none=True)
                    loss=(model(x)-target).square().mean();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
                    opt.step();update_teacher(teacher,model,.996)
                    scheduler.step(float(loss.detach()))
                    return x,target,loss.detach()
                step()
                with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
                    path=Path(folder)/'resume.pt'
                    frozen={'data':'fixture','mask':'fixture','code':'fixture'}
                    save_checkpoint(path,modules={'model':model,'teacher':teacher},optimizer=opt,scheduler=scheduler,logical_update=1,sampler={'next_update':1,'next_members':['fixture']},frozen_ids=frozen)
                    expected=step()
                    parameters={k:v.clone() for k,v in model.state_dict().items()}
                    ema={k:v.clone() for k,v in teacher.state_dict().items()}
                    moments={p:{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in state.items()} for p,state in opt.state.items()}
                    schedule=scheduler.state_dict().copy();lr=opt.param_groups[0]['lr']
                    restored=restore_checkpoint(path,modules={'model':model,'teacher':teacher},optimizer=opt,scheduler=scheduler,frozen_ids=frozen)
                    self.assertEqual(restored['logical_update'],1)
                    self.assertEqual(restored['sampler'],{'next_update':1,'next_members':['fixture']})
                    actual=step()
                    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
                    for k,v in model.state_dict().items():torch.testing.assert_close(v,parameters[k],rtol=0,atol=0)
                    for k,v in teacher.state_dict().items():torch.testing.assert_close(v,ema[k],rtol=0,atol=0)
                    for p,state in opt.state.items():
                        for k,v in state.items():
                            if isinstance(v,torch.Tensor):torch.testing.assert_close(v,moments[p][k],rtol=0,atol=0)
                            else:self.assertEqual(v,moments[p][k])
                    self.assertEqual(scheduler.state_dict(),schedule)
                    self.assertEqual(opt.param_groups[0]['lr'],lr)


if __name__=='__main__': unittest.main()

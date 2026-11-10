"""Actual training-loop mechanics on synthetic values only."""
import copy
import unittest
import torch
from encoder_v3.latent_response.train import require_training,select_rows,update,validate
from encoder_v3.latent_response.checkpoint import digest
from encoder_v3.latent_response.data import make_view
from encoder_v3.latent_response.execution import view_loss
from tests.encoder_v3.latent_response.tests.test_pipeline import fixture


class SyntheticDataset:
    def __init__(self,channels,observation):
        self.original=channels;self.observation=observation
        self.rows=[dict(id=observation.identifier,role='TRAIN',sensor='S2',bands=[{}, {}, {}])]
    def __getitem__(self,index):return self.observation
    def channels(self,observation,device):return copy.deepcopy(self.original)


class TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_authorization_is_bound_to_config_and_budget(self):
        config=dict(phase='E2',steps=20)
        approval=dict(source='user_in_current_chat',config_sha256=digest(config),phase='E2',steps=20)
        require_training(approval,config)
        for bad in (None,dict(approval,steps=21),dict(approval,config_sha256='changed')):
            with self.assertRaises(PermissionError):require_training(bad,config)

    def test_pool_cycles_without_pixel_cap_and_excludes_VAL(self):
        rows=[dict(id=str(i),role='TRAIN',sensor='S2',bands=[{},{}],area=999999999) for i in range(5)]
        rows+=[dict(id='heldout',role='VAL',sensor='S2',bands=[{},{}])]
        selected=[select_rows(rows,17,u)[0][1]['id'] for u in range(5)]
        self.assertEqual(len(set(selected)),5)
        self.assertNotIn('heldout',selected)
        self.assertEqual(select_rows(rows,17,2),select_rows(list(reversed(rows)),17,2))

    def test_task_weight_scales_all_parameter_gradients(self):
        model,cs,obs=fixture(16);view=make_view(obs,cs,'B1')
        view_loss(model,view,backward=True,chunk_pixels=256,loss_weight=1.)
        grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        view_loss(model,view,backward=True,chunk_pixels=256,loss_weight=.5)
        for n,p in model.named_parameters():
            if n in grads:torch.testing.assert_close(p.grad,grads[n]*.5,atol=1e-5,rtol=1e-4)

    def test_update_and_validation_commit_latest_state(self):
        model,cs,obs=fixture(24);dataset=SyntheticDataset(cs,obs)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
        config=dict(phase='E2',seed=17,tile_rows=16,chunk_rows=1,chunk_pixels=256,clip_norm=3.)
        state=dict(cursor=0,validation={},best=None)
        before=model.registry.get(cs[0].sensor_key).detach().clone()
        record=update(model,dataset,optimizer,scheduler,state,config,'cpu')
        self.assertEqual(state['cursor'],1);self.assertGreater(record['loss'],0)
        self.assertFalse(torch.equal(before,model.registry.get(cs[0].sensor_key)))
        result=validate(model,dataset,optimizer,scheduler,state,[dict(id=obs.identifier,sensor='S2')],config,'cpu')
        self.assertEqual(state['validation'],result);self.assertEqual(result['cursor'],1)
        a=make_view(obs,cs,'B1',seed=17,update=4);b=make_view(obs,cs,'B1',seed=17,update=4)
        self.assertEqual([c.channel_meta for c in a.channels],[c.channel_meta for c in b.channels])


    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_checkpoint_next_update(self):
        import tempfile
        from pathlib import Path
        from encoder_v3.latent_response.checkpoint import save_checkpoint,load_checkpoint,parameter_sha
        model,cs,obs=fixture(16);model.cuda()
        for c in cs:
            c.values=c.values.cuda();c.valid=c.valid.cuda();c.visible=c.visible.cuda()
        dataset=SyntheticDataset(cs,obs)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5)
        state=dict(cursor=0,validation={})
        config=dict(phase='E2',seed=17,tile_rows=16,chunk_rows=1,chunk_pixels=256,clip_norm=3.)
        identity=dict(code='synthetic',config='synthetic',registry=model.registry.identity_sha256(),catalog='synthetic',data='synthetic',sampling='synthetic')
        update(model,dataset,optimizer,scheduler,state,config,'cuda:0')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.pt';saved=parameter_sha(model)
            save_checkpoint(p,model,optimizer,state,identity,scheduler)
            update(model,dataset,optimizer,scheduler,state,config,'cuda:0')
            expected=copy.deepcopy(model.state_dict())
            load_checkpoint(p,model,optimizer,state,identity,scheduler)
            self.assertEqual(parameter_sha(model),saved);self.assertEqual(state['cursor'],1)
            update(model,dataset,optimizer,scheduler,state,config,'cuda:0')
            maximum=max(float((model.state_dict()[n]-v).abs().max()) for n,v in expected.items())
            print('CUDA resumed next-update max_abs',maximum)
            for n,v in expected.items():torch.testing.assert_close(model.state_dict()[n],v,atol=1e-5,rtol=1e-4)

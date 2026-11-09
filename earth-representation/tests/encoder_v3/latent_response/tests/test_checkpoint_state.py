"""Checkpoint scheduler, optimizer, gradients and RNG restoration are transactional."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from encoder_v3.latent_response.model import LatentResponseModel
from encoder_v3.latent_response import SensorKey,ChannelCatalog
from encoder_v3.latent_response.checkpoint import save_checkpoint,load_checkpoint,parameter_sha,validation_transaction


class CheckpointStateTest(unittest.TestCase):
    def test_complete_live_state_roundtrip(self):
        key=SensorKey('test','instrument');model=LatentResponseModel({key:ChannelCatalog.from_ordered(['a'])})
        opt=torch.optim.AdamW(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.5)
        model.registry.get(key).square().sum().backward();opt.step();scheduler.step()
        state=dict(cursor=5,validation=dict(completed=True,metric=.2))
        identity=dict(code='a',config='b',registry=model.registry.identity_sha256(),catalog='c',data='d',sampling='e')
        expected=parameter_sha(model);old_scheduler=copy.deepcopy(scheduler.state_dict())
        old_optimizer=copy.deepcopy(opt.state_dict());old_grad=model.registry.get(key).grad.clone()
        with validation_transaction(model,opt,state,scheduler):
            model.eval();model.zero_grad();state['cursor']=99
            random.random();np.random.rand();torch.rand(2)
            with torch.no_grad():model.registry.get(key).add_(1)
        torch.testing.assert_close(model.registry.get(key).grad,old_grad,rtol=0,atol=0)
        self.assertTrue(model.training);self.assertEqual(state['cursor'],5)
        self.assertEqual(parameter_sha(model),expected)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'c.pt';save_checkpoint(p,model,opt,state,identity,scheduler)
            rng_expected=(random.random(),np.random.rand(),torch.rand(3))
            state['cursor']=123;opt.param_groups[0]['lr']=.9
            load_checkpoint(p,model,opt,state,identity,scheduler)
            self.assertEqual((random.random(),np.random.rand()),rng_expected[:2])
            torch.testing.assert_close(torch.rand(3),rng_expected[2],rtol=0,atol=0)
            self.assertEqual(scheduler.state_dict(),old_scheduler)
            self.assertEqual(opt.param_groups[0]['lr'],old_optimizer['param_groups'][0]['lr'])
            for index,values in old_optimizer['state'].items():
                for name,value in values.items():torch.testing.assert_close(opt.state_dict()['state'][index][name],value,rtol=0,atol=0)
            self.assertEqual(state['cursor'],5);self.assertTrue(state['validation']['completed'])
            payload=torch.load(p,weights_only=False);payload['model']['registry.p.test|instrument']=torch.zeros(33);torch.save(payload,p)
            with self.assertRaises(RuntimeError):load_checkpoint(p,model,opt,state,identity,scheduler)
            self.assertEqual(parameter_sha(model),expected);self.assertEqual(state['cursor'],5)

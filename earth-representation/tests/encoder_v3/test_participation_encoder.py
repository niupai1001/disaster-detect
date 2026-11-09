import copy,unittest
from dataclasses import replace
import torch
from encoder_v3.participation_layout import AreaEquivalentHillRate
from encoder_v3.participation_encoder import ParticipationEncoder,prepare_participation_task
from encoder_v3.adaptive_streaming import stream_adaptive_task
from encoder_v3.model import ReflectanceDecoder
from tests.encoder_v3.test_adaptive_window import source
from encoder_v3.data import Observation
from encoder_v3.tasks import spectral_task,spatial_task


class ParticipationEncoderChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(91)
    def curve(self):return AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
    def model(self,seed=0):return ParticipationEncoder(rate_curve=self.curve(),selection_seed=seed,hyper_hidden=64,local_batch_size=32)

    def test_hidden_isolation_all_gradients_and_recipe_checkpoint(self):
        b=source(.5,1.21);b.visible=b.valid_mask.clone();b.visible[3:7,3:7]=False;model=self.model().eval();x=model([b],origin_xy=(99.8,180.))
        changed=replace(b,reflectance=b.reflectance.clone());changed.reflectance[~b.keep()]=float('nan');y=model([changed],origin_xy=(99.8,180.));torch.testing.assert_close(x.tokens,y.tokens,atol=0,rtol=0)
        self.assertEqual(sum(len(v) for v in x.visible_cells),int(b.keep().sum()));stats=x.participation_audit[0];self.assertEqual(stats['merged_participants']+stats['independent_positions'],int(b.keep().sum()))
        x.tokens.square().mean().backward();self.assertTrue(all(p.grad is not None for p in model.parameters()));self.assertFalse(hasattr(model,'exponent'));self.assertFalse(any('rate' in name for name,_ in model.named_parameters()))
        restored=self.model().eval();restored.load_state_dict(model.state_dict());torch.testing.assert_close(restored([b],origin_xy=(99.8,180.)).tokens,x.tokens,atol=0,rtol=0)
        with self.assertRaises(ValueError):self.model(seed=3).load_state_dict(model.state_dict())

    def test_full_target_reconstruction_and_seed_mismatch(self):
        bands=[replace(source(.5,.5),record_id=str(i)) for i in range(3)];obs=Observation(bands,dict(id='generated_B',role='TRAIN'))
        for task in (spectral_task(obs,seed=0,update=0),spatial_task(obs,(102.13,196.77,105.13,199.77))):
            model=self.model().eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True)
            plan=prepare_participation_task(task,rate_curve=self.curve(),selection_seed=0,origin_xy=(99.8,180.),chunk_pixels=41,cell_indexed=True)
            reference=None
            for chunk in (17,89):
                model.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
                result=stream_adaptive_task(model,decoder,task,plan,train_stds={str(i):.2 for i in range(3)},chunk_pixels=chunk,loss_weight=1.,backward=True)
                self.assertEqual(result['status'],'OK');self.assertTrue(all(p.grad is not None for p in model.parameters()))
                grads=[p.grad.clone() for p in list(model.parameters())+list(decoder.parameters())]
                if reference is None:reference=(result['loss'],grads)
                else:
                    self.assertAlmostEqual(result['loss'],reference[0],places=4)
                    for a,b in zip(grads,reference[1]):torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
            with self.assertRaisesRegex(ValueError,'Merge recipe'):
                stream_adaptive_task(self.model(seed=1),decoder,task,plan,train_stds={},chunk_pixels=41,loss_weight=1.,backward=True)

    def test_rate_configuration_cannot_add_learned_window_parameters(self):
        class LearnedRate(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()))
            def forward(self,gx,gy):return self.weight.sigmoid()
            def get_config(self):return {'invalid':'learned'}
        with self.assertRaisesRegex(ValueError,'must not be learned'):ParticipationEncoder(rate_curve=LearnedRate(),selection_seed=0,hyper_hidden=64)
        with self.assertRaisesRegex(ValueError,'physical rate curve'):ParticipationEncoder(rate_curve=None,selection_seed=0,hyper_hidden=64)

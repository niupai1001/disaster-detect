import unittest
import torch
from encoder_v3.model import Encoder,ReflectanceDecoder
from encoder_v3.data import Observation,to_device
from encoder_v3.tasks import spectral_task,spatial_task
from encoder_v3.streaming import prepare_task,stream_task
from encoder_v3.losses import band_loss
from tests.encoder_v3.test_model import band
from encoder_v3.cuda_probe import probe_cuda


class StreamingChecks(unittest.TestCase):
    @unittest.skipUnless(probe_cuda()['available'],'Isolated CUDA initialization failed; GPU case not verified')
    def test_cpu_plan_gpu_targets(self):
        device='cuda:0'
        observation=Observation([band(),band()],dict(id='gpu_plan',role='TRAIN'))
        for i,b in enumerate(observation.bands):b.record_id=str(i)
        task=spectral_task(observation,seed=0,update=0)
        plan=prepare_task(task,patch_m=20,origin_xy=(0,0),chunk_pixels=5)
        gpu_task=spectral_task(to_device(observation,device),seed=0,update=0)
        result=stream_task(Encoder(hyper_hidden=64,layout_mode='indexed').to(device),ReflectanceDecoder(indexed=True).to(device),gpu_task,plan,train_stds={'0':.1,'1':.2},chunk_pixels=5,loss_weight=1.,backward=True)
        self.assertEqual(result['status'],'OK')

    def test_stale_plan_rejected(self):
        task=spectral_task(Observation([band(),band()],dict(id='changed',role='TRAIN')),seed=0,update=0)
        plan=prepare_task(task,patch_m=20,origin_xy=(0,0),chunk_pixels=5)
        task.targets[0].mask[0,0]=False
        with self.assertRaisesRegex(ValueError,'changed after planning'):
            stream_task(Encoder(hyper_hidden=64),ReflectanceDecoder(indexed=True),task,plan,train_stds={},chunk_pixels=5,loss_weight=1.,backward=True)

    def test_full_loss_and_all_parameter_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        bands=[band() for _ in range(4)]
        for i,b in enumerate(bands):b.record_id=str(i)
        observation=Observation(bands,dict(id='test',role='TRAIN'))
        stds={str(i):.1*(i+1) for i in range(4)}
        for task in (spectral_task(observation,seed=3,update=9),spatial_task(observation,(0,0,20,60))):
            reference=Encoder(hyper_hidden=64,local_batch_size=8,layout_mode='indexed').eval();head=ReflectanceDecoder(indexed=True)
            representation=reference(task.visible,patch_m=20,origin_xy=(0,0));losses=[]
            for target in task.targets:
                condition=reference.srf(target.band.wavelengths_nm,target.band.response)
                ids=torch.where(target.mask.flatten())[0];truth=target.band.reflectance.flatten()[ids]
                pred,supported=head(representation,target.band.cells()[ids],condition,kind=task.kind,mask_bounds=None if task.rectangle is None else representation.bounds.new_tensor(task.rectangle))
                loss,_=band_loss(pred,truth,torch.ones_like(supported),supported,train_std=stds[target.band.record_id])
                if loss is not None:losses.append(loss)
            expected=torch.stack(losses).mean()*1.7;expected.backward()
            for chunk in (1,5,19,100):
                candidate=Encoder(hyper_hidden=64,local_batch_size=8,layout_mode='indexed').eval();decoder=ReflectanceDecoder(indexed=True)
                candidate.load_state_dict(reference.state_dict());decoder.load_state_dict(head.state_dict())
                baseline_plan=prepare_task(task,patch_m=20,origin_xy=(0,0),chunk_pixels=chunk)
                plan=prepare_task(task,patch_m=20,origin_xy=(0,0),chunk_pixels=chunk,cell_indexed=True)
                for a,b in zip(baseline_plan.bands,plan.bands):
                    self.assertEqual(a.supported_count,b.supported_count);self.assertTrue(torch.equal(a.indices,b.indices))
                result=stream_task(candidate,decoder,task,plan,train_stds=stds,chunk_pixels=chunk,loss_weight=1.7,backward=True)
                self.assertAlmostEqual(result['weighted_loss'],float(expected.detach()),places=4)
                for p,q in zip(list(reference.parameters())+list(head.parameters()),list(candidate.parameters())+list(decoder.parameters())):
                    if p.grad is None:self.assertIsNone(q.grad)
                    else:torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=2e-4)

    def test_unsupported_is_not_zero_loss(self):
        observation=Observation([band(),band()],dict(id='empty',role='TRAIN'))
        task=spectral_task(observation,seed=0,update=0)
        task.targets[0].band.affine=(10.,0.,1000.,0.,-10.,1060.)
        plan=prepare_task(task,patch_m=20,origin_xy=(0,0),chunk_pixels=5)
        result=stream_task(Encoder(hyper_hidden=64),ReflectanceDecoder(indexed=True),task,plan,train_stds={},chunk_pixels=5,loss_weight=1.,backward=True)
        self.assertIsNone(result['loss'])

import unittest
import torch
from .data import Observation
from .tasks import spectral_task, fixed_spectral_task, spatial_task, cross_source_task, choose_patch, curriculum
from .losses import band_loss, hierarchical_loss
from .test_model import band


class TaskChecks(unittest.TestCase):
    def test_fixed_panel_hides_exact_reviewed_bands(self):
        observation=self.observation()
        task=fixed_spectral_task(observation,['0','2'])
        self.assertEqual([t.band.record_id for t in task.targets],['0','2'])
        self.assertEqual([b.record_id for b in task.visible],['1','3'])
        for bad in ([],['missing'],['0','0'],['0','1','2','3']):
            with self.assertRaises(ValueError):fixed_spectral_task(observation,bad)

    def observation(self, count=4):
        bands=[band() for _ in range(count)]
        for i,b in enumerate(bands): b.record_id=str(i)
        return Observation(bands,dict(id='example',role='TRAIN'))

    def test_b1_replay_and_isolation(self):
        observation=self.observation()
        a=spectral_task(observation,seed=0,update=12)
        for step in range(20): spectral_task(observation,seed=0,update=step)
        b=spectral_task(observation,seed=0,update=12)
        self.assertEqual(a.metadata['hidden_band_ids'],b.metadata['hidden_band_ids'])
        self.assertFalse(set(id(x) for x in a.visible)&set(id(x.band) for x in a.targets))
        counts=[len(spectral_task(observation,seed=0,update=i).targets) for i in range(100)]
        self.assertIn(1,counts); self.assertIn(3,counts)
        self.assertEqual(choose_patch(0,12,'example'),choose_patch(0,12,'example'))

    def test_spatial_guard_and_pair_gate(self):
        task=spatial_task(self.observation(),(5,0,35,60))
        self.assertGreater(task.metadata['protection_pixels'],0)
        for target,visible in zip(task.targets,task.visible):
            self.assertFalse(bool((target.mask & visible.keep()).any()))
        with self.assertRaises(ValueError): cross_source_task(self.observation(),self.observation(),{'qualified':True})

    def test_losses_and_curriculum(self):
        p=torch.tensor([2.,4.,float('nan')],requires_grad=True); y=torch.tensor([1.,2.,9.])
        loss,metrics=band_loss(p,y,torch.ones(3,dtype=torch.bool),torch.tensor([True,True,False]),train_std=2.)
        self.assertAlmostEqual(float(loss.detach()),.625)
        self.assertEqual(metrics['unsupported'],1)
        records=[dict(source='s1',observation='a',band='a',loss=torch.tensor(1.)),dict(source='s1',observation='a',band='b',loss=torch.tensor(3.)),dict(source='s2',observation='b',band='a',loss=torch.tensor(10.))]
        self.assertEqual(float(hierarchical_loss(records)),6.)
        self.assertIsNone(hierarchical_loss([]))
        self.assertEqual(curriculum(0),(1.,.25));self.assertEqual(curriculum(2000),(.25,1.))
        self.assertEqual(curriculum(9000),(.25,1.))


if __name__=='__main__': unittest.main()

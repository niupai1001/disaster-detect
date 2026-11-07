import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .tasks import Task
from .joint import PlannedView,accumulate_joint,layout
from .streaming import task_signature


class AbsentFamilyChecks(unittest.TestCase):
    def fixtures(self,missing=()):
        result=[]
        for i in range(8):
            for kind in ['A','B1']:
                task=Task(kind,[],[],dict(id=str(i),role='TRAIN',fixture_loss=1. if i<3 else 4.));family='A' if kind=='A' else 'B';plan=None if family in missing else SimpleNamespace(signature=task_signature(task),bands=[SimpleNamespace(supported_count=1 if i<3 else 99,indices=torch.arange(1),target=SimpleNamespace(band=SimpleNamespace(record_id='fixture')))])
                result.append(PlannedView(task,plan,{},'S2' if i<3 else 'Landsat'))
        return result

    def run_fixture(self,missing=()):
        def loss(encoder,decoder,task,*args,**kwargs):return dict(loss=task.metadata['fixture_loss'],tokens=1,bands=[])
        with patch('encoder_v3.joint.stream_task',side_effect=loss):
            return accumulate_joint(torch.nn.Linear(1,1),torch.nn.Linear(1,1),self.fixtures(missing),logical_update=0,chunk_pixels=1,precision='fp32',clip_norm=None,backward=False)

    def test_missing_A_is_none_not_zero_and_does_not_rescale_B(self):
        row=self.run_fixture(('A',));self.assertIsNone(row['A_loss']);self.assertIsNone(row['losses_by_kind']['A']);self.assertEqual(row['B_loss'],2.5);self.assertEqual(row['weighted_loss'],.625);self.assertEqual(row['effective_batch'],8);self.assertEqual(row['active_views'],8)
        self.assertTrue(all(x['result']['loss'] is None for x in row['rows'] if x['kind']=='A'))

    def test_missing_B_is_none_not_zero(self):
        row=self.run_fixture(('B',));self.assertIsNone(row['B_loss']);self.assertIsNone(row['losses_by_kind']['B1']);self.assertEqual(row['A_loss'],2.5);self.assertEqual(row['weighted_loss'],2.5)

    def test_supported_reference_weights_unchanged_despite_unequal_counts(self):
        row=self.run_fixture();self.assertEqual(row['A_loss'],2.5);self.assertEqual(row['B_loss'],2.5);self.assertEqual(row['weighted_loss'],3.125)
        families,weights=layout(self.fixtures())
        for values in families.values():
            self.assertAlmostEqual(sum(weights[id(v)] for v in values if v.loss_source=='S2'),.5);self.assertAlmostEqual(sum(weights[id(v)] for v in values if v.loss_source=='Landsat'),.5)

    def test_all_unsupported_aborts_without_dummy_loss(self):
        with self.assertRaisesRegex(ValueError,'no supported supervision'):self.run_fixture(('A','B'))

    def test_unreviewed_B2_still_rejected(self):
        values=self.fixtures();values[1].task.kind='B2';values[1].plan.signature=task_signature(values[1].task)
        with self.assertRaisesRegex(ValueError,'reviewed physical pair admission'):layout(values)

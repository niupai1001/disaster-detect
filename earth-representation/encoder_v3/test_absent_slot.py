import copy,unittest
from dataclasses import replace
from unittest.mock import patch
from . import test_absent_family as fixtures
from .joint import layout,accumulate_joint
from .plan_prefetch import move_views
from .route_step import update_signature
from .streaming import task_signature
import torch


class AbsentSlotChecks(unittest.TestCase):
    def views(self):
        views=fixtures.AbsentFamilyChecks().fixtures()
        for v in views:
            if v.plan is not None:v.plan.origin_xy=(0.,0.);v.plan.physical_patch_m=60.
        for i in range(1,8,2):
            v=views[i];v.task.kind='B2';v.plan=None;v.absence_reason='NO_ADMITTED_PAIR'
        return views

    def test_absent_B2_is_recorded_without_dummy_loss_or_pair_admission(self):
        views=self.views()
        def fake(encoder,decoder,task,*args,**kwargs):return dict(loss=task.metadata['fixture_loss'],tokens=1,bands=[])
        with patch('encoder_v3.joint.stream_task',side_effect=fake):row=accumulate_joint(torch.nn.Linear(1,1),torch.nn.Linear(1,1),views,logical_update=0,chunk_pixels=1,precision='fp32',clip_norm=None,backward=False)
        missing=[x for x in row['rows'] if x['kind']=='B2'];self.assertEqual(len(missing),4);self.assertTrue(all(x['result']['status']=='TASK_ABSENT' and x['result']['loss'] is None for x in missing));self.assertIsNone(row['losses_by_kind']['B2']);self.assertEqual(row['active_views'],12)

    def test_real_B2_target_or_plan_cannot_use_absence_to_bypass_admission(self):
        for field,value in [('targets',[object()]),('pair_admission',{'checks':{}}),('plan',object())]:
            values=self.views();v=values[1]
            if field=='targets':v.task.targets=value
            elif field=='plan':v.plan=value
            else:v.task.metadata[field]=value
            with self.assertRaises(ValueError):layout(values)
        values=self.views();values[1].absence_reason=None
        with self.assertRaisesRegex(ValueError,'reviewed physical pair'):layout(values)

    def test_CPU_transfer_preserves_absence_and_signature(self):
        values=self.views();before=update_signature(values);moved=move_views(values,'cpu');self.assertEqual([v.absence_reason for v in values],[v.absence_reason for v in moved]);self.assertEqual(update_signature(moved),before)
        changed=copy.deepcopy(values);changed[1].absence_reason='other';self.assertNotEqual(update_signature(changed),before)

    def test_reason_must_match_task_kind(self):
        values=self.views();values[1].absence_reason='NO_HIDEABLE_BAND'
        with self.assertRaisesRegex(ValueError,'eligibility reason'):layout(values)
        values=self.views();values[0].absence_reason='NO_ADMITTED_PAIR';values[0].plan=None
        with self.assertRaisesRegex(ValueError,'eligibility reason'):layout(values)

    def test_adaptive_control_records_absence_without_calling_its_stream(self):
        from .adaptive_joint import accumulate_adaptive_joint
        values=self.views();calls=[]
        def fake(encoder,decoder,task,*args,**kwargs):
            calls.append(task.kind)
            return dict(loss=task.metadata['fixture_loss'],tokens=1,bands=[])
        with patch('encoder_v3.adaptive_joint.stream_adaptive_task',side_effect=fake):
            row=accumulate_adaptive_joint(torch.nn.Linear(1,1),torch.nn.Linear(1,1),values,logical_update=0,chunk_pixels=1,clip_norm=None)
        self.assertEqual(len(calls),12);self.assertNotIn('B2',calls)
        missing=[x for x in row['AB'] if x['kind']=='B2'];self.assertTrue(all(x['result']['status']=='TASK_ABSENT' and x['result']['loss'] is None for x in missing))

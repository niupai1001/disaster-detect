import json
from pathlib import Path
import tempfile
import unittest
import torch
from .budget import PilotBudget
from .equivalence import require_cell_equivalence
from .pilot_protocol import load_study
from .pilot_fit import reconcile_pending,validation_means
from .plan_prefetch import PlanPrefetch,move_views
from .streaming import task_signature
from . import test_joint


class PilotFitChecks(unittest.TestCase):
    def test_prefetch_commit_only_resume_and_cpu_task_transfer(self):
        for prefetch in (False,True):
            stream=PlanPrefetch(lambda index:index*7,start=0,stop=3,identity='fixture',prefetch=prefetch)
            iterator=iter(stream);self.assertEqual(next(iterator),(0,0));self.assertEqual(stream.state_dict()['next_update'],0)
            iterator.close();self.assertEqual(stream.next_update,0)
            restored=PlanPrefetch(lambda index:index*7,start=stream.next_update,stop=3,identity='fixture',prefetch=prefetch)
            observed=[]
            for index,value in restored:observed.append(value);restored.commit(index)
            self.assertEqual(observed,[0,7,14]);self.assertEqual(restored.next_update,3)
        views=test_joint.JointChecks().views();moved=move_views(views,'cpu')
        self.assertEqual([task_signature(v.task) for v in views],[task_signature(v.task) for v in moved])

    def test_validation_hierarchy_and_missing_task(self):
        rows=[dict(panel='A',source='S2',observation='a',result={'loss':value}) for value in (1.,3.)]
        rows += [dict(panel='A',source='S2',observation='b',result={'loss':8.}),dict(panel='A',source='L8',observation='c',result={'loss':10.}),dict(panel='B2',source='L8',observation='c',result={'loss':None})]
        self.assertEqual(validation_means(rows),{'A':7.5})

    def test_pending_completion_requires_exact_durable_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder);budget=PilotBudget(root/'budget.json',study_sha256='a'*64)
            token=budget.reserve('CAdamW',0)
            with self.assertRaisesRegex(ValueError,'durable'):reconcile_pending(budget,root,{'fixture':True})
            (root/'CAdamW').mkdir();torch.save(dict(frozen_ids={'fixture':True},logical_update=1,sampler=dict(next_update=1,reservation=token)),root/'CAdamW/u000001.pt')
            reconcile_pending(budget,root,{'fixture':True})
            self.assertEqual(budget.summary()['completed'],1);self.assertEqual(budget.summary()['remaining_total'],399)

    def test_reviewed_limits_are_enforced(self):
        root=Path(__file__).parent;config=json.loads((root/'config.json').read_text(encoding='utf-8'));study=load_study(root/'reports/pilot_study_candidate.json')
        self.assertIn('PASS',require_cell_equivalence(config,root,study)['status'])
        with tempfile.TemporaryDirectory(dir=root) as folder:
            limit=json.loads((root/config['confirmed_preflight']['numerical_equivalence']['review']).read_text(encoding='utf-8'));limit['prediction_max_absolute']=0
            path=Path(folder)/'strict.json';path.write_text(json.dumps(limit),encoding='utf-8');config['confirmed_preflight']['numerical_equivalence']['review']=str(path.resolve())
            with self.assertRaisesRegex(ValueError,'equivalence failed'):require_cell_equivalence(config,root,study)

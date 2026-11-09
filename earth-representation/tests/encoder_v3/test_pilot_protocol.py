import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from encoder_v3.pilot_protocol import load_study,optimizer_groups,LinearWarmup,require_update_review


class PilotProtocolChecks(unittest.TestCase):
    def test_frozen_candidate_and_missing_execution_review(self):
        root=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')
        study=load_study(root/'reports/pilot_study_candidate.json')
        config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        config['required_before_pilot']={'execution_review':None}
        with self.assertRaisesRegex(ValueError,'execution_review'):require_update_review(config,study,root)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            changed=copy.deepcopy(study);changed['schedule'][0]['patch_m']=123.
            path=Path(folder)/'tampered.json';path.write_text(json.dumps(changed),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'content changed'):load_study(path)

    def test_optimizer_bias_norm_and_shared_parameters(self):
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.LayerNorm(4),torch.nn.MultiheadAttention(4,2,batch_first=True))
        groups=optimizer_groups([model,model],.04)
        decay={id(p) for g in groups if g['weight_decay']==.04 for p in g['params']}
        excluded={id(p) for g in groups if g['weight_decay']==0 for p in g['params']}
        self.assertFalse(decay&excluded)
        self.assertEqual(len(decay|excluded),len(list(model.parameters())))
        for name,p in model.named_parameters():
            if name.endswith('bias') or name.startswith('1.'):self.assertIn(id(p),excluded)
            else:self.assertIn(id(p),decay)
        self.assertIn(id(model[2].in_proj_bias),excluded)

    def test_warmup_progress_and_restore(self):
        model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        schedule=LinearWarmup(optimizer,peak=3e-4,warmup=100)
        schedule.apply(0);self.assertAlmostEqual(optimizer.param_groups[0]['lr'],3e-6)
        state=schedule.state_dict();schedule.apply(99);self.assertEqual(optimizer.param_groups[0]['lr'],3e-4)
        schedule.load_state_dict(state);self.assertEqual(schedule.last_update,0)
        schedule.apply(1);self.assertAlmostEqual(optimizer.param_groups[0]['lr'],6e-6)
        schedule.apply(199);self.assertEqual(optimizer.param_groups[0]['lr'],3e-4)
        with self.assertRaises(ValueError):schedule.apply(-1)

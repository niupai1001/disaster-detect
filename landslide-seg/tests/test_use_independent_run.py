import importlib.util
import json
from pathlib import Path
import sys

import pytest

scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('independent_run_audit',scripts/'audit_use_independent_run.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_trace_rejects_hidden_target_leak_and_wrong_members(tmp_path):
    p=tmp_path/'trace.jsonl'
    config=dict(batch=1,updates=1,train_ids={'s2':['train.npz']})
    members=[dict(file='train.npz',role='TRAIN',sensor='s2',band_ids=['a','b'])]
    row=dict(step=1,loss=1.,lr=audit.learning_rate(1),exposure=[dict(file='train.npz',sensor='s2',visible_bands=['a'],target_bands=['b'])])
    def write():p.write_text(json.dumps(row)+'\n')
    write();assert audit.audit_trace(p,config,members)['observations']==1
    row['exposure'][0]['target_bands']=['a'];write()
    with pytest.raises(ValueError,match='overlap'):
        audit.audit_trace(p,config,members)
    row['exposure'][0]['file']='val.npz';write()
    with pytest.raises(ValueError,match='TRAIN'):
        audit.audit_trace(p,config,members)

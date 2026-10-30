"""Strict complete-state comparison of disposable original-path resumes."""
import json
from pathlib import Path
import sys
import numpy as np
import torch


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    else:assert a==b,path


if __name__=='__main__':
    base=Path('outputs/use_next_v2/geometry_recovery')
    a=torch.load(base/'original_a/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(base/'original_b/last.pt',map_location='cpu',weights_only=False)
    assert a['step']==b['step']==1679
    equal(a,b)
    result=dict(status='ORIGINAL_FD_FULL_STATE_EXACT',initial_step=1678,final_step=1679,
        new_engineering_optimizer_updates=2,formal_updates=0,
        exact_fields=list(a),loss=a['history'][-1]['loss'],
        expected_failure_json='SystemExit(0) is the explicit post-atomic-save probe stop; not a native crash',
        candidate_adopted=False,root_cause='UNCONFIRMED',production_uuid_lock_modified=False)
    (base/'original_cuda_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

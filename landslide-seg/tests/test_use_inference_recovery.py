import json
from pathlib import Path
import runpy
import pytest


def test_cpu_inference_receipt_cannot_reopen_cuda_training_gate(tmp_path):
    path=Path(__file__).resolve().parents[1]/'outputs/use_independent_frozen/run_s2_probes.py'
    check=runpy.run_path(str(path))['require_cuda_runtime_gate']
    gate=tmp_path/'cuda_runtime_gate.json'
    gate.write_text(json.dumps(dict(passed=True,inference_device='cpu')))
    with pytest.raises(RuntimeError,match='CPU inference does not authorize'):
        check(tmp_path)
    gate.write_text(json.dumps(dict(passed=True,cuda_available=True,gpu_backward_finite=True,real_data_engineering_passed=True)))
    check(tmp_path)
    gate.write_text(json.dumps(dict(passed=False,cuda_available=False)))
    with pytest.raises(RuntimeError,match='runtime gate failed'):
        check(tmp_path)


def test_inference_recovery_preserves_failure_and_never_restarts_training():
    path=Path(__file__).resolve().parents[1]/'outputs/use_independent_frozen/run_s2_probes.py'
    recover=runpy.run_path(str(path))['apply_inference_receipts']
    bad=dict(label='eval_probe_shallow_attention_best_full',command=['python','best'],status='failed',exit_code=3221225477)
    train=dict(label='formal_probe_shallow_attention',command=['python','train'],status='finished',exit_code=0)
    state=dict(status='failed',jobs=[train,bad])
    receipt=dict(passed=True,new_training_updates=0,inference_device='cpu',commands=[
        dict(label='eval_probe_shallow_attention_best_full',command=['python','best'],status='finished',exit_code=0),
        dict(label='eval_probe_shallow_attention_last_full',command=['python','last'],status='finished',exit_code=0)])
    repaired=recover(state,receipt)
    assert state['jobs'][1]['status']=='failed'
    assert repaired['jobs'][0]==train and len(repaired['jobs'])==3
    assert repaired['jobs'][1]['previous_failure']==bad
    assert repaired['jobs'][1]['inference_device']=='cpu'
    receipt['new_training_updates']=1
    with pytest.raises(RuntimeError,match='read-only'):
        recover(state,receipt)


def test_waiter_cannot_resume_when_training_started_or_parent_still_failed():
    path=Path(__file__).resolve().parents[1]/'outputs/use_independent_frozen/run_multisensor_pretraining.py'
    validate=runpy.run_path(str(path))['validate_wait_recovery']
    state=dict(status='failed',jobs=[],error='S2 probe queue failed; no next-stage training')
    validate(state,dict(status='running'))
    with pytest.raises(RuntimeError,match='resumed first'):
        validate(state,dict(status='failed'))
    state['jobs']=[dict(label='formal_pretraining_multisensor')]
    with pytest.raises(RuntimeError,match='Only the recorded'):
        validate(state,dict(status='running'))

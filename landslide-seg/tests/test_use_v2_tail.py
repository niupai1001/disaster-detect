"""Second-stage tail: coverage, schedule endpoints and the strict 30k source gate."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_use_v2_tail as tail


def test_tail_allocation_covers_jobs_and_balances_cards():
    jobs = tail.tail_jobs()
    plan = tail.tail_allocation()
    assert len(jobs) == 10 and set(plan) == set(jobs)
    assert {plan[j] for j in plan} == {'cuda:0', 'cuda:1'}
    assert plan['probe_tail_attention_fine'] == 'cuda:0' and plan['probe_tail_mean_fine'] == 'cuda:1'
    steps = {card: sum(jobs[j]['steps'] for j in plan if plan[j] == card) for card in ('cuda:0', 'cuda:1')}
    assert steps == {'cuda:0': 17000, 'cuda:1': 17000}
    waves = tail.tail_waves()
    assert [w['wave'] for w in waves] == ['evaluation']
    assert sum(len(v) for v in waves[0]['jobs'].values()) == 10


def test_tail_schedule_endpoints():
    import yaml
    spec = yaml.safe_load((tail.CONTRACT / 'execution.yaml').read_text(encoding='utf8'))['use_v2']['tail']
    assert tail.lr(spec['warmup'], spec['steps'], spec['warmup']) == pytest.approx(spec['lr'])
    assert tail.lr(spec['steps'], spec['steps'], spec['warmup']) == pytest.approx(0.0, abs=1e-9)
    assert spec['global_step_offset'] == 20000 and spec['steps'] == 10000
    assert tail.TOTAL == 30000


def test_source_gate_requires_thirty_thousand_updates(tmp_path, monkeypatch):
    import run_use_v2 as runtime
    run = tmp_path / 'runs' / 'pretrain_mean_tail'
    run.mkdir(parents=True)
    (run / 'u30000.pt').write_bytes(b'weights')
    monkeypatch.setattr(runtime, 'BASE', tmp_path)
    job = dict(source='pretrain_mean_tail/u30000.pt')
    summary = dict(complete=True, updates=30000, checkpoints={'u30000.pt': runtime.sha(run / 'u30000.pt')})
    runtime.write_json(run / 'summary.json', summary)
    assert tail.source_checkpoint(job) == run / 'u30000.pt'
    summary['updates'] = 20000
    runtime.write_json(run / 'summary.json', summary)
    with pytest.raises(ValueError, match='incomplete'):
        tail.source_checkpoint(job)
    summary['updates'] = 30000
    summary['checkpoints']['u30000.pt'] = '0' * 64
    runtime.write_json(run / 'summary.json', summary)
    with pytest.raises(ValueError, match='changed'):
        tail.source_checkpoint(job)


def test_cpu_job_does_not_block_a_card(tmp_path, monkeypatch):
    import run_use_v2 as runtime
    run = tmp_path / 'runs' / 'smoke'
    run.mkdir(parents=True)
    runtime.write_json(run / 'process.json', dict(status='RUNNING', pid=os.getpid(), device='cpu'))
    monkeypatch.setattr(runtime, 'BASE', tmp_path)
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    assert tail.active_jobs_cuda_only() == []
    runtime.write_json(run / 'process.json', dict(status='RUNNING', pid=os.getpid(), device='cuda:0'))
    assert tail.active_jobs_cuda_only() == [('runs/smoke', 0)]

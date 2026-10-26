"""Two-card launch gate: display use allowed, memory, duplicate-start and mapping rules enforced."""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_use_v2 as runtime

CARDS = {0: dict(uuid='GPU-a', display='Enabled', free_mib=20000),
         1: dict(uuid='GPU-b', display='Disabled', free_mib=20000)}


@pytest.fixture
def card(monkeypatch):
    monkeypatch.setattr(runtime, 'gpu_cards', lambda: CARDS)
    monkeypatch.setattr(runtime, 'active_jobs', lambda: [])
    monkeypatch.setattr(runtime, 'probe_device', lambda index: dict(probe='stub', free_mib_after_probe=19000,
                                                                  total_mib=32607))
    return monkeypatch


def test_display_card_is_accepted(card):
    gate = runtime.gpu_available('cuda:0')
    assert gate['physical_index'] == 0 and gate['display_active'] == 'Enabled'
    assert gate['uuid'] == 'GPU-a' and gate['probe'] == 'stub'


def test_low_free_memory_is_rejected(card):
    card.setattr(runtime, 'gpu_cards', lambda: {1: dict(uuid='GPU-b', display='Disabled', free_mib=1024)})
    with pytest.raises(ValueError, match='MiB required'):
        runtime.gpu_available('cuda:1')


def test_running_job_on_the_same_card_is_rejected(card):
    card.setattr(runtime, 'active_jobs', lambda: [('outputs/use_v2/runs/pretrain_mean', 1)])
    with pytest.raises(ValueError, match='already hosts a running'):
        runtime.gpu_available('cuda:1')
    assert runtime.gpu_available('cuda:0')['physical_index'] == 0


def test_visible_device_mapping_is_recorded(card, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,0')
    gate = runtime.gpu_available('cuda:0')
    assert gate['physical_index'] == 1 and gate['visible_devices'] == '1,0'
    with pytest.raises(ValueError, match='does not expose'):
        runtime.gpu_available('cuda:2')


def test_one_process_per_card_and_stale_lock_recovery(tmp_path, card):
    card.setattr(runtime, 'BASE', tmp_path)
    first = runtime.claim_card(0)
    assert first.exists()
    card.setattr(runtime, 'card_process_alive', lambda pid: True)
    with pytest.raises(ValueError, match='already hosts'):
        runtime.claim_card(0)
    card.setattr(runtime, 'card_process_alive', lambda pid: False)
    second = runtime.claim_card(0)
    assert second.exists() and second.read_text() == str(os.getpid())


def test_two_card_allocation_covers_every_job_once():
    jobs = runtime.matrix()
    plan = runtime.allocation()
    assert set(plan) == set(jobs) and len(plan) == 30
    assert {plan[j] for j in jobs} == {'cuda:0', 'cuda:1'}
    assert plan['pretrain_mean'] == 'cuda:0' and plan['pretrain_attention'] == 'cuda:1'
    assert sum(jobs[j]['steps'] for j in plan if plan[j] == 'cuda:0') == 46000
    assert sum(jobs[j]['steps'] for j in plan if plan[j] == 'cuda:1') == 46000
    waves = runtime.waves()
    flat = [j for wave in waves for jobs_on_card in wave['jobs'].values() for j in jobs_on_card]
    assert sorted(flat) == sorted(jobs) and len(flat) == 30
    assert waves[0]['jobs'] == {'cuda:0': ['pretrain_mean'], 'cuda:1': ['pretrain_attention']}


def test_cpu_needs_no_gate():
    assert runtime.gpu_available('cpu') == {'device': 'cpu'}

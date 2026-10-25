from pathlib import Path
import runpy
import json
import pytest

QUEUE = Path(__file__).resolve().parents[1]/'outputs/use_independent_frozen/run_s2_probes.py'


def test_status_write_retries_transient_windows_sharing_error(tmp_path, monkeypatch):
    queue = runpy.run_path(str(QUEUE))
    original = Path.replace
    attempts = []

    def replace(path, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError('transient Windows sharing error')
        return original(path, target)

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(queue['time'], 'sleep', lambda seconds:None)
    queue['save_status'](tmp_path/'status.json', dict(status='running'))
    assert len(attempts) == 3
    assert queue['read'](tmp_path/'status.json') == dict(status='running')


def test_completed_commands_require_exact_identity_and_success():
    queue = runpy.run_path(str(QUEUE))
    state = dict(jobs=[dict(label='train', command=['python','train'], status='finished', exit_code=0)])
    assert queue['already_completed'](state, ['python','train'], 'train')
    assert not queue['already_completed'](state, ['python','next'], 'next')
    with pytest.raises(RuntimeError, match='identity'):
        queue['already_completed'](state, ['python','other'], 'train')
    state['jobs'][0]['status'] = 'running'
    with pytest.raises(RuntimeError, match='unverified'):
        queue['already_completed'](state, ['python','train'], 'train')


def test_recovery_requires_terminal_children_and_unchanged_accepted_weights(tmp_path, monkeypatch):
    queue = runpy.run_path(str(QUEUE))
    recover = queue['check_recovery']
    monkeypatch.setitem(recover.__globals__, 'OUT', tmp_path)
    monkeypatch.setitem(recover.__globals__, 'CANDIDATES', ('random',))
    state = dict(status='failed', error='[WinError 5] status replacement failed',
                 jobs=[dict(label='formal_probe_linear_random', status='finished', exit_code=0)])
    folder = tmp_path/'probes/s2_10band_v4r1_p20/use_independent_probe_linear_random/1234'
    folder.mkdir(parents=True)
    for name in ('best','last'):
        (folder/f'{name}.pt').write_bytes(b'accepted weight')
    proof = dict(passed=True, checkpoints={name:dict(checkpoint_sha256=queue['sha'](folder/f'{name}.pt')) for name in ('best','last')})
    (folder/'acceptance.json').write_text(json.dumps(proof))
    recover(state)
    (folder/'last.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='weights changed'):
        recover(state)
    state['jobs'][0]['status'] = 'running'
    with pytest.raises(RuntimeError, match='Unresolved child'):
        recover(state)

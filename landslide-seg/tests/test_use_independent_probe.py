import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_use_independent_probe import check_history


def test_selection_and_frozen_history_reject_pollution():
    history = [dict(global_step=step, val_iou=.2, samples_exposed=step*8, skipped_batches=0,
                    probe_integrity=dict(use_sha256='fixed', encoder_unchanged=True, encoder_has_grad=False, encoder_training=False),
                    dev=dict(by_event=dict(hiroshima=dict(iou=.1), thrissur=dict(iou=.3))))
               for step in range(100, 2001, 100)]
    assert check_history(history, 'fixed')['global_step'] == 100
    for key, value in [('use_sha256', 'changed'), ('encoder_has_grad', True), ('encoder_training', True)]:
        bad = copy.deepcopy(history)
        bad[7]['probe_integrity'][key] = value
        with pytest.raises(ValueError, match='integrity'):
            check_history(bad, 'fixed')
    with pytest.raises(ValueError, match='validation points'):
        check_history(history[:-1], 'fixed')

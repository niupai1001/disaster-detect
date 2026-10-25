from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_use_probe_subsets import check_report,evaluation_code


def test_subset_uses_same_weight_and_fixed_event_members():
    report = dict(band_subset='rgb',checkpoint_sha256='weight',manifest_sha256='members',stats_sha256='stats',
        n_samples=669,role='DEV',threshold=.5,event_mean_iou=.2,
        by_event=dict(hiroshima=dict(n_tiles=448,pixel=dict(iou=.1)),thrissur=dict(n_tiles=221,pixel=dict(iou=.3))))
    assert check_report(report,'rgb','weight','members','stats') == .2
    with pytest.raises(ValueError,match='provenance'):
        check_report(report,'rgb','other weight','members','stats')
    report['by_event']['thrissur']['n_tiles'] = 220
    with pytest.raises(ValueError,match='event members'):
        check_report(report,'rgb','weight','members','stats')


def test_cpu_thread_setting_keeps_seed_and_evaluator_identical():
    gpu=evaluation_code('cuda:1');cpu=evaluation_code('cpu')
    assert cpu=='import torch; torch.set_num_threads(4); '+gpu
    assert 'set_seed(1234,True)' in cpu and 'evaluate_segformer.py' in cpu

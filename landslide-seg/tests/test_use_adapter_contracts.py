from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from freeze_use_adapters import select_common, prerequisites


def test_common_use_is_chosen_by_unlabeled_score_and_fixed_tie_rule():
    proofs = dict(multisensor=dict(passed=True, best_score=.4), spatial=dict(passed=True, best_score=.3))
    assert select_common(proofs) == 'spatial'
    proofs['multisensor']['best_score'] = .3
    assert select_common(proofs) == 'multisensor'
    proofs['spatial']['passed'] = False
    with pytest.raises(ValueError, match='Accepted finite'):
        select_common(proofs)


def test_missing_probe_gate_stops_before_any_adapter_directory_is_created(tmp_path, monkeypatch):
    import freeze_use_adapters
    monkeypatch.setattr(freeze_use_adapters, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        prerequisites()
    assert not list(tmp_path.iterdir())


def test_adapters_keep_shared_data_budget_loss_and_use(tmp_path, monkeypatch):
    import copy
    import yaml
    import freeze_use_adapters
    project = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project/'configs/segformer/use/physical_mixed.yaml').read_text())
    before = copy.deepcopy(base)
    weights = tmp_path/'best.pt'
    weights.write_bytes(b'fixture')
    monkeypatch.setattr(freeze_use_adapters, 'ROOT', tmp_path)
    configs = [freeze_use_adapters.adapter_config(base, mode, 'spatial', weights, {})
               for mode in freeze_use_adapters.MODES]
    for config in configs:
        s = config['segformer']
        assert s['train'] == base['segformer']['train']
        assert s['loss'] == base['segformer']['loss']
        assert s['optim'] == base['segformer']['optim']
        assert s['data'] == dict(base['segformer']['data'], mixed_bands=False)
        assert s['model']['freeze_use'] and s['model']['use_pretrained'] == 'best.pt'
    assert base == before

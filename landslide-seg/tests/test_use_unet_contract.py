from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import freeze_use_unet


def test_unet_cannot_change_source_before_original_adapter_chain_finishes(tmp_path,monkeypatch):
    monkeypatch.setattr(freeze_use_unet,'OUT',tmp_path)
    state=tmp_path/'common_use_adapter_status.json'
    state.write_text(json.dumps(dict(status='formal_adapters',jobs=[])))
    with pytest.raises(ValueError,match='must finish'):
        freeze_use_unet.require_finished_original_pipeline()
    assert list(tmp_path.iterdir())==[state]


def test_claimed_finish_with_failed_command_does_not_open_unet_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(freeze_use_unet,'OUT',tmp_path)
    (tmp_path/'common_use_adapter_status.json').write_text(json.dumps(dict(
        status='finished_requires_final_study_publication_and_acceptance',jobs=[dict(status='failed',exit_code=1)])))
    with pytest.raises(ValueError,match='unresolved'):
        freeze_use_unet.require_finished_original_pipeline()


def test_unet_reuses_common_encoder_and_rejects_changed_weight(tmp_path,monkeypatch):
    import hashlib
    monkeypatch.setattr(freeze_use_unet,'OUT',tmp_path)
    common=tmp_path/'adapter_contracts';common.mkdir()
    pretrained=tmp_path/'pretraining/multisensor';pretrained.mkdir(parents=True)
    weights=pretrained/'best.pt';weights.write_bytes(b'frozen common weight fixture')
    (pretrained/'resolved_config.json').write_text(json.dumps(dict(use_config={})))
    (common/'acceptance.json').write_text(json.dumps(dict(passed=True,common_use='multisensor',
        common_checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),selection='unlabeled MBM validation')))
    candidate,path,config,stage,basis=freeze_use_unet.source_for()
    assert candidate=='multisensor' and path==weights and config=={} and stage=='independent_ssl'
    assert basis['basis']=='unlabeled_validation'
    weights.write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='changed'):
        freeze_use_unet.source_for()

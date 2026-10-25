import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('independent_preflight',Path(__file__).resolve().parents[1]/'scripts/check_research_protocol.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_independent_sources_reject_unfrozen_or_changed_members(tmp_path):
    folder = tmp_path/'outputs/use_independent_frozen'
    audit_dir, training = folder/'cache_audit_v1', folder/'pretraining/physical'
    audit_dir.mkdir(parents=True); training.mkdir(parents=True)
    def write(path, data):
        path.write_text(json.dumps(data),encoding='utf8')
    members = audit_dir/'members.jsonl'
    members.write_text('\n'.join(json.dumps(row) for row in [dict(file='train.npz',sensor='s2',role='TRAIN'),dict(file='val.npz',sensor='s2',role='PRETRAIN_VAL')]))
    members_sha = preflight.sha(members)
    acceptance = audit_dir/'acceptance.json'
    write(acceptance,dict(ok=True,members_sha256=members_sha,source_sha256={}))
    weights = training/'best.pt'; weights.write_bytes(b'checkpoint bytes verified here; tensor loading belongs to factory')
    config = dict(updates=10000,task_labels=False,supervision='masked_band_reconstruction_only',use_config={},
                  cache_acceptance_sha256=preflight.sha(acceptance),members_sha256=members_sha,
                  train_ids={'s2':['train.npz']},validation_ids={'s2':['val.npz']})
    config_path = training/'resolved_config.json'
    write(config_path,config)
    summary = dict(complete=True,updates=10000,config_sha256=preflight.sha(config_path),members_sha256=members_sha,best_sha256=preflight.sha(weights))
    write(training/'summary.json',summary)
    model = dict(freeze_use=True,use_pretrained=str(weights),use_pretrained_sha256=preflight.sha(weights),use={})
    assert str(members) in preflight.independent_pretraining_sources(tmp_path,model)
    with pytest.raises(ValueError,match='必须冻结'):
        preflight.independent_pretraining_sources(tmp_path,dict(model,freeze_use=False))
    config['train_ids']['s2']=['val.npz']
    write(config_path,config);summary['config_sha256']=preflight.sha(config_path);write(training/'summary.json',summary)
    with pytest.raises(ValueError,match='实际训练'):
        preflight.independent_pretraining_sources(tmp_path,model)

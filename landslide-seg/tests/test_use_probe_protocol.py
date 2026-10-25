import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from landslide_seg.models.frozen_use import FrozenUSE, state_sha256
from landslide_seg.training.segformer_trainer import initialization_identity

spec=importlib.util.spec_from_file_location('probe_preflight',Path(__file__).resolve().parents[1]/'scripts/check_research_protocol.py')
preflight=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_formal_initialization_and_optimizer_must_match():
    model=nn.Module()
    model.use=FrozenUSE(nn.Linear(2,2))
    model.head=nn.Linear(2,1)
    optimizer=torch.optim.AdamW(model.head.parameters())
    expected=state_sha256(model)
    result=initialization_identity(model,optimizer,expected)
    assert result['trainable_names']==['head.weight','head.bias']
    assert result['use_sha256']==model.use.reference_sha256
    with pytest.raises(ValueError,match='initialization'):
        initialization_identity(model,optimizer,'changed')
    with pytest.raises(RuntimeError,match='Optimizer'):
        initialization_identity(model,torch.optim.AdamW(model.parameters()),expected)


def test_random_probe_cannot_load_a_trained_weight(tmp_path):
    folder=tmp_path/'outputs/use_independent_frozen'
    run=folder/'pretraining/physical';run.mkdir(parents=True)
    weights=run/'initialization.pt';weights.write_bytes(b'untrained initialization')
    config=dict(use_config={},task_labels=False)
    cp=run/'resolved_config.json';cp.write_text(json.dumps(config))
    ap=folder/'frozen_contracts_acceptance.json'
    ap.write_text(json.dumps(dict(passed=True,common_initialization_verified=True,records={'physical':dict(initialization_file_sha256=preflight.sha(weights),config_sha256=preflight.sha(cp))})))
    model=dict(freeze_use=True,use_downstream='linear',use_pretrained=str(weights),use_pretrained_sha256=preflight.sha(weights),use={})
    assert str(weights) in preflight.independent_random_sources(tmp_path,model)
    with pytest.raises(ValueError,match='未训练'):
        preflight.independent_random_sources(tmp_path,dict(model,use_pretrained=str(run/'best.pt')))

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from use_next_p3 import run_guarded_p2_head as guard
from use_next_p2.run import tensor_digest
from use_next_p2.protocol import sha


def test_complete_same_mode_lineage_required(tmp_path,monkeypatch):
    monkeypatch.setattr(guard,'ROOT',tmp_path)
    ready=tmp_path/'outputs/use_next_v2/audit/READY_P2.json';ready.parent.mkdir(parents=True)
    ready.write_text(json.dumps(dict(status='READY_P2',release_sha256='parent')))
    release=tmp_path/'release';release.mkdir();(release/'manifest.json').write_text(json.dumps(dict(parent_release_sha256='parent')))
    config=tmp_path/'configs/use_next_p2/pixel_pretrain.yaml';config.parent.mkdir(parents=True);config.write_text('test')
    identity=dict(engineering=False,release_sha256='parent',mode='pixel',stage='P2',task='pretrain',config_sha256=sha(config))
    source=tmp_path/'runs/pretrain/last.pt';source.parent.mkdir(parents=True)
    local={'optical.weight':torch.zeros(2,1)}
    saved=dict(step=6000,identity=identity,model={'use.local.'+k:v for k,v in local.items()})
    torch.save(saved,source)
    (source.parent/'summary.json').write_text(json.dumps(dict(complete=True,updates=6000,identity=identity,final_local_sha256=tensor_digest(local))))
    marker=dict(parent_release_sha256='parent',ready_p2_sha256=sha(ready))
    values=dict(mode='pixel',frozen_source='runs/pretrain/last.pt')
    guard.check_dependencies(values,marker,release)
    saved['step']=5999;torch.save(saved,source)
    with pytest.raises(ValueError):guard.check_dependencies(values,marker,release)

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from use_next_p3 import run


def test_complete_p3_source_binding(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    config=tmp_path/'configs/use_next_p3/joint_pretrain.yaml'
    config.parent.mkdir(parents=True);config.write_text('same config')
    release=tmp_path/'release';release.mkdir();(release/'manifest.json').write_text('manifest')
    source=tmp_path/'runs/joint/last.pt';source.parent.mkdir(parents=True)
    identity=dict(engineering=False,release_sha256=run.sha(release/'manifest.json'),config_sha256=run.sha(config))
    local={'weight':torch.ones(2)}
    saved=dict(step=10000,identity=identity,model={'use.local.weight':local['weight']})
    torch.save(saved,source)
    summary=dict(updates=10000,complete=True,identity=identity,final_local_sha256=run.engine.tensor_digest(local))
    report=source.parent/'summary.json';report.write_text(json.dumps(summary))
    values=dict(source_objective='joint',frozen_source='runs/joint/last.pt')
    run.check_source(None,values,release)
    summary['complete']=False;report.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='complete'):run.check_source(None,values,release)
    summary['complete']=True;summary['final_local_sha256']='wrong';report.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='tensor hash'):run.check_source(None,values,release)

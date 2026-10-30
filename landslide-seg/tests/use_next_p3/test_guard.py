import sys
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from landslide_seg.models.use_next_p3.readout import GuardedReadout
from landslide_seg.models.use_next_p2.readout import ScaleReadout
from landslide_seg.models.use_next_v2.readout import state_digest


def test_guard_preserves_forward_and_detects_mutation():
    assert GuardedReadout.forward is ScaleReadout.forward and GuardedReadout.train is ScaleReadout.train
    model=GuardedReadout.__new__(GuardedReadout);torch.nn.Module.__init__(model)
    model.use=torch.nn.Linear(2,2).requires_grad_(False)
    model.frozen_encoder_sha256=state_digest(model.use)
    before={k:v.clone() for k,v in model.state_dict().items()}
    model.after_optimizer(0,3000)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    with torch.no_grad():model.use.weight.add_(1)
    with pytest.raises(ValueError):model.after_optimizer(1,3000)

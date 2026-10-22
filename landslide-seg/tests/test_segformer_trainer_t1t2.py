"""T1（首层适配学习率）与 T2（事件/尺度均衡采样）的单元测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"

pytestmark = pytest.mark.skipif(not PD.is_dir(), reason="B0 权重未下载")


def _model(adapter="direct10", stem=False):
    from landslide_seg.models.segformer import SegFormerSeg

    return SegFormerSeg(adapter=adapter, stem=stem, pretrained_dir=str(PD))


def test_first_conv_lr_splits_only_the_adapter_layer():
    from landslide_seg.training.segformer_trainer import _make_optimizer

    cfg = {"segformer": {"optim": {
        "encoder_lr": 6e-5, "head_lr": 6e-4, "first_conv_lr": 6e-4,
        "weight_decay": 0.01,
    }}}
    model = _model("direct10")
    opt = _make_optimizer(model, cfg)
    assert len(opt.param_groups) == 3
    assert [g["lr"] for g in opt.param_groups] == [6e-5, 6e-4, 6e-4]
    first_group = opt.param_groups[1]["params"]
    # direct10 的首层 Conv2d(10,32,7,7) = 15,680 权重 + 32 bias
    assert sum(p.numel() for p in first_group) == 10 * 32 * 7 * 7 + 32


def test_first_conv_lr_without_adapter_raises():
    from landslide_seg.training.segformer_trainer import _make_optimizer

    cfg = {"segformer": {"optim": {
        "encoder_lr": 6e-5, "head_lr": 6e-4, "first_conv_lr": 6e-4,
        "weight_decay": 0.01,
    }}}
    with pytest.raises(ValueError, match="direct10"):
        _make_optimizer(_model("project10to3"), cfg)


def test_scale_bin_boundaries():
    from landslide_seg.training.segformer_trainer import _scale_bin

    assert _scale_bin(0) == "empty"
    assert _scale_bin(1) == "small"
    assert _scale_bin(49) == "small"
    assert _scale_bin(50) == "medium"
    assert _scale_bin(199) == "medium"
    assert _scale_bin(200) == "large"


def _records():
    rows = []
    # 事件 A 多、事件 B 少；尺度以小目标为主
    for i in range(20):
        rows.append({"sample_id": f"a{i}", "event_id": "A", "mask_px": 10 if i < 15 else 300})
    for i in range(4):
        rows.append({"sample_id": f"b{i}", "event_id": "B", "mask_px": 10})
    return rows


def test_balanced_weights_sum_and_rare_event_upweighted():
    from landslide_seg.training.segformer_trainer import _balanced_weights

    rows = _records()
    w = _balanced_weights(rows, event_power=0.5, scale_power=0.5, max_weight=4.0)
    assert len(w) == len(rows)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    w_a = float(np.mean(w[:20]))
    w_b = float(np.mean(w[20:]))
    assert w_b > w_a  # 稀有事件 B 权重更高


def test_balanced_indices_deterministic_and_covers_events():
    from landslide_seg.training.segformer_trainer import _balanced_indices

    rows = _records()
    cfg = {"segformer": {"sampling": {
        "mode": "balanced", "event_power": 0.5, "scale_power": 0.5, "max_weight": 4.0,
    }}}
    a = _balanced_indices(rows, cfg, seed=1234, epoch=0)
    b = _balanced_indices(rows, cfg, seed=1234, epoch=0)
    assert a == b
    assert len(a) == len(rows)
    seen = set()
    for epoch in range(20):
        for idx in _balanced_indices(rows, cfg, seed=1234, epoch=epoch):
            seen.add(rows[idx]["event_id"])
    assert seen == {"A", "B"}

"""时间控制接口的行为验收：帧选择、T=1 恒等、填充不变、汇聚语义与日期梯度。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from landslide_seg.config import load_segformer_config
from landslide_seg.data.temporal_v6 import select_frame_indices
from landslide_seg.models.spectral_temporal import TemporalPool, TemporalSpectralSegFormer

ROOT = Path(__file__).resolve().parents[1]
STATS = json.loads((ROOT / "manifests/v4r1/stats_core25.json").read_text(encoding="utf-8"))
PRETRAINED = "weights/segformer/segformer-b0-finetuned-ade-512-512"


def _model(aggregation: str) -> TemporalSpectralSegFormer:
    torch.manual_seed(0)
    return TemporalSpectralSegFormer(
        spectral=False, backbone="b0", pretrained_dir=PRETRAINED,
        stats=STATS, aggregation=aggregation,
    ).eval()


@pytest.fixture(scope="module")
def identity_model():
    return _model("identity")


@pytest.mark.parametrize("aggregation", ["mean", "content", "date"])
def test_t1_identity_matches_core(identity_model, aggregation):
    model = _model(aggregation)
    model.core.load_state_dict(identity_model.core.state_dict())
    model.eval()
    x = torch.randn(2, 10, 64, 64)
    dates = torch.tensor([[1200.0], [1500.0]])
    keep = torch.ones(2, 1, dtype=torch.bool)
    anchor = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        new = model(x, dates, keep, anchor)["logits"]
        core = identity_model.core(x)["logits"]
    assert torch.allclose(new, core, atol=1e-6), (new - core).abs().max()


@pytest.mark.parametrize("aggregation", ["mean", "content", "date"])
def test_padding_invariance(identity_model, aggregation):
    model = _model(aggregation)
    model.core.load_state_dict(identity_model.core.state_dict())
    model.eval()
    x = torch.randn(1, 6, 10, 64, 64)
    dates = torch.arange(6.0).reshape(1, 6) * 30 + 1000
    keep = torch.ones(1, 6, dtype=torch.bool)
    anchor = torch.tensor([3])
    with torch.no_grad():
        full = model(x, dates, keep, anchor)["logits"]
        padded = torch.cat([x, torch.zeros(1, 4, 10, 64, 64)], dim=1)
        p_dates = torch.cat([dates, torch.zeros(1, 4)], dim=1)
        p_keep = torch.cat([keep, torch.zeros(1, 4, dtype=torch.bool)], dim=1)
        out = model(padded, p_dates, p_keep, anchor)["logits"]
    assert torch.allclose(full, out, atol=1e-6), (full - out).abs().max()


def test_content_ignores_dates(identity_model):
    model = _model("content")
    model.core.load_state_dict(identity_model.core.state_dict())
    model.eval()
    x = torch.randn(1, 5, 10, 64, 64)
    keep = torch.ones(1, 5, dtype=torch.bool)
    anchor = torch.tensor([2])
    with torch.no_grad():
        a = model(x, torch.tensor([[100.0, 200, 300, 400, 500]]), keep, anchor)["logits"]
        b = model(x, torch.tensor([[900.0, 950, 990, 1000, 1200]]), keep, anchor)["logits"]
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_mean_pool_weights():
    pool = TemporalPool(4, "mean", width=8)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1, 1)
    keep = torch.tensor([[True, False, True, False]])
    out = pool(x, torch.zeros(1, 4), keep, torch.zeros(1, dtype=torch.long))
    assert out.item() == pytest.approx(2.0)


def test_date_gradient_after_updates(identity_model):
    model = _model("date")
    model.core.load_state_dict(identity_model.core.state_dict())
    model.train()
    x = torch.randn(1, 4, 10, 64, 64)
    dates = torch.tensor([[100.0, 200, 300, 400]])
    keep = torch.ones(1, 4, dtype=torch.bool)
    anchor = torch.tensor([2])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    target = (torch.rand(1, 1, 64, 64) > 0.5).float()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out = model(x, dates, keep, anchor)["logits"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
        loss.backward()
        opt.step()
    grads = [p.time[0].weight.grad for p in model.pools if getattr(p, "time", None) is not None]
    assert grads and all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0


def test_history_uses_dates_not_indices():
    days = np.array([50, 10, 40, 20, 30], dtype=np.int32)   # 故意打乱
    idx, pos = select_frame_indices(days, anchor=0, mode="history")
    assert days[idx].tolist() == [10, 20, 30, 40, 50]
    assert int(idx[pos]) == 0 and days[int(idx[pos])] == 50
    idx_cut, pos_cut = select_frame_indices(days, anchor=3, mode="history")
    assert days[idx_cut].tolist() == [10, 20]
    assert int(idx_cut[pos_cut]) == 3
    idx_multi, pos_multi = select_frame_indices(days, anchor=3, mode="multi")
    assert days[idx_multi].tolist() == sorted(days.tolist())
    assert int(idx_multi[pos_multi]) == 3


def test_history_respects_cutoff():
    days = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    idx, pos = select_frame_indices(days, anchor=2, mode="history")
    assert idx.tolist() == [0, 1, 2]
    assert pos == 2


def test_config_switches_are_consistent():
    single = load_segformer_config("configs/segformer/v4r1_temporal_control/single.yaml")
    assert single["segformer"]["temporal"] == {"mode": "single", "aggregation": "identity"}
    for name in ("mean", "content", "date"):
        cfg = load_segformer_config(f"configs/segformer/v4r1_temporal_control/{name}.yaml")
        assert cfg["segformer"]["temporal"] == {"mode": "multi", "aggregation": name}
    hist = load_segformer_config("configs/segformer/v4r1_temporal_control/history.yaml")
    assert hist["segformer"]["temporal"]["mode"] == "history"

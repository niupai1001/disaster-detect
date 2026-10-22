"""粗定位引导精分（CF0/CF1）与 U-Net 对照的回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"
CFG = ROOT / "configs" / "segformer" / "coarse_to_fine"

pytestmark = pytest.mark.skipif(not PD.is_dir(), reason="B0 权重未下载")


def _cfg(name):
    from landslide_seg.config import load_segformer_config

    return load_segformer_config(CFG / f"{name}.yaml")["segformer"]


def test_cf1_and_cf0_initialise_to_coarse_output():
    from landslide_seg.models.factory import build_model

    x = torch.randn(1, 10, 128, 128)
    for name in ("cf1", "cf0"):
        model = build_model(_cfg(name), pretrained_dir=PD).eval()
        with torch.no_grad():
            out = model(x, return_delta=True)
        assert float(out["delta_logits"].abs().max()) == 0.0
        assert torch.equal(out["logits"], out["base_logits"])
        assert "coarse_prob" in out


def test_guided_condition_changes_output_after_nonzero_head():
    from landslide_seg.models.factory import build_model

    model = build_model(_cfg("cf1"), pretrained_dir=PD).eval()
    with torch.no_grad():  # 让精分头离开零初始化，模拟训练若干步后
        model.refiner.out.weight.fill_(0.05)
        model.refiner.out.bias.fill_(0.0)
    x = torch.randn(1, 10, 128, 128)
    with torch.no_grad():
        base = model(x)["logits"]
        replaced = model(x, cond_override=torch.zeros_like(base))["logits"]
    assert float((base - replaced).abs().max()) > 1e-6


def test_zero_cond_ignores_condition_override():
    from landslide_seg.models.factory import build_model

    model = build_model(_cfg("cf0"), pretrained_dir=PD).eval()
    with torch.no_grad():
        model.refiner.out.weight.fill_(0.05)
    x = torch.randn(1, 10, 128, 128)
    with torch.no_grad():
        base = model(x)["logits"]
        replaced = model(x, cond_override=torch.ones_like(base))["logits"]
    assert float((base - replaced).abs().max()) == 0.0


def test_unet_shape_and_param_groups():
    from landslide_seg.models.factory import build_model

    model = build_model(_cfg("u0")).eval()
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert tuple(out["logits"].shape) == (2, 1, 128, 128)
    assert out["base_logits"] is out["logits"]
    groups = model.param_groups()
    n = sum(p.numel() for ps in groups.values() for p in ps)
    assert n == sum(p.numel() for p in model.parameters())
    assert len(groups["encoder"]) > 0 and len(groups["decoder"]) > 0


def test_coarse_loss_increases_total_and_rejects_unet():
    from landslide_seg.training.segformer_trainer import _loss

    cfg = {"segformer": {"loss": {"bce_weight": 0.5, "dice_weight": 0.5, "coarse_weight": 0.5}}}
    logits = torch.randn(2, 1, 32, 32)
    base = torch.randn(2, 1, 32, 32)
    mask = (torch.rand(2, 32, 32) > 0.9).float()
    valid = torch.ones(2, 32, 32, dtype=torch.bool)
    total, parts = _loss(cfg, logits, mask, valid, base)
    fine, _ = _loss(
        {"segformer": {"loss": {"bce_weight": 0.5, "dice_weight": 0.5, "coarse_weight": 0.0}}},
        logits, mask, valid, base,
    )
    assert parts["coarse"] is not None
    assert float(total) > float(fine)

    with pytest.raises(RuntimeError, match="U-Net"):
        _loss(cfg, logits, mask, valid, logits)


def test_old_a1_checkpoint_reloads_through_factory():
    ckpt = ROOT / "outputs" / "segformer" / "s2_10band_v3_p20" / "A1" / "1234" / "best.pt"
    if not ckpt.is_file():
        pytest.skip("v3 A1 checkpoint 不存在")
    from landslide_seg.training.segformer_trainer import load_checkpoint

    model, meta = load_checkpoint(ckpt, device="cpu")
    assert getattr(model, "refine", None) == "none"
    with torch.no_grad():
        out = model(torch.randn(1, 10, 128, 128))
    assert tuple(out["logits"].shape) == (1, 1, 128, 128)

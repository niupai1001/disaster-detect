"""单头语义—细节融合（N1 SUM / N2 AFF）回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"
CFG = ROOT / "configs" / "segformer" / "single_head"

pytestmark = pytest.mark.skipif(not PD.is_dir(), reason="B0 权重未下载")


def _cfg(name):
    from landslide_seg.config import load_segformer_config

    return load_segformer_config(CFG / f"{name}.yaml")["segformer"]


def test_single_head_shared_init_and_output_equivalence():
    from landslide_seg.models.factory import build_model
    from landslide_seg.training.segformer_trainer import set_seed

    set_seed(1234, True)
    m1 = build_model(_cfg("n1"), pretrained_dir=PD).eval()
    set_seed(1234, True)
    m2 = build_model(_cfg("n2"), pretrained_dir=PD).eval()
    for key in ("proj_d", "proj_s", "head"):
        a, b = getattr(m1.fusion_head, key).state_dict(), getattr(m2.fusion_head, key).state_dict()
        for k in a:
            assert torch.equal(a[k], b[k]), f"共享张量不一致: {key}.{k}"
    x = torch.randn(1, 10, 128, 128)
    with torch.no_grad():
        o1, o2 = m1(x)["logits"], m2(x)["logits"]
    assert torch.equal(o1, o2)


def test_aff_initial_weight_is_half_and_reduces_to_sum():
    from landslide_seg.models.feature_fusion import AFF

    aff = AFF(64, 4).eval()
    u, v = torch.randn(2, 64, 16, 16), torch.randn(2, 64, 16, 16)
    with torch.no_grad():
        z, w = aff(u, v)
    assert float((w - 0.5).abs().max()) == 0.0
    assert torch.equal(z, u + v)


def test_single_head_output_shape_and_param_groups():
    from landslide_seg.models.factory import build_model

    model = build_model(_cfg("n2"), pretrained_dir=PD).eval()
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        out = model(x, return_attn=True)
    assert tuple(out["logits"].shape) == (2, 1, 128, 128)
    assert out["base_logits"] is None
    assert tuple(out["attn"].shape) == (2, 64, 128, 128)
    groups = model.param_groups()
    n = sum(p.numel() for ps in groups.values() for p in ps)
    assert n == sum(p.numel() for p in model.parameters())
    assert len(groups["encoder"]) and len(groups["decoder"]) and len(groups["new"])


def test_single_head_reload_is_exact():
    from landslide_seg.models.factory import build_model

    model = build_model(_cfg("n2"), pretrained_dir=PD).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    fresh = build_model(_cfg("n2"), pretrained_dir=PD).eval()
    fresh.load_state_dict(state, strict=True)
    x = torch.randn(1, 10, 128, 128)
    with torch.no_grad():
        assert torch.equal(model(x)["logits"], fresh(x)["logits"])


def test_single_head_rejects_coarse_loss():
    from landslide_seg.models.factory import build_model
    from landslide_seg.training import segformer_trainer as tr

    cfg = _cfg("n2")
    cfg = {**cfg, "loss": {**cfg["loss"], "coarse_weight": 0.5}}
    model = build_model(cfg, pretrained_dir=PD).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 10, 128, 128))
    with pytest.raises(RuntimeError, match="base_logits"):
        tr._loss({"segformer": cfg}, out["logits"], torch.zeros(1, 128, 128), torch.ones(1, 128, 128, dtype=torch.bool), out["base_logits"])

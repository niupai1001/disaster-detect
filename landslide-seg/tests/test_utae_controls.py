"""U-TAE 同架构时间控制：均值/内容注意力的开关必须真实改变前向与参数。"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402

spec = importlib.util.spec_from_file_location("research_preflight", ROOT / "scripts/check_research_protocol.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def _model(positional, aggregation):
    return UTAERealDates(in_channels=10, num_classes=1,
                         positional=positional, aggregation=aggregation)


@pytest.mark.parametrize("positional,aggregation", [("dates", "attention"),
                                                    ("none", "attention"),
                                                    ("dates", "mean")])
def test_forward_is_finite(positional, aggregation):
    model = _model(positional, aggregation)
    x = torch.randn(2, 3, 10, 32, 32)
    dates = torch.arange(3).float()[None].repeat(2, 1) + 1800
    with torch.no_grad():
        out = model(x, dates)["logits"]
    assert out.shape == (2, 1, 32, 32)
    assert torch.isfinite(out).all()


def test_switches_remove_the_unused_modules():
    attention = _model("dates", "attention")
    content = _model("none", "attention")
    mean = _model("dates", "mean")
    assert attention.net.temporal_encoder.positional_encoder is not None
    assert content.net.temporal_encoder.positional_encoder is None
    assert mean.net.temporal_encoder.attention_heads is None
    assert mean.count_params()["trainable"] < attention.count_params()["trainable"]


def test_invalid_switch_rejected():
    with pytest.raises(ValueError):
        _model("dates", "max")


def test_preflight_records_knobs_and_epoch_warning():
    result = preflight.audit_config("configs/utae/v4r1/mean.yaml")
    assert result["model_knobs"]["aggregation"] == "mean"
    assert result["model_knobs"]["positional"] == "none"
    assert result["model_knobs"]["epochs"] == 30
    assert any("不是等轮数" in w for w in result["warnings"])

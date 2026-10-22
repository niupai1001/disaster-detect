"""SegFormer 配置 schema 测试：_base 合并、--set 覆盖、backbone 校验。"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"


def test_base_merge_and_group_override():
    from landslide_seg.config import load_segformer_config

    cfg = load_segformer_config(ROOT / "configs" / "segformer" / "a1.yaml")["segformer"]
    assert cfg["experiment"] == "A1"
    assert cfg["model"]["adapter"] == "direct10"
    assert cfg["model"]["stem"] is True
    # base.yaml 的公共项必须被继承
    assert cfg["data"]["version"] == "s2_10band_v1"
    assert cfg["optim"]["encoder_lr"] > 0
    assert cfg["train"]["batch_size"] >= 1
    # 四组公共预算字段来自 v2 修正
    assert cfg["data"]["post_event_window_days"] == 120
    assert cfg["data"]["preload"] is True


def test_set_override_applies():
    from landslide_seg.config import load_segformer_config

    cfg = load_segformer_config(
        ROOT / "configs" / "segformer" / "a0.yaml",
        ["segformer.train.epochs=7", "segformer.model.stem=true"],
    )["segformer"]
    assert cfg["train"]["epochs"] == 7
    assert cfg["train"]["batch_size"] == 64  # 未被覆盖的保持 base
    assert cfg["model"]["stem"] is True


def test_defaults_have_no_legacy_fields():
    from landslide_seg.config import DEFAULTS_SEGFORMER

    data = DEFAULTS_SEGFORMER["segformer"]["data"]
    # 旧路线的 boa_offset 不该混进新 schema
    assert "boa_offset_dn" not in data
    assert data["test_event"] == "italy"
    assert data["val_event"] == "hiroshima"


@pytest.mark.skipif(not PD.is_dir(), reason="B0 权重未下载")
def test_backbone_mismatch_is_rejected():
    from landslide_seg.models.segformer import SegFormerSeg

    with pytest.raises(ValueError, match="B2|b2"):
        SegFormerSeg(backbone="b2", adapter="direct10", stem=False, pretrained_dir=str(PD))


def test_unknown_backbone_is_rejected():
    from landslide_seg.models.segformer import SegFormerSeg

    with pytest.raises(ValueError, match="backbone"):
        SegFormerSeg(backbone="b9", adapter="direct10", stem=False, pretrained_dir=str(PD))

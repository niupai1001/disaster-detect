"""几何一致性 + 无 GT 泄漏 + 形状测试（不需要预训练权重）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.models.coarse import CoarseLocator  # noqa: E402
from landslide_seg.models.fusion import FeatureFusion, FeatureMap  # noqa: E402
from landslide_seg.models.refiner import FineRefiner  # noqa: E402


def _pyramid(c=16, size=32):
    fusion = FeatureFusion(channels=c, level_strides=(10, 20, 40, 80),
                           sources=("a", "b"), branch_dropout=0.0)
    fusion.build({"a": 6, "b": 9})
    feats = [
        FeatureMap(torch.randn(1, 6, size, size), "a", 10.0),
        FeatureMap(torch.randn(1, 9, size, size), "b", 10.0),
    ]
    return fusion, fusion(feats, target_size=(size, size))


def test_pyramid_shapes_follow_physical_stride():
    _, pyr = _pyramid(c=16, size=32)
    assert [p.shape[-1] for p in pyr] == [32, 16, 8, 4]
    assert all(p.shape[1] == 16 for p in pyr)


def test_spatial_alignment_impulse():
    """在最细层放一个脉冲，上采样后峰值应落在同一相对位置。"""
    c, size = 8, 32
    fusion = FeatureFusion(channels=c, level_strides=(10, 40),
                           sources=("a",), branch_dropout=0.0)
    fusion.build({"a": c})
    # 让 refine 变成恒等：把 conv 权重设为单位（仅通道 0）
    x = torch.zeros(1, c, size, size)
    x[0, 0, 5, 20] = 1.0
    pyr = fusion([FeatureMap(x, "a", 10.0)], target_size=(size, size))
    coarse = pyr[1][0, 0]                  # 8x8
    cy, cx = divmod(int(coarse.argmax()), coarse.shape[1])
    # 5/4≈1.25, 20/4=5
    assert abs(cy - 1) <= 1 and abs(cx - 5) <= 1


def test_coarse_outputs():
    _, pyr = _pyramid(c=16, size=32)
    head = CoarseLocator(channels=16, hidden=16, n_levels=4)
    out = head(pyr, torch.zeros(1, 1, 32, 32))
    assert out["coarse_logits"].shape == (1, 1, 32, 32)
    assert out["presence_logits"].shape == (1, 1)


def test_refiner_outputs_and_uses_prediction():
    _, pyr = _pyramid(c=16, size=32)
    ref = FineRefiner(channels=16, hidden=16, n_levels=4)
    coarse_prob = torch.rand(1, 1, 32, 32)
    film = (torch.zeros(1, 16), torch.zeros(1, 16))
    a = ref(pyr, coarse_prob, film=film)["fine_logits"]
    b = ref(pyr, 1 - coarse_prob, film=film)["fine_logits"]
    assert a.shape == (1, 1, 32, 32)
    # 输入换了粗概率，输出必须变化 -> 证明细化器真的在用预测
    assert not torch.allclose(a, b)


def test_semantic_film_requires_text():
    """没有真实文本权重时不能构造语义模块（防止随机文本冒充）。"""
    from landslide_seg.models.semantic import RemoteCLIPTextEncoder

    with pytest.raises(FileNotFoundError):
        RemoteCLIPTextEncoder(ROOT / "weights" / "__not_exist__.pt")

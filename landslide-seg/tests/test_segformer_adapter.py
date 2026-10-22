"""输入适配测试：direct10 的精确权重拷贝、与三通道模型的初始化等价、Stem 初始 ΔL=0。

等价性容差说明：10 通道卷积与 3 通道卷积在 FP 上的累加顺序不同，
解码特征只能要求 ``allclose(atol=1e-5)``；权重拷贝本身是严格相等。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"

pytestmark = pytest.mark.skipif(not PD.is_dir(), reason="B0 权重未下载")


@pytest.fixture(scope="module")
def models():
    from transformers import SegformerForSemanticSegmentation

    from landslide_seg.models.segformer import SegFormerSeg

    torch.manual_seed(0)
    base = SegformerForSemanticSegmentation.from_pretrained(str(PD)).eval()
    direct = SegFormerSeg(adapter="direct10", stem=False, pretrained_dir=str(PD)).eval()
    project = SegFormerSeg(adapter="project10to3", stem=False, pretrained_dir=str(PD)).eval()
    return base, direct, project


def test_direct10_weight_copy_is_exact(models):
    base, direct, _ = models
    old = base.segformer.encoder.patch_embeddings[0].proj
    new = direct.model.segformer.encoder.patch_embeddings[0].proj
    assert tuple(new.weight.shape) == (32, 10, 7, 7)
    # R/G/B → B04/B03/B02
    assert torch.equal(new.weight[:, 2], old.weight[:, 0])
    assert torch.equal(new.weight[:, 1], old.weight[:, 1])
    assert torch.equal(new.weight[:, 0], old.weight[:, 2])
    zeros = [i for i in range(10) if i not in (0, 1, 2)]
    assert float(new.weight[:, zeros].detach().abs().max()) == 0.0
    if old.bias is not None:
        assert torch.equal(new.bias, old.bias)
    # 保存/重载时配置应与权重一致
    assert direct.model.config.num_channels == 10


def test_direct10_matches_rgb_model_at_init(models):
    base, direct, _ = models
    torch.manual_seed(1)
    x10 = torch.randn(2, 10, 128, 128)
    rgb = x10[:, [2, 1, 0]].clone()

    captured = {}
    handle = base.decode_head.classifier.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("d", args[0])
    )
    with torch.no_grad():
        base(pixel_values=rgb)
    handle.remove()
    ref = captured["d"]

    with torch.no_grad():
        direct(x10)
    got = direct._decoder_feat
    assert got is not None and got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4)
    # 明确记录：这不是逐位相等，而是容差内相等
    assert float((got - ref).abs().max()) < 1e-4


def test_project10to3_matches_rgb_model_at_init(models):
    base, _, project = models
    torch.manual_seed(2)
    x10 = torch.randn(2, 10, 128, 128)
    rgb = x10[:, [2, 1, 0]].clone()

    captured = {}
    handle = base.decode_head.classifier.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("d", args[0])
    )
    with torch.no_grad():
        base(pixel_values=rgb)
    handle.remove()

    with torch.no_grad():
        project(x10)
    got = project._decoder_feat
    assert torch.allclose(got, captured["d"], atol=1e-6, rtol=1e-5)


def test_stem_delta_is_zero_at_init():
    from landslide_seg.models.segformer import SegFormerSeg

    model = SegFormerSeg(adapter="direct10", stem=True, pretrained_dir=str(PD)).eval()
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        out = model(x, return_delta=True)
    assert out["delta_logits"] is not None
    assert float(out["delta_logits"].abs().max()) == 0.0
    assert torch.equal(out["logits"], out["base_logits"])

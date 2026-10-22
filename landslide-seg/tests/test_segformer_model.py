"""模型级测试：尺寸、Stem 开关、梯度通路、参数分组、保存→全新实例重载。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "weights" / "segformer" / "segformer-b0-finetuned-ade-512-512"

pytestmark = pytest.mark.skipif(not PD.is_dir(), reason="B0 权重未下载")


@pytest.fixture(scope="module")
def model():
    from landslide_seg.models.segformer import SegFormerSeg

    torch.manual_seed(0)
    return SegFormerSeg(adapter="direct10", stem=True, pretrained_dir=str(PD))


def test_forward_shape_and_outputs(model):
    model.eval()
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert tuple(out["logits"].shape) == (2, 1, 128, 128)
    assert tuple(out["base_logits"].shape) == (2, 1, 128, 128)
    # 输入尺寸可以变化（全卷积）
    with torch.no_grad():
        out2 = model(torch.randn(1, 10, 64, 96))
    assert tuple(out2["logits"].shape) == (1, 1, 64, 96)


def test_bad_channel_count_raises(model):
    with pytest.raises(ValueError, match="期望"):
        model(torch.randn(1, 3, 128, 128))


def test_stem_gradient_flow(model):
    """第 1 步 fusion 输出层有梯度、Stem/body 为 0；对末层走一步后全部非零。"""
    model.train()
    x = torch.randn(2, 10, 128, 128)
    params = dict(model.named_parameters())
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    model.zero_grad(set_to_none=True)
    model(x)["logits"].mean().backward()
    assert params["fusion.out.weight"].grad is not None
    assert float(params["fusion.out.weight"].grad.abs().max()) > 0
    assert float(params["stem.net.0.weight"].grad.abs().max()) == 0.0
    assert float(params["fusion.body.0.weight"].grad.abs().max()) == 0.0

    opt.step()  # 让末层权重离开 0，梯度才会回传到 body/stem
    model.zero_grad(set_to_none=True)
    model(x)["logits"].mean().backward()
    assert float(params["stem.net.0.weight"].grad.abs().max()) > 0
    assert float(params["fusion.body.0.weight"].grad.abs().max()) > 0


def test_first_conv_new_channels_get_gradient(model):
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randn(2, 10, 128, 128)
    model(x)["logits"].mean().backward()
    w = dict(model.named_parameters())["model.segformer.encoder.patch_embeddings.0.proj.weight"]
    assert float(w.grad[3:].abs().max()) > 0


def test_param_groups_cover_all_trainable(model):
    groups = model.param_groups()
    n_group = sum(p.numel() for ps in groups.values() for p in ps)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_group == n_trainable
    assert len(groups["encoder"]) > 0 and len(groups["decoder"]) > 0
    assert len(groups["new"]) > 0  # direct10 + stem


def test_save_reload_to_fresh_instance(model):
    from landslide_seg.models.segformer import SegFormerSeg

    model.eval()
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        ref = model(x)["logits"]
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    fresh = SegFormerSeg(adapter="direct10", stem=True, pretrained_dir=str(PD)).eval()
    fresh.load_state_dict(state, strict=True)
    with torch.no_grad():
        got = fresh(x)["logits"]
    assert float((got - ref).abs().max()) < 1e-6


def test_eval_is_deterministic(model):
    model.eval()
    x = torch.randn(2, 10, 128, 128)
    with torch.no_grad():
        a = model(x)["logits"]
        b = model(x)["logits"]
    assert torch.equal(a, b)


def test_b2_dimension_wiring(tmp_path):
    """P7 工程项：B2 规格（解码器 768 维）能完成 构造→前向→反向。

    这里用随机初始化的 B2 结构权重，只验证**维度接线**，不代表任何预训练能力；
    B2 官方 ADE 权重尚未下载，许可证也未确认（见 docs/segformer_environment_lock.md）。
    """
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    from landslide_seg.models.segformer import SegFormerSeg

    cfg = SegformerConfig(
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 4, 6, 3],
        decoder_hidden_size=768,
        num_labels=1,
        num_channels=3,
    )
    base = SegformerForSemanticSegmentation(cfg)
    out_dir = tmp_path / "fake-b2"
    base.save_pretrained(str(out_dir))

    model = SegFormerSeg(
        adapter="direct10", backbone="b2", stem=True, pretrained_dir=str(out_dir)
    )
    model.train()
    x = torch.randn(1, 10, 128, 128)
    logits = model(x)["logits"]
    assert tuple(logits.shape) == (1, 1, 128, 128)
    logits.mean().backward()
    params = dict(model.named_parameters())
    assert params["stem.net.0.weight"].grad is not None
    assert model.count_params()["decoder"] > 0

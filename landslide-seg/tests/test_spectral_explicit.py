"""新显式光谱调用与历史钩子路径保持权重、输出及梯度兼容。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
import torch
from landslide_seg.models.segformer import SegFormerSeg
from landslide_seg.models.spectral_fusion import SpectralFusionSeg

PD = Path(__file__).resolve().parents[1] / "weights/segformer/segformer-b0-finetuned-ade-512-512"


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def pair(fusion="gated"):
    old = SegFormerSeg(pretrained_dir=str(PD), stem=True)
    new = SegFormerSeg(pretrained_dir=str(PD), stem=True, capture_decoder=False)
    legacy = SpectralFusionSeg(old, [2000.] * 10, [500.] * 10, feature="identity", fusion=fusion)
    explicit = SpectralFusionSeg(new, [2000.] * 10, [500.] * 10, feature="identity", fusion=fusion,
                                 implementation="explicit")
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    return legacy.eval(), explicit.eval()


@pytest.mark.parametrize("fusion", ["gated", "additive", "fixed_gate", "lowrank"])
def test_nonzero_branch_matches_legacy_with_same_checkpoint(fusion):
    old, new = pair(fusion)
    with torch.no_grad():
        old.branch.alpha.fill_(.2)
        # 细节头也使用非零参数，确保兼容融合前特征的历史语义。
        for p in old.visual.fusion.parameters():
            p.normal_(std=.01)
        new.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(2, 10, 64, 64)
        a, b = old(x, return_features=True, return_delta=True), new(x, return_features=True, return_delta=True)
    for key in ("logits", "base_logits", "decoder_feat", "delta_logits"):
        assert torch.allclose(a[key], b[key], atol=1e-6, rtol=1e-5), key
    assert not new.visual.model.decode_head.classifier._forward_pre_hooks
    assert new._S is None and new._last == {}


def test_zero_alpha_equivalence_and_gradients_after_updates():
    _, model = pair()
    x = torch.randn(2, 10, 64, 64)
    plain = SegFormerSeg(pretrained_dir=str(PD), stem=True).eval()
    plain.load_state_dict(model.visual.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.allclose(model(x)["logits"], plain(x)["logits"], atol=1e-6, rtol=1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    for step in range(3):
        opt.zero_grad()
        out = model(x)
        out["logits"].square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if step == 0:
            assert model.branch.alpha.grad.abs().sum() > 0
            assert model.branch.proj.weight.grad.abs().sum() == 0
        opt.step()
    assert model.branch.proj.weight.grad.abs().sum() > 0
    assert model.branch.gate.weight.grad.abs().sum() > 0


def test_parallel_forward_has_no_shared_feature_cache():
    _, model = pair()
    with torch.no_grad():
        model.branch.alpha.fill_(.2)
    xs = [torch.randn(1, 10, 64, 64) for _ in range(2)]
    def forward(x):
        with torch.no_grad():
            return model(x)["logits"]
    expected = [forward(x) for x in xs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(forward, xs))
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(actual, expected))
    assert model.visual._decoder_feat is None and model._S is None

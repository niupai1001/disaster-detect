"""S2 真波长条件映射与同参数量对照：只差“矩阵是否由波长生成”。"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.models.spectral_ladder import SpectralLadder  # noqa: E402

STATS = {"mean": [0.0] * 10, "std": [1.0] * 10}


def test_same_parameter_budget_and_identity_start():
    conditional = SpectralLadder("S2", "conditional", STATS)
    free = SpectralLadder("S2", "free", STATS)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(conditional) == count(free) == 110
    x = torch.randn(2, 10, 8, 8)
    for module in (conditional, free):
        assert torch.allclose(module(x), x, atol=1e-6)


def test_only_conditional_depends_on_wavelengths():
    conditional = SpectralLadder("S2", "conditional", STATS)
    free = SpectralLadder("S2", "free", STATS)
    torch.nn.init.normal_(conditional.wave_to_row.weight, std=0.1)
    torch.nn.init.normal_(conditional.wave_to_row.bias, std=0.1)
    x = torch.randn(1, 10, 4, 4)
    before = conditional(x).clone()
    conditional.waves.copy_(conditional.waves * 1.1)
    assert not torch.allclose(before, conditional(x))
    before = free(x).clone()
    free.waves.copy_(free.waves * 1.1)
    assert torch.allclose(before, free(x))


def test_mechanism_names_are_explicit():
    assert SpectralLadder("S2", "conditional", STATS).report()["mechanism"] == \
        "wavelength_conditional_input_mix"
    assert SpectralLadder("S2", "free", STATS).report()["mechanism"] == "free_channel_input_mix"
    # 历史静态偏置仍是独立变体，不冒充这次的映射
    assert SpectralLadder("S2", "treatment", STATS).report()["mechanism"] == "static_wavelength_bias"

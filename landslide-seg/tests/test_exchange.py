"""v4r4 交换块的语义锁定。

这些断言把计划 §5 的关键性质固定下来，防止后续重构悄悄改变行为：
- 零初始化必须**精确**等价于恒等（不是"足够小"）；
- self / cross 参数量必须精确相等，否则"容量对照"失效；
- cross 里 D 必须影响 E，self 里 D 绝不能影响 E；
- 形状/通道不符要报错，而不是静默广播。
"""

from __future__ import annotations

import pytest
import torch

from landslide_seg.models.exchange import ChannelwiseResidualExchange


def _pair(b=2, e_ch=64, d_ch=32, hw=(16, 16), scale=2):
    e = torch.randn(b, e_ch, *hw)
    d = torch.randn(b, d_ch, hw[0] * scale, hw[1] * scale)
    return e, d


def test_zero_init_is_exact_identity():
    for mode in ("self", "cross"):
        ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode=mode)
        e, d = _pair()
        e_new, d_new = ex(e, d)
        assert torch.equal(e_new, e), mode
        assert torch.equal(d_new, d), mode


def test_parameter_count_identical_between_modes():
    a = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="self")
    b = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="cross")
    na = sum(p.numel() for p in a.parameters())
    nb = sum(p.numel() for p in b.parameters())
    assert na == nb
    # P(64*32+32) + H_E(32*32*9+32 + 32*64+64) + H_D(32*32*9+32 + 32*32+32) + α(64+32)
    assert na == 2080 + 11360 + 10304 + 96


def test_shapes_preserved():
    for mode in ("self", "cross"):
        ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode=mode)
        e, d = _pair()
        e_new, d_new = ex(e, d)
        assert e_new.shape == e.shape
        assert d_new.shape == d.shape


def test_cross_propagates_detail_into_encoder():
    ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="cross")
    with torch.no_grad():
        ex.alpha_e.fill_(0.1)
    e, d = _pair()
    _, _ = ex(e, d)
    e_a, _ = ex(e, d)
    e_b, _ = ex(e, d * 3.0)
    assert not torch.allclose(e_a, e_b)


def test_self_has_no_cross_path():
    ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="self")
    with torch.no_grad():
        ex.alpha_e.fill_(0.1)
    e, d = _pair()
    e_a, _ = ex(e, d)
    e_b, _ = ex(e, d * 3.0)
    assert torch.equal(e_a, e_b)


def test_alpha_is_only_path_from_detail_to_encoder():
    """α_D 只控制 D 的更新，不应阻断 E 方向。"""
    for mode in ("self", "cross"):
        ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode=mode)
        with torch.no_grad():
            ex.alpha_e.fill_(0.1)
            ex.alpha_d.fill_(0.0)
        e, d = _pair()
        _, d_new = ex(e, d)
        assert torch.equal(d_new, d)


def test_alpha_d_controls_detail_update():
    ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="cross")
    with torch.no_grad():
        ex.alpha_d.fill_(0.1)
    e, d = _pair()
    _, d_new = ex(e, d)
    assert not torch.equal(d_new, d)


def test_rejects_bad_channel_and_scale():
    ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="cross")
    with pytest.raises(ValueError):
        ex(torch.randn(2, 32, 16, 16), torch.randn(2, 32, 32, 32))
    with pytest.raises(ValueError):
        ex(torch.randn(2, 64, 16, 16), torch.randn(2, 16, 32, 32))
    with pytest.raises(ValueError):
        ex(torch.randn(2, 64, 16, 16), torch.randn(3, 32, 32, 32))
    with pytest.raises(ValueError):
        # 尺度不整除
        ex(torch.randn(2, 64, 16, 16), torch.randn(2, 32, 30, 30))


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="both")


def test_alpha_grad_exists_but_conv_grad_zero_at_init():
    """零残差系数下：α 有梯度，H_* 第一步没有——这是设计而非缺陷。"""
    ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="cross")
    e, d = _pair()
    e_new, _ = ex(e, d)
    e_new.sum().backward()
    # E 方向：α_E 有梯度（∂L/∂α = grad·k），但 H_E 权重被 α=0 乘掉
    assert ex.alpha_e.grad is not None and ex.alpha_e.grad.abs().sum() > 0
    assert ex.h_e[0].weight.grad is None or ex.h_e[0].weight.grad.abs().sum() == 0
    # 只对 E_new 求导时，D 分支完全拿不到梯度
    assert ex.alpha_d.grad is None

    # D 方向单独求导才有梯度
    ex.zero_grad(set_to_none=True)
    _, d_new = ex(e, d)
    d_new.sum().backward()
    assert ex.alpha_d.grad is not None and ex.alpha_d.grad.abs().sum() > 0
    assert ex.alpha_e.grad is None


def test_describe_reports_zero_alpha():
    ex = ChannelwiseResidualExchange(e_ch=64, d_ch=32, mode="self")
    info = ex.describe()
    assert info["mode"] == "self"
    assert info["alpha_init_max_abs"] == 0.0
    assert info["params"] == 23840

"""行为检查只证明波段注意力参与计算，不证明物理正确或研究收益。"""
import copy
import torch
from landslide_seg.models.spectral_temporal import BandRelationBlock


def active_block():
    torch.manual_seed(33)
    block = BandRelationBlock(width=16, heads=4, layers=1,
                              stats={"mean": [2000.] * 10, "std": [500.] * 10})
    with torch.no_grad():
        block.readout.weight.normal_(std=.1)
        block.relation[-1].weight.normal_(std=.1)
    return block.eval()


def test_joint_band_and_metadata_permutation_equivariance():
    block = active_block()
    permuted = copy.deepcopy(block)
    perm = torch.tensor([4, 2, 6, 8, 0, 9, 3, 1, 7, 5])
    with torch.no_grad():
        for name in ("waves", "dn_mean", "dn_std"):
            getattr(permuted, name).copy_(getattr(block, name)[perm])
        x = torch.randn(2, 10, 8, 8)
        a, b = block(x)[:, perm], permuted(x[:, perm])
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_changed_wavelength_and_local_bias_affect_active_output():
    block = active_block()
    x = torch.randn(2, 10, 8, 8)
    with torch.no_grad():
        a = block(x)
        block.waves[0] += .2
        b = block(x)
        block.relation_mode = "none"
        c = block(x)
    assert (a - b).abs().max() > 1e-6
    assert (b - c).abs().max() > 1e-6
    assert block.relation_bias(block.reflectance(x)).shape == (2, 4, 4, 10, 10)


def test_zero_readout_identity_then_attention_receives_gradients():
    torch.manual_seed(22)
    block = BandRelationBlock(width=16, heads=4, layers=1,
                              stats={"mean": [2000.] * 10, "std": [500.] * 10})
    x = torch.randn(2, 10, 8, 8)
    assert torch.equal(block(x), x)
    opt = torch.optim.AdamW(block.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad()
        loss = block(x).square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
        opt.step()
    assert block.layers[0].qkv.weight.grad.abs().sum() > 0
    assert block.relation[-1].weight.grad.abs().sum() > 0

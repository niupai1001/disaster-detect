import torch

from src.models.unified_encoder import UnifiedEncoder


def _batch(b, c, h=64, w=64):
    x = torch.rand(b, c, h, w)
    centre = torch.linspace(450, 2200, c)
    gsd = torch.full((c,), 10.0)
    return x, centre, gsd


def test_same_model_accepts_different_band_counts():
    model = UnifiedEncoder(width=8).eval()
    for c in (6, 7, 10):
        x, centre, gsd = _batch(2, c)
        with torch.no_grad():
            out = model(x, centre, gsd)
        assert out["global"].shape == (2, model.backbone.out_channels["s32"])
        assert out["s4"].shape[-1] == 64 // 4


def test_reordering_bands_with_descriptors_keeps_output():
    """Band order must not matter once descriptors travel with the bands."""
    torch.manual_seed(0)
    model = UnifiedEncoder(width=8).eval()
    x, centre, gsd = _batch(1, 10)
    perm = torch.tensor([9, 3, 0, 7, 1, 8, 2, 6, 4, 5])
    with torch.no_grad():
        a = model(x, centre, gsd)["global"]
        b = model(x[:, perm], centre[perm], gsd[perm])["global"]
    assert torch.allclose(a, b, atol=1e-4)


def test_band_mask_changes_only_dropped_bands():
    model = UnifiedEncoder(width=8).eval()
    x, centre, gsd = _batch(1, 6)
    mask = torch.tensor([True, True, True, False, False, False])
    with torch.no_grad():
        full = model(x, centre, gsd)["global"]
        masked = model(x, centre, gsd, band_mask=mask.unsqueeze(0))["global"]
    assert full.shape == masked.shape
    assert not torch.allclose(full, masked)


def test_output_is_dense_and_pyramidal():
    model = UnifiedEncoder(width=8).eval()
    x, centre, gsd = _batch(1, 6, 128, 128)
    with torch.no_grad():
        out = model(x, centre, gsd)
    strides = {"s4": 4, "s8": 8, "s16": 16, "s32": 32}
    for key, stride in strides.items():
        assert out[key].shape[-1] == 128 // stride

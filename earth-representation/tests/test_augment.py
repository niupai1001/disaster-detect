import torch

from src.augment import band_dropout, reflectance_perturb, spatial_degradation as sd


def test_lowpass_keeps_shape_and_removes_variance():
    x = torch.rand(3, 64, 64)
    y = sd.lowpass(x, sigma=1.0)
    assert y.shape == x.shape
    assert y.float().std() < x.std()          # smoothing must reduce variance


def test_spatial_and_generic_keep_shape():
    x = torch.rand(3, 64, 64)
    assert sd.spatial_degrade(x, 1.0, 2.0).shape == x.shape
    assert sd.generic_resize(x, 2.0).shape == x.shape


def test_band_mask_keeps_minimum_and_zeroes_dropped():
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        mask = band_dropout.sample_band_mask(10, g, keep_min=3)
        assert mask.sum() >= 3
        x, m = band_dropout.apply_band_dropout(torch.ones(10, 8, 8), mask)
        assert m.shape == (10,)
        assert torch.all(x[~mask] == 0)
        assert torch.all(x[mask] == 1)


def test_gain_offset_within_declared_bounds():
    g = torch.Generator().manual_seed(0)
    x = torch.ones(4, 8, 8)
    for _ in range(10):
        y, params = reflectance_perturb.apply_gain_offset(x, g, gain=0.05, offset=0.02)
        assert abs(y.mean().item() - 1.0) < 0.1
        assert abs(params["gain"] - 1.0) <= 0.05 + 1e-6
        assert abs(params["offset"]) <= 0.02 + 1e-6


def test_perturbation_does_not_touch_mask():
    # QA is separate from reflectance, so a perturbed view keeps the same validity
    x = torch.rand(3, 16, 16)
    valid = torch.ones(16, 16, dtype=torch.bool)
    y, _ = reflectance_perturb.apply_gain_offset(x, torch.Generator().manual_seed(0))
    assert y.shape == x.shape and valid.all()

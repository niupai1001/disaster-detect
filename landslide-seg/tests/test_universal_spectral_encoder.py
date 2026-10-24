import pytest
import torch
from landslide_seg.data.optical_observation import OpticalObservation, collate_optical
from landslide_seg.models.universal_spectral_encoder import UniversalSpectralEncoder


def obs(c):
    return OpticalObservation(torch.randn(c, 7, 9), tuple(map(str, range(c))),
        torch.ones(c, 3), torch.ones(c, 3, dtype=torch.bool),
        torch.ones(c, dtype=torch.bool), torch.ones(c, 7, 9, dtype=torch.bool), {})


@pytest.mark.parametrize('c', [3, 4, 6, 10, 13])
def test_shapes_gradients_permutation_padding(c):
    torch.manual_seed(12)
    model = UniversalSpectralEncoder().eval()
    sample = obs(c)
    out = model(collate_optical([sample]))['features']
    assert out.shape == (1, 32, 7, 9)
    reverse = model(collate_optical([sample.subset(sample.band_ids[::-1])]))['features']
    mixed = model(collate_optical([sample, obs(13)]))['features'][:1]
    torch.testing.assert_close(out, reverse, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out, mixed, atol=1e-5, rtol=1e-4)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_local_missing_and_invalid_intervention():
    model = UniversalSpectralEncoder().eval()
    sample = obs(3)
    sample.pixel_valid[:, :4, :4] = False
    batch = collate_optical([sample])
    out = model(batch)['features']
    assert torch.isfinite(out).all() and not out[:, :, :4, :4].any()
    batch['x'][:, :, :4, :4] = float('nan')
    torch.testing.assert_close(out, model(batch)['features'])
    batch['pixel_valid'][:] = False
    with pytest.raises(ValueError):
        model(batch)


def test_reload_and_pulse():
    model = UniversalSpectralEncoder().eval()
    clone = UniversalSpectralEncoder().eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    batch = collate_optical([obs(3)])
    before = model(batch)['features']
    torch.testing.assert_close(before, clone(batch)['features'])
    batch['x'][0, 0, 3, 3] += 1
    assert not torch.equal(before[:, :, 3, 3], model(batch)['features'][:, :, 3, 3])

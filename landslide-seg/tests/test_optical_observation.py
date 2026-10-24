import pytest
import torch
from landslide_seg.data.optical_observation import (
    OpticalObservation, collate_optical, physical_reflectance, training_subset,
)


def observation(c):
    return OpticalObservation(torch.randn(c, 7, 9), tuple(f'b{i}' for i in range(c)),
        torch.ones(c, 3), torch.ones(c, 3, dtype=torch.bool),
        torch.ones(c, dtype=torch.bool), torch.ones(c, 7, 9, dtype=torch.bool), {})


@pytest.mark.parametrize('c', [3, 4, 6, 10, 13])
def test_padding_and_reordering(c):
    obs = observation(c)
    batch = collate_optical([obs, observation(13)])
    torch.testing.assert_close(batch['x'][0, :c], obs.x)
    assert not batch['band_valid'][0, c:].any()
    assert not batch['pixel_valid'][0, c:].any()
    reverse = obs.subset(obs.band_ids[::-1])
    torch.testing.assert_close(reverse.x, obs.x.flip(0))


def test_invalid_values_and_empty():
    obs = observation(3)
    obs.x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        obs.validate()
    obs.pixel_valid[0, 0, 0] = False
    assert torch.isfinite(collate_optical([obs])['x']).all()
    obs.band_valid[:] = False
    with pytest.raises(ValueError):
        collate_optical([obs])


def test_metadata_validation():
    obs = observation(3)
    obs.known_mask[0, 1] = False
    with pytest.raises(ValueError):
        obs.validate()
    obs.band_metadata[0, 1] = 0
    obs.validate()
    obs.band_metadata[0, 0] = 490
    with pytest.raises(ValueError):
        obs.validate()


def test_landsat_transform_and_sampling():
    rho, valid = physical_reflectance(torch.tensor([18639., float('nan')]),
                                     .0000275, -.2, torch.ones(2, dtype=torch.bool))
    assert rho[0].item() == pytest.approx(.3125725)
    assert not valid[1] and rho[1] == 0
    choices = [training_subset(1234, 'sample', i) for i in range(6000)]
    assert choices == [training_subset(1234, 'sample', i) for i in range(6000)]
    assert set(choices) == {'rgb', 'rgb_nir', 'six', 'full'}
    assert .47 < choices.count('full') / len(choices) < .53


def test_augmented_geographic_grid_matches_pixels():
    import numpy as np
    from landslide_seg.data.segformer_dataset import _augment
    from landslide_seg.data.use_dataset import augmentation_grid_map
    source=np.arange(15).reshape(3,5)
    for seed in range(16):
        augmented,_,_=_augment(source[None],source,np.ones_like(source),np.random.default_rng(seed))
        mapping,_=augmentation_grid_map(3,5,seed,True)
        for row,column in np.ndindex(augmented.shape[1:]):
            old=mapping@np.array([column+.5,row+.5,1])
            assert augmented[0,row,column]==source[round(old[1]-.5),round(old[0]-.5)]

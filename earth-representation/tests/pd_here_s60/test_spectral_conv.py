from dataclasses import replace

import torch

from encoder_v4.data import Attribute, Band, Observation, Physics
from encoder_v4.geometry import Grid
from pd_here_s60.whole_scene import tiles
from pd_here_s60.whole_scene_spectral_conv import WholeSceneSpectralConv, pack_spectral_conv_scene


def test_spectral_conv_native_groups_tiling_and_isolation():
    torch.set_num_threads(2)

    def band(key, center, spacing, shift, shape):
        grid = Grid(shape, (spacing, 0., 500000.+shift, 0., -spacing, 3000000.), 'EPSG:32632')
        x = torch.rand(shape)
        qa = torch.ones_like(x, dtype=torch.bool)
        qa[::5, ::7] = False
        physics = Physics(center=Attribute('observed', 'um', 'test', (center,)))
        return Band(key, x, qa, grid, physics)

    observation = Observation('test', 'instrument', (
        band('blue', .49, 10., 0., (13, 520)),
        band('red', .66, 10., 0., (13, 520)),
        band('nir', .86, 22., 5., (6, 236)),
    ))
    scene = pack_spectral_conv_scene(observation, 1., 'test')
    assert [g.bands for g in scene.metadata.groups] == [(0, 1), (2,)]
    keep = torch.ones_like(scene.qa)
    keep[0, ::2] = False
    band_mask = torch.tensor([True, False, True])
    model = WholeSceneSpectralConv({scene.layout: [0.] * 3})
    head = model.projections['0']
    torch.nn.init.normal_(head.output.weight, std=.02)

    split = list(tiles(scene, keep))
    assert len(split) == 2
    assert split[1][0].metadata.groups[1].x.shape[0] == split[1][0].grid[1]
    for tile, visible, _ in split:
        values = tile.values.requires_grad_()
        result = model(tile, visible, band_mask)
        assert result.shape == tile.values[tile.qa.any(-1)].shape
        active = tile.qa & visible & band_mask[head.band_slots]
        changed = replace(tile, values=torch.where(active, values, float('nan')))
        torch.testing.assert_close(model(changed, visible, band_mask), result, atol=0, rtol=0)
        gradient, = torch.autograd.grad(result.square().mean(), values)
        assert gradient[active].abs().sum() > 0
        assert not gradient[~active].any()

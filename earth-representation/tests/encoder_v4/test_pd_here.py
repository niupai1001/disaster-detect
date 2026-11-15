from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.encoder_v4.data import Attribute, Band, Observation, Physics, scene, task_view
from src.encoder_v4.experiment import native_metrics
from src.encoder_v4.geometry import Grid, build_work_grid
from src.encoder_v4.model import Attention, Config, Decoder, GroupBlock, Pretraining, merge, objective


@pytest.fixture(autouse=True, scope='module')
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def sample():
    torch.manual_seed(17)
    grid = Grid((17, 19), (10., 0., 500000., 0., -10., 4200000.), 'EPSG:32633')
    coarse = Grid((9, 10), (20., 0., 499995., 0., -20., 4200005.), grid.crs)
    bands = []
    for i, native in enumerate((grid, coarse, grid, coarse)):
        valid = torch.ones(native.shape, dtype=torch.bool)
        valid[-2:, -2:] = False
        attributes = Physics(center=Attribute('observed', 'um', 'fixture', (.4 + .15 * i,)),
                             gsd=Attribute('observed', 'm', 'affine', native.spacing))
        bands.append(Band(str(i), torch.rand(native.shape).requires_grad_(), valid, native, attributes))
    return Observation('test', 'S2', tuple(bands))


def test_scene_keys():
    assert scene({'acquisition_key': 'LC08:20261102:193045'}) == scene(
        {'acquisition_key': 'LC08_L2SP_193045_20261102_02_T1'})
    assert scene({'acquisition_key': 'S2A:20261111:35RNK'}) == scene(
        {'acquisition_key': 'S2A_MSIL2A_20261111T085121_R107_T35RNK_20261111T140707'})


@pytest.mark.parametrize('kind', ('A', 'B1'))
def test_hidden_values_predictions_and_gradients(kind):
    obs = sample()
    model = Pretraining(Config(slots=4))
    view = task_view(obs, kind, 2)
    expected = model(view)
    changed = []
    target_by_band = {key: request.pixels for key, request in zip(view.keys, view.requests)}
    for band in obs.bands:
        replacement = band.values.detach().clone()
        replacement[~band.valid] = float('nan')
        if band.key in target_by_band:
            replacement.flatten()[target_by_band[band.key]] += 100
        changed.append(replace(band, values=replacement))
    perturbed = task_view(replace(obs, bands=tuple(changed)), kind, 2)
    torch.testing.assert_close(expected, model(perturbed))
    assert objective(expected, perturbed.truth) > objective(expected, view.truth)
    objective(expected, view.truth).backward()
    for band in obs.bands:
        if band.values.grad is not None:
            assert not band.values.grad[~band.valid].any()
            if band.key in target_by_band:
                assert not band.values.grad.flatten()[target_by_band[band.key]].any()
        if kind == 'B1' and band.key in target_by_band:
            assert band.values.grad is None
    model_grads = [model.encoder.stems[0][0].weight, model.encoder.stems[1][0].weight,
                   model.encoder.steps[-1][-1].window.q.weight, model.encoder.descriptor.fields[0][0].weight]
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().max() > 0 for p in model_grads)


def test_permutation_and_common_geography():
    obs = sample()
    model = Pretraining(Config(slots=4)).eval()
    a, b = task_view(obs, 'A', 2), task_view(obs, 'B1', 2)
    assert a.visible.work is b.visible.work
    # Equal-geometry bands hide the identical native geography.
    target = {key: req.pixels for key, req in zip(a.keys, a.requests)}
    torch.testing.assert_close(target['0'], target['2'])
    torch.testing.assert_close(target['1'], target['3'])
    reversed_obs = replace(obs, sensor='renamed', bands=obs.bands[::-1])
    reordered = task_view(reversed_obs, 'A', 2)
    with torch.no_grad():
        stages = model.encoder(a.visible).stages
        for x, y in zip(stages, model.encoder(reordered.visible).stages):
            torch.testing.assert_close(x.values, y.values, atol=3e-6, rtol=2e-5)
    assert task_view(replace(obs, bands=obs.bands[:1]), 'B1', 2) is None
    assert task_view(replace(obs, bands=obs.bands[:1]), 'A', 2) is not None


def test_geometry_area_round_trip_and_padding():
    obs = sample()
    grids = tuple(b.grid for b in obs.bands)
    for stride in (1, 2):
        work, mapping = build_work_grid(grids, stride, torch.device('cpu'))
        for grid, area in mapping.items():
            keep = torch.ones(grid.shape, dtype=torch.bool)
            for native in (torch.ones(grid.shape), torch.eye(*grid.shape), torch.arange(grid.shape[1]).expand(grid.shape).float()):
                values, support = area.aggregate(native.flatten()[:, None], keep)
                native_integral = native.sum() * area.pixel_area
                mapped_integral = (values.flatten() * support * area.region_area).sum()
                torch.testing.assert_close(native_integral, mapped_integral)
            torch.testing.assert_close(area.native(torch.ones(work.valid.numel())), torch.ones(area.pixels))
        assert work.merge().shape == tuple((n+1)//2 for n in work.shape)
    rotated = replace(grids[0], affine=(10., 2., 0., 0., -10., 0.))
    with pytest.raises(ValueError, match='axis-aligned'):
        build_work_grid((rotated,), 2, torch.device('cpu'))


def test_attention_against_explicit_scores():
    torch.manual_seed(17)
    layer = Attention(8, 2).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    bias = torch.randn(1, 2, 5, 5, dtype=torch.float64)
    out = layer(x, valid=valid, bias=bias)
    q = layer.q(x).reshape(2, 5, 2, 4).transpose(1, 2)
    k, v = [p.reshape(2, 5, 2, 4).transpose(1, 2) for p in layer.kv(x).chunk(2, -1)]
    explicit = torch.softmax((q[:1] @ k[:1].transpose(-2, -1) / 2 + bias).masked_fill(
        ~valid[:1, None, None], -torch.inf), -1) @ v[:1]
    expected = layer.out(explicit.transpose(1, 2).flatten(2))
    torch.testing.assert_close(out[:1], expected)
    assert torch.count_nonzero(out[1]) == 0
    torch.testing.assert_close(torch.autograd.grad(out.square().sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(expected.square().sum(), x)[0])
    identity = torch.tensor([[[1., 0.], [0., 1.]]])
    mask = torch.tensor([[[True, False], [True, False]]])
    torch.testing.assert_close(F.scaled_dot_product_attention(identity, identity, identity, attn_mask=mask),
                               identity[:, :1].expand(-1, 2, -1))
    mha = torch.nn.MultiheadAttention(2, 1, bias=False, batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.eye(2).repeat(3, 1))
        mha.out_proj.weight.copy_(torch.eye(2))
    torch.testing.assert_close(mha(identity, identity, identity, key_padding_mask=~valid[:1, :2])[0],
                               identity[:, :1].expand(-1, 2, -1))


def test_shift_no_wrap_and_slot_merge():
    torch.manual_seed(17)
    block = GroupBlock(64, 4, shifted=True)
    x = torch.randn((3, 17, 4, 64), requires_grad=True)
    valid = torch.ones((3, 17), dtype=torch.bool)
    valid[:, 13:] = False
    out = block(x, valid, torch.zeros(4, 64), torch.randn(64), torch.randn(64), (20., 20.))
    gradient = torch.autograd.grad(out[:, 0].square().sum(), x)[0]
    assert not gradient[:, -1].any()
    assert gradient[:, 1].abs().max() > 0
    assert not out[:, 13:].any()
    simple = torch.arange(18).reshape(3, 3, 2, 1)
    merged = merge(simple, torch.nn.Identity())
    assert merged[0, 0].tolist() == [[0, 2, 6, 8], [1, 3, 7, 9]]
    assert merged[1, 1].tolist() == [[16, 0, 0, 0], [17, 0, 0, 0]]


@pytest.mark.parametrize('slots', (4, 8, 16, 32))
def test_slot_shape_decoder_chunking_and_target_properties(slots):
    view = task_view(sample(), 'B1', 2)
    config = Config(slots=slots)
    model = Pretraining(config)
    encoded = model.encoder(view.visible)
    assert [s.values.shape[2:] for s in encoded.stages] == [(slots+1, c) for c in (64, 128, 256)]
    tiny = Decoder(replace(config, target_chunk=17, position_chunk=9))
    tiny.load_state_dict(model.decoder.state_dict())
    left, right = model.decoder(encoded, view.requests), tiny(encoded, view.requests)
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        torch.autograd.grad(objective(left, view.truth), encoded.stages[0].values, retain_graph=True)[0],
        torch.autograd.grad(objective(right, view.truth), encoded.stages[0].values, retain_graph=True)[0],
        atol=2e-6, rtol=2e-5)
    req = view.requests[0]
    curve = Attribute('observed', 'um', 'fixture', samples=((.6, 0., 0., 1.),))
    revised = replace(req, physics=replace(req.physics, srf=curve))
    assert not torch.allclose(model.decoder(encoded, (req,))[0], model.decoder(encoded, (revised,))[0])


def test_native_metrics():
    truth = np.array([[1., 2.], [np.nan, 4.]])
    pred = truth + 1
    assert native_metrics(pred, truth) == dict(mse=1., mae=1., bias=1., gradient_mse=0.,
                                                texture_mse=0., mse_2x2=1., mse_4x4=1.)

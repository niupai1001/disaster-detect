import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from encoder_v3.hierarchical.model import HierarchicalMAE
from encoder_v3.hierarchical_revised.attention import IndexedAttention, Metrics
from encoder_v3.hierarchical_revised.frontend import Descriptor, typed_fields
from encoder_v3.hierarchical_revised.model import Config, Decoder
from encoder_v3.hierarchical_revised.coarse_shared import SharedCoarseDecoder
from encoder_v3.hierarchical_revised.profile import CONFIG, synthetic
from encoder_v3.hierarchical_revised.tasks import Pretraining, TaskView, task_masks
from encoder_v3.shared_mae.metadata import PhysicalChannel


@pytest.fixture(scope="module", autouse=True)
def threads():
    count = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(count)


@pytest.fixture
def cfg():
    return Config(**json.loads(CONFIG.read_text(encoding="utf-8"))["model"])


def test_descriptor_shared_typed_and_conditional(cfg):
    d = Descriptor(cfg.descriptor)
    optical = PhysicalChannel("optical", "surface_reflectance", 10, 10, .45, source="x")
    sar = SimpleNamespace(modality="sar", quantity="sigma0_db", gx=10, gy=10,
                          frequency_ghz=5.4, polarization="VV")
    assert typed_fields(sar) == typed_fields(SimpleNamespace(**vars(sar), center_um=1.7))
    missing = SimpleNamespace(modality="optical", quantity="surface_reflectance", gx=10, gy=10)
    a, b, c = d([optical, sar, missing])
    assert not torch.allclose(a, b) and not torch.allclose(a, c)
    torch.testing.assert_close(d([sar, optical, missing]), torch.stack((b, a, c)))
    torch.testing.assert_close(d([replace(optical, source="other")])[0], a)
    conv = Pretraining(cfg).encoder.conv
    kernels, _ = conv.generator(torch.stack((a, b)))
    assert kernels.shape == (2, cfg.fine, 1, 3, 3)
    assert not torch.allclose(kernels[0], kernels[1])


def test_sparse_attention_dense_output_gradient_and_empty_rows():
    torch.manual_seed(2)
    q, k, v = [torch.randn(5, 2, 3, requires_grad=True) for _ in range(3)]
    mask = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0],
                         [1, 0, 0, 1, 1], [0, 0, 0, 1, 0]], dtype=torch.bool)
    def graph(start, end):
        return np.nonzero(mask[start:end].numpy())
    actual, present = IndexedAttention.apply(q, k, v, graph, 2, "test", None, q.device)
    expected = torch.stack([
        torch.einsum("hk,khd->hd", (torch.einsum("hd,khd->hk", q[i], k[row])/3**.5).softmax(-1), v[row])
        if row.any() else torch.zeros_like(q[i]) for i, row in enumerate(mask)])
    torch.testing.assert_close(actual, expected)
    assert torch.equal(present, mask.any(1))
    g = torch.randn_like(q)
    got = torch.autograd.grad(actual, (q, k, v), g, retain_graph=True)
    ref = torch.autograd.grad(expected, (q, k, v), g)
    for left, right in zip(got, ref):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("task", ["A", "B1"])
def test_targets_never_enter_prediction_or_gradient(cfg, task):
    model = Pretraining(cfg)
    sample = synthetic(torch.device("cpu"))
    sample.channels[0].valid[0, 0] = False
    keeps, hidden = task_masks(sample, cfg, task, 17, 0)
    changed = copy.deepcopy(sample)
    for channel, mask in zip(changed.channels, hidden):
        channel.values[mask] += 123
        channel.values.requires_grad_()
    first = model(sample, kind=task, seed=17)
    second = model(changed, kind=task, seed=17)
    torch.testing.assert_close(first.predictions, second.predictions, rtol=0, atol=0)
    for name in ("fine", "preserving", "joint", "regional", "global_latents"):
        torch.testing.assert_close(getattr(first.encoded, name), getattr(second.encoded, name), rtol=0, atol=0)
    assert not torch.isclose(first.loss, second.loss)
    grads = torch.autograd.grad(second.predictions.sum(), [c.values for c in changed.channels],
                                allow_unused=True)
    for grad, keep in zip(grads, keeps):
        if grad is not None:
            assert torch.count_nonzero(grad[~keep]) == 0
    if task == "B1":
        for c, mask in enumerate(hidden):
            if mask.any():
                assert not len(second.encoded.native.ids[c])
                assert c not in second.encoded.middle_channels


def test_fully_hidden_metadata_only_conditions_decoder(cfg):
    sample = synthetic(torch.device("cpu"), bands=16, shape=(3, 4))
    for seed in range(20):
        keeps, targets = task_masks(sample, cfg, "B1", seed, 0)
        if sum(bool(k.any()) for k in keeps) > cfg.channel_cap+1:
            break
    else:
        pytest.fail("fixture requires more visible channels than the cap")
    changed = copy.deepcopy(sample)
    for i, mask in enumerate(targets):
        if mask.any():
            spec = changed.channels[i].physical
            changed.channels[i].physical = replace(spec, center_um=spec.center_um+1.2)
    model = Pretraining(cfg)
    with torch.no_grad():
        first = model(sample, kind="B1", seed=seed)
        second = model(changed, kind="B1", seed=seed)
    for name in ("fine", "preserving", "joint", "regional", "global_latents"):
        torch.testing.assert_close(getattr(first.encoded, name), getattr(second.encoded, name), rtol=0, atol=0)
    assert not torch.allclose(first.predictions, second.predictions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("decoder_class", [Decoder, SharedCoarseDecoder])
def test_cuda_tiled_offload_matches_resident(cfg, monkeypatch, decoder_class):
    from encoder_v3.hierarchical_revised import attention

    device = torch.device("cuda:0")
    torch.manual_seed(24)
    resident = Pretraining(cfg, decoder_class=decoder_class).to(device)
    spilled = copy.deepcopy(resident)
    sample = synthetic(device)
    first = resident(sample, kind="B1", seed=17)
    first.loss.backward()
    monkeypatch.setattr(attention, "DEVICE_PROJECT_BYTES", 1)
    with torch.autograd.graph.save_on_cpu():
        second = spilled(sample, kind="B1", seed=17)
    second.loss.backward()
    torch.testing.assert_close(first.predictions, second.predictions, atol=5e-6, rtol=1e-5)
    for a, b in zip(resident.parameters(), spilled.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=5e-6, rtol=1e-4)


def test_a_masks_match_legacy(cfg):
    sample = synthetic(torch.device("cpu"))
    legacy = HierarchicalMAE(patch_metres=cfg.mask_patch_metres, mask_ratio=cfg.mask_ratio)
    keep, target, _ = legacy.masks(sample, legacy.layout(sample), "A", 17, 3)
    k, t = task_masks(sample, replace(cfg, middle_metres=512), "A", 17, 3)
    assert all(torch.equal(a, b) for a, b in zip(keep + target, k + t))


def test_b1_optical_only_and_single_channel_a(cfg):
    sample = synthetic(torch.device("cpu"), bands=3)
    sample.channels[-1].physical = PhysicalChannel("sar", "sigma0_db", 10, 10,
                                                  frequency_ghz=5.4, polarization="VV", source="sar")
    keep, target = task_masks(sample, cfg, "B1", 17, 0)
    assert not target[-1].any() and torch.equal(keep[-1], sample.channels[-1].valid)
    sample.channels = sample.channels[-1:]
    with pytest.raises(ValueError, match="two valid optical"):
        task_masks(sample, cfg, "B1", 17, 0)
    result = Pretraining(cfg)(sample, kind="A", seed=17)
    assert torch.isfinite(result.loss)


def test_hierarchy_one_way_and_read_paths(cfg):
    model = Pretraining(cfg)
    sample = synthetic(torch.device("cpu"))
    view = TaskView.build(sample, *task_masks(sample, cfg, "B1", 17, 0))
    metrics = Metrics()
    first = model.encoder(view.visible, metrics)
    assert len(first.fine) == sum(int(c.keep.sum()) for c in view.visible)
    assert first.preserving.shape == (len(first.middle_cells)*cfg.spatial_queries, cfg.middle)
    assert first.joint.shape == (len(first.joint_cells)*cfg.joint_queries, cfg.middle)
    assert first.regional.shape == (len(first.regional_cells)*cfg.regional_queries, cfg.coarse)
    assert first.global_latents.shape == (cfg.global_queries, cfg.coarse)
    assert metrics.attention["global_self"]["pairs"] == cfg.global_queries**2
    assert metrics.attention["global_read"]["pairs"] == cfg.global_queries*len(first.regional)
    assert sum(metrics.channel_degree.values()) == len(first.fine)*cfg.depth
    assert not torch.allclose(first.joint[0], first.joint[1])
    with torch.no_grad():
        model.encoder.global_queries.add_(.2)
        model.encoder.region_queries.add_(.3)
        second = model.encoder(view.visible)
        torch.testing.assert_close(first.preserving, second.preserving)
        torch.testing.assert_close(first.joint, second.joint)
        model.encoder.compress_queries.add_(.4)
        third = model.encoder(view.visible)
        torch.testing.assert_close(first.fine, third.fine)
    assert not torch.allclose(second.regional, first.regional)
    del model.decoder
    assert len(model.encoder(view.visible).fine) == len(first.fine)


def test_tiles_cache_and_gradients_match(cfg):
    torch.manual_seed(15)
    sample = synthetic(torch.device("cpu"))
    tiled = Pretraining(replace(cfg, query_chunk=53, conv_tile_rows=3, decoder_query_chunk=53))
    regular = Pretraining(replace(cfg, query_chunk=100000, conv_tile_rows=100, decoder_query_chunk=100000))
    regular.load_state_dict(tiled.state_dict())
    a = tiled(sample, kind="A", seed=17)
    b = regular(sample, kind="A", seed=17)
    torch.testing.assert_close(a.predictions, b.predictions, atol=2e-6, rtol=1e-5)
    a.loss.backward()
    b.loss.backward()
    for left, right in zip(tiled.parameters(), regular.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=4e-6, rtol=2e-4)


@pytest.mark.parametrize("decoder_class", [Decoder, SharedCoarseDecoder])
def test_synthetic_overfit_and_exact_next_step_resume(cfg, tmp_path, decoder_class):
    torch.manual_seed(2)
    sample = synthetic(torch.device("cpu"), bands=3)
    model = Pretraining(cfg, decoder_class=decoder_class)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        losses = [model(sample, kind=kind, seed=17).loss for kind in ("A", "B1")]
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    first = step(model, opt)
    for _ in range(7):
        last = step(model, opt)
    assert last < first*.6
    path = tmp_path/"engineering.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, path)
    data = torch.load(path, weights_only=True)
    restored = Pretraining(cfg, decoder_class=decoder_class)
    restored.load_state_dict(data["model"])
    other = torch.optim.AdamW(restored.parameters(), lr=.002)
    other.load_state_dict(data["optimizer"])
    torch.testing.assert_close(step(model, opt), step(restored, other), rtol=0, atol=0)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)



@pytest.mark.parametrize("decoder_class", [Decoder, SharedCoarseDecoder])
def test_complete_channel_permutation_with_fixed_physical_masks(cfg, decoder_class):
    sample = synthetic(torch.device("cpu"), bands=12)
    keeps, targets = task_masks(sample, cfg, "A", 17, 0)
    order = list(reversed(range(len(sample.channels))))
    permuted = copy.deepcopy(sample)
    permuted.channels = [permuted.channels[i] for i in order]
    left = TaskView.build(sample, keeps, targets)
    right = TaskView.build(permuted, [keeps[i] for i in order], [targets[i] for i in order])
    model = Pretraining(cfg, decoder_class=decoder_class)
    with torch.no_grad():
        first, second = model.encoder(left.visible), model.encoder(right.visible)
        for new, old in enumerate(order):
            a, b = first.native.offsets[old:old+2]
            x, y = second.native.offsets[new:new+2]
            torch.testing.assert_close(first.fine[a:b], second.fine[x:y], atol=2e-6, rtol=1e-5)
        lookup = {(*cell, int(channel)): i for i, (cell, channel) in enumerate(
            zip(first.middle_cells, first.middle_channels))}
        rows = [lookup[(*cell, order[int(channel)])] for cell, channel in
                zip(second.middle_cells, second.middle_channels)]
        torch.testing.assert_close(first.preserving.reshape(-1, cfg.spatial_queries, cfg.middle)[rows],
                                   second.preserving.reshape(-1, cfg.spatial_queries, cfg.middle),
                                   atol=3e-6, rtol=1e-5)
        for name in ("joint", "regional", "global_latents"):
            torch.testing.assert_close(getattr(first, name), getattr(second, name), atol=3e-6, rtol=1e-5)
        spec_a, spec_b = [c.physical for c in sample.channels], [c.physical for c in permuted.channels]
        pa = model.decoder(first, left.requests, spec_a, model.encoder.descriptor(spec_a))
        pb = model.decoder(second, right.requests, spec_b, model.encoder.descriptor(spec_b))
        for new, old in enumerate(order):
            a, b = left.requests.offsets[old:old+2]
            x, y = right.requests.offsets[new:new+2]
            torch.testing.assert_close(pa[a:b], pb[x:y], atol=3e-6, rtol=1e-5)
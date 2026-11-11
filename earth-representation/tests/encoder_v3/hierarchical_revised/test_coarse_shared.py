import copy
import json

import numpy as np
import pytest
import torch

from encoder_v3.hierarchical_revised import readout
from encoder_v3.hierarchical_revised.attention import Metrics
from encoder_v3.hierarchical_revised.coarse_shared import SharedCoarseDecoder, regional_supports
from encoder_v3.hierarchical_revised.geometry import Grid
from encoder_v3.hierarchical_revised.model import Config
from encoder_v3.hierarchical_revised.profile import CONFIG, synthetic
from encoder_v3.hierarchical_revised.tasks import Pretraining, task_masks


@pytest.mark.parametrize("task", ["A", "B1"])
def test_shared_coarse_contract_leakage_and_eager_gradient(task, monkeypatch):
    torch.set_num_threads(2)
    cfg = Config(**json.loads(CONFIG.read_text())["model"])
    sample = synthetic(torch.device("cpu"))
    torch.manual_seed(24)
    model = Pretraining(cfg, decoder_class=SharedCoarseDecoder)
    reference = copy.deepcopy(model)
    baseline = Pretraining(cfg)
    baseline.load_state_dict(model.state_dict())
    metrics = Metrics()
    first = model(sample, kind=task, seed=17, metrics=metrics)
    first.loss.backward()
    previous = baseline(sample, kind=task, seed=17)
    torch.testing.assert_close(first.encoded.fine, previous.encoded.fine, atol=0, rtol=0)
    torch.testing.assert_close(first.encoded.global_latents, previous.encoded.global_latents, atol=0, rtol=0)
    assert metrics.attention["decoder_coarse"]["q"] == metrics.tokens["coarse_readout_groups"]
    assert metrics.attention["decoder_coarse"]["q"] < len(first.predictions)

    def eager(desc, lk, lv, rk, rv, decoder, query_features, local, coarse, count, metrics, *parameters):
        bands, geography = query_features(0, count)
        q = decoder.condition(torch.cat((desc[bands], geography), dim=-1))
        q = decoder.local.read_projected(q, lk, lv, local, count, "decoder_local", metrics)
        return decoder.output(q+rk[coarse.groups]).squeeze(-1)
    monkeypatch.setattr(readout.ReadoutTiles, "apply", eager)
    second = reference(sample, kind=task, seed=17)
    second.loss.backward()
    torch.testing.assert_close(first.predictions, second.predictions, atol=3e-6, rtol=1e-5)
    for p, q in zip(model.parameters(), reference.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=2e-4)

    keeps, hidden = task_masks(sample, cfg, task, 17, 0)
    changed = copy.deepcopy(sample)
    for channel, mask in zip(changed.channels, hidden):
        channel.values[mask] += 100
        channel.values.requires_grad_()
    third = reference(changed, kind=task, seed=17)
    torch.testing.assert_close(second.predictions, third.predictions, atol=0, rtol=0)
    assert not torch.isclose(second.loss, third.loss)
    gradients = torch.autograd.grad(third.predictions.sum(), [c.values for c in changed.channels],
                                    allow_unused=True)
    for gradient, keep in zip(gradients, keeps):
        if gradient is not None:
            assert not torch.count_nonzero(gradient[~keep])


def test_shared_coarse_supports_keep_boundary_identity(tmp_path, monkeypatch):
    from encoder_v3.hierarchical_revised import coarse_shared
    grid = Grid("EPSG:8857", (20., 0., 502., 0., -20., 522.), (1, 1))
    monkeypatch.setenv("V3_H_R_GEOMETRY_CACHE", str(tmp_path))
    support = regional_supports(grid, 128, 4)
    assert set(map(tuple, support.cells[support.keys])) == {(0,0),(0,1),(1,0),(1,1)}
    assert len(support.pixel_group) == 1
    regional_supports.cache_clear()
    monkeypatch.setattr(coarse_shared, "middle_map", lambda *args: pytest.fail("rebuilt fixed geometry"))
    restored = regional_supports(grid, 128, 4)
    np.testing.assert_array_equal(support.keys, restored.keys)

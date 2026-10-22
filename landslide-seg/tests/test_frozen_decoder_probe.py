"""Actual official model: freeze mode, source identity, shared head and updates."""
from pathlib import Path
import copy
import json
import pytest
import torch
from landslide_seg.config import load_segformer_config, set_seed
from landslide_seg.models.factory import build_model
from landslide_seg.models.segformer_probe import state_digest
from landslide_seg.training.segformer_trainer import _make_optimizer

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/segformer/v4r2_probe/experiments.json"


@pytest.fixture(scope="module")
def pair():
    torch.set_num_threads(2)
    configs, models = [], []
    for entry in json.loads(REGISTRY.read_text()):
        cfg = load_segformer_config(ROOT / entry["config"])
        set_seed(1234, True)
        configs.append(cfg)
        models.append(build_model(cfg["segformer"]))
    return configs, models


def test_source_encoders_and_common_decoder(pair):
    configs, models = pair
    assert models[0].probe_report["decoder_initial_sha256"] == models[1].probe_report["decoder_initial_sha256"]
    assert models[0].probe_report["encoder_initial_sha256"] != models[1].probe_report["encoder_initial_sha256"]
    for cfg, model in zip(configs, models):
        source = torch.load(ROOT / cfg["segformer"]["model"]["probe"]["checkpoint"], map_location="cpu", weights_only=False)
        prefix = "model.segformer."
        assert state_digest({k[len(prefix):]: v for k, v in source["model"].items() if k.startswith(prefix)}) == model.probe_report["encoder_initial_sha256"]
        prefix = "model.decode_head."
        assert state_digest({k[len(prefix):]: v for k, v in source["model"].items() if k.startswith(prefix)}) != model.probe_report["decoder_initial_sha256"]


def test_train_mode_and_optimizer_updates_only_decoder(pair):
    cfg, model = pair[0][0], pair[1][0]
    model.train()
    assert model.training and model.model.decode_head.training
    assert all(not module.training for module in model.model.segformer.modules())
    optimizer = _make_optimizer(model, cfg)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert ids == {id(p) for p in model.model.decode_head.parameters()}
    x = torch.randn(2, 10, 32, 32)
    for _ in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)["logits"]
        assert logits.shape == (2, 1, 32, 32)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
        assert all(p.grad is None for p in model.model.segformer.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model.decode_head.parameters())
        optimizer.step()
    integrity = model.probe_integrity()
    assert integrity["encoder_unchanged"] and integrity["decoder_changed"]
    assert not integrity["encoder_has_grad"] and not integrity["encoder_training"]


def test_probe_source_hash_rejected(pair):
    cfg = copy.deepcopy(pair[0][0])
    cfg["segformer"]["model"]["probe"]["checkpoint_sha256"] = "invalid"
    with pytest.raises(ValueError, match="checkpoint hash"):
        build_model(cfg["segformer"])


def test_probe_normalization_mismatch_rejected(pair):
    cfg = copy.deepcopy(pair[0][0])
    cfg["segformer"]["model"]["probe"]["stats_sha256"] = "invalid"
    with pytest.raises(ValueError, match="normalization"):
        build_model(cfg["segformer"])

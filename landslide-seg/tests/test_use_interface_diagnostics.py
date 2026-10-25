import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "diagnose_use_interface",
    Path(__file__).resolve().parents[1] / "scripts/diagnose_use_interface.py",
)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_size_class_boundaries():
    assert diag.size_class(0) == "1_4"
    assert diag.size_class(1) == "1_4"
    assert diag.size_class(4) == "1_4"
    assert diag.size_class(5) == "5_25"
    assert diag.size_class(25) == "5_25"
    assert diag.size_class(26) == "26_100"
    assert diag.size_class(100) == "26_100"
    assert diag.size_class(101) == "101_plus"
    assert diag.size_class(10**9) == "101_plus"


def test_stratified_allocation_total_and_minimum():
    counts = {f"e{i}::{s}": (i + 1) * 2 for i in range(9) for s in ("1_4", "5_25", "26_100", "101_plus")}
    allocation = diag.stratified_allocation(counts, 128)
    assert sum(allocation.values()) == 128
    assert set(allocation) == set(counts)
    assert all(1 <= allocation[key] <= counts[key] for key in counts)
    assert allocation == diag.stratified_allocation(counts, 128)


def test_allocate_checks_leaves_one_fit_per_stratum():
    sizes = {f"e{i}::{s}": 3 + (i % 4) for i in range(9) for s in ("1_4", "5_25", "26_100", "101_plus")}
    checks = diag.allocate_checks(sizes, 32)
    assert sum(checks.values()) == 32
    assert all(0 <= checks[key] <= max(0, sizes[key] - 1) for key in sizes)


def test_selection_key_is_deterministic_and_order_independent():
    assert diag.selection_key("a") == diag.selection_key("a")
    assert diag.selection_key("a") != diag.selection_key("b")
    assert len(diag.selection_key("a")) == 64


def test_feature_accumulator_matches_torch_moments():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 7, 9)
    acc = diag.FeatureAccumulator()
    acc.add(x)
    stats = acc.finalize()
    assert stats["shape"] == [5, 7, 9]
    assert abs(stats["mean"] - float(x.mean())) < 1e-6
    channel_std = x.std(dim=(0, 2, 3), unbiased=False).mean()
    assert abs(stats["std"] - float(channel_std)) < 1e-6
    assert 0.0 <= stats["effective_rank_ratio"] <= 1.0
    assert stats["lag1_horizontal"] is not None
    assert -1.001 <= stats["lag1_horizontal"] <= 1.001


def test_pair_accumulator_self_correlation_is_one():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 6, 8)
    pair = diag.PairAccumulator()
    pair.add(x, x)
    result = pair.finalize()
    for value in result["diag"]:
        assert abs(value - 1.0) < 1e-5
    assert all(abs(v) <= 1.0 + 1e-6 for row in result["corr"] for v in row)


def test_spatial_energy_radius_point_and_uniform():
    point = np.zeros((17, 17))
    point[8, 8] = 1.0
    radius = diag.spatial_energy_radius(point, (8, 8))
    assert radius["r50"] == 0.0 and radius["r90"] == 0.0
    assert radius["center_share"] == 1.0
    uniform = np.ones((17, 17))
    radius = diag.spatial_energy_radius(uniform, (8, 8))
    assert radius["r50"] < radius["r90"]
    assert radius["r90"] > 8.0


def _fake_batch(channels=10):
    x = torch.arange(channels, dtype=torch.float32).view(1, channels, 1, 1).expand(1, channels, 4, 4).clone()
    metadata = torch.arange(channels, dtype=torch.float32).view(1, channels, 1).expand(1, channels, 3).clone()
    known = torch.ones(1, channels, 3, dtype=torch.bool)
    band_valid = torch.ones(1, channels, dtype=torch.bool)
    pixel_valid = torch.ones(1, channels, 4, 4, dtype=torch.bool)
    observation = diag.OpticalBatch(
        x=x, band_metadata=metadata, known_mask=known, band_valid=band_valid,
        pixel_valid=pixel_valid, band_ids=[list(diag.S2_10_BANDS[:channels])],
        spatial_metadata=[{}],
    )
    return {
        "image": observation,
        "mask": torch.zeros(1, 4, 4),
        "valid": torch.ones(1, 4, 4, dtype=torch.bool),
        "sample_id": ["s"],
        "event_id": ["e"],
    }


def test_permute_batch_joint_keeps_band_identity_pairs():
    batch = _fake_batch()
    perm = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    joint = diag.permute_batch(batch, perm, "joint", "use")
    for new_position, old_position in enumerate(perm):
        assert joint["image"]["x"][0, new_position, 0, 0] == batch["image"]["x"][0, old_position, 0, 0]
        assert joint["image"]["band_metadata"][0, new_position, 0] == batch["image"]["band_metadata"][0, old_position, 0]
        assert joint["image"]["band_ids"][0][new_position] == batch["image"]["band_ids"][0][old_position]
    metadata_only = diag.permute_batch(batch, perm, "metadata", "use")
    assert torch.equal(metadata_only["image"]["x"], batch["image"]["x"])
    assert not torch.equal(metadata_only["image"]["band_metadata"], batch["image"]["band_metadata"])


def test_clone_batch_is_independent():
    batch = _fake_batch()
    clone = diag.clone_batch(batch)
    clone["image"]["x"][0, 0, 0, 0] = -1
    assert batch["image"]["x"][0, 0, 0, 0] != -1
    assert isinstance(clone["image"], diag.OpticalBatch)


def test_subset_batch_uses_named_order():
    batch = _fake_batch()
    subset = diag.subset_batch(batch, ("B05", "B02"))
    assert subset["image"]["x"].shape[1] == 2
    assert subset["image"]["x"][0, 0, 0, 0] == batch["image"]["x"][0, 3, 0, 0]
    assert subset["image"]["x"][0, 1, 0, 0] == batch["image"]["x"][0, 0, 0, 0]
    assert subset["image"]["band_ids"][0] == ["B05", "B02"]


def test_prob_delta_identical_and_changed():
    torch.manual_seed(0)
    p = torch.rand(2, 4, 4)
    valid = torch.ones(2, 4, 4, dtype=torch.bool)
    same = diag.prob_delta(p, p, valid)
    assert same["mean_abs_diff"] == 0.0 and same["iou_agreement"] == 1.0
    changed = diag.prob_delta(p, 1 - p, valid)
    assert changed["mean_abs_diff"] > 0.0


def test_group_names_for_both_interfaces():
    assert diag.group_name("use.tokenizer.0.weight", "use") == "use.tokenizer"
    assert diag.group_name("use.attention.in_proj_weight", "use") == "use.attention"
    assert diag.group_name("bridge.weight", "use") == "bridge"
    assert diag.group_name("visual.model.segformer.encoder.patch_embeddings.0.proj.weight", "use") == "mit.embed1"
    assert diag.group_name("model.segformer.encoder.layer_norm.1.weight", "direct") == "mit.stage2"
    assert diag.group_name("stem.net.0.weight", "direct") == "detail.stem"
    assert diag.group_name("model.decode_head.classifier.weight", "direct") == "decoder"


def test_probe_settings_short_horizon():
    cfg = {"segformer": {"optim": {
        "warmup_frac": 0.05, "encoder_lr": 6e-5, "head_lr": 6e-4,
        "weight_decay": 0.01, "grad_clip": 1.0,
    }}}
    settings = diag.probe_settings(cfg)
    assert settings["updates"] == 300
    assert settings["schedule_updates"] == 300
    assert settings["warmup_updates"] == 15
    assert settings["eval_every_updates"] == 50


def test_select_diagnostic_tiles_is_mixed_and_unique():
    records = []
    for event in ("a", "b", "c"):
        for i in range(10):
            records.append({"sample_id": f"{event}_{i}", "event_id": event})
    picked = diag.select_diagnostic_tiles(records, 9)
    assert len(picked) == 9
    assert len({r["sample_id"] for r in picked}) == 9
    assert {r["event_id"] for r in picked} == {"a", "b", "c"}
    assert picked == diag.select_diagnostic_tiles(records, 9)


def test_canonical_hash_is_key_order_independent():
    assert diag.content_sha256(diag.canonical_json({"b": 1, "a": 2})) == diag.content_sha256(
        diag.canonical_json({"a": 2, "b": 1})
    )

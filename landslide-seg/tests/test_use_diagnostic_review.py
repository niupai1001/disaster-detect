import importlib.util
import json
import tempfile
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "review_use_diagnostics",
    Path(__file__).resolve().parents[1] / "scripts/review_use_diagnostics.py",
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_linear_cka_identity_and_invertible_transform():
    torch.manual_seed(0)
    x = torch.randn(512, 6)
    assert abs(review.linear_cka(x, x) - 1.0) < 1e-6
    # Linear CKA is invariant to orthogonal transforms and isotropic scaling,
    # but not to arbitrary invertible mixing.
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    assert abs(review.linear_cka(x, x @ q) - 1.0) < 1e-5
    assert abs(review.linear_cka(x, -2.5 * x) - 1.0) < 1e-6


def test_linear_cka_independent_features_is_small():
    torch.manual_seed(1)
    x = torch.randn(4096, 8)
    y = torch.randn(4096, 8)
    value = review.linear_cka(x, y)
    assert 0.0 <= value < 0.1


def test_linear_cka_zero_variance_returns_zero():
    x = torch.ones(128, 4)
    y = torch.randn(128, 4)
    assert review.linear_cka(x, y) == 0.0


def test_json_roundtrip_and_formatting(tmp_path):
    path = tmp_path / "nested" / "value.json"
    review.write_json(path, {"a": 1, "b": "中文"})
    assert review.read_json(path) == {"a": 1, "b": "中文"}
    assert review._f(None) == "—"
    assert review._f(float("nan")) == "非有限"
    assert "e" in review._f(1e-5)


def test_sha_is_content_addressed(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    first = review.sha(path)
    path.write_bytes(b"abcd")
    assert first != review.sha(path)


def test_tile_manifest_identity_if_present():
    # The evidence is intentionally optional for a fresh checkout; when it is
    # present, the review must use the exact same fixed subset hash.
    manifest = review.OLD / "selection/diagnostic_tiles.json"
    if not manifest.is_file():
        return
    tiles = review.read_json(manifest)
    assert tiles["n_tiles"] == 128
    assert tiles["n_fit"] == 96
    assert tiles["n_check"] == 32
    assert len({row["sample_id"] for row in tiles["tiles"]}) == 128
    assert all(row["split"] == "TRAIN" for row in tiles["tiles"])

"""数据集测试：标准化只按训练统计、增强同步、preload 与懒加载一致、划分过滤。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from landslide_seg.data.segformer_dataset import (
    S2SegDataset,
    collate,
    filter_split,
    normalize_s2,
)


def _make_cache(tmp_path, n: int = 4):
    rng = np.random.default_rng(0)
    records, arrays = [], []
    for i in range(n):
        x = rng.integers(500, 3000, size=(10, 128, 128)).astype(np.int16)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:40, 30:50] = 1
        mask[60:65, 70:75] = 1
        valid = np.ones((128, 128), dtype=bool)
        valid[:5, :] = False
        p = tmp_path / f"s{i}.npz"
        np.savez_compressed(p, x10=x, mask=mask, valid=valid)
        records.append(
            {
                "sample_id": f"s{i}",
                "event_id": "e0",
                "cache": str(p),
                "split": "train" if i % 2 == 0 else "val",
                "mask_px": int(mask.sum()),
            }
        )
        arrays.append((x, mask, valid))
    stacked = np.stack([a[0] for a in arrays]).astype(np.float64)
    vmask = np.stack([a[2] for a in arrays])
    mean = [float(stacked[:, b][vmask].mean()) for b in range(10)]
    std = [float(stacked[:, b][vmask].std()) or 1.0 for b in range(10)]
    return records, arrays, {"mean": mean, "std": std}


def test_normalize_uses_given_stats(tmp_path):
    records, arrays, stats = _make_cache(tmp_path, 1)
    ds = S2SegDataset(records, stats)
    item = ds[0]
    raw = arrays[0][0].astype(np.float32)
    expected = (raw - np.asarray(stats["mean"], np.float32)[:, None, None]) / np.maximum(
        np.asarray(stats["std"], np.float32)[:, None, None], 1e-6
    )
    assert np.allclose(item["image"].numpy(), expected, atol=1e-5)
    assert np.array_equal(item["valid"].numpy(), arrays[0][2])
    assert np.array_equal(item["mask"].numpy(), arrays[0][1].astype(np.float32))


def test_normalize_formula_matches_manual():
    x = np.random.default_rng(1).normal(1000, 100, size=(10, 8, 8)).astype(np.float32)
    stats = {"mean": [1000.0] * 10, "std": [100.0] * 10}
    out = normalize_s2(x, stats)
    assert np.allclose(out, (x - 1000.0) / 100.0, atol=1e-5)


def test_preload_matches_lazy(tmp_path):
    records, _, stats = _make_cache(tmp_path, 2)
    lazy = S2SegDataset(records, stats, preload=False)[0]
    pre = S2SegDataset(records, stats, preload=True)[0]
    assert torch.equal(lazy["image"], pre["image"])
    assert torch.equal(lazy["mask"], pre["mask"])
    assert torch.equal(lazy["valid"], pre["valid"])


def test_augment_preserves_counts_and_is_deterministic(tmp_path):
    records, arrays, stats = _make_cache(tmp_path, 4)
    ds = S2SegDataset(records, stats, augment=True, seed=1234)
    mean = np.asarray(stats["mean"], np.float32)[:, None, None]
    std = np.maximum(np.asarray(stats["std"], np.float32)[:, None, None], 1e-6)
    for idx in range(len(records)):
        ds.set_epoch(0)
        a = ds[idx]
        ds.set_epoch(0)
        b = ds[idx]
        assert torch.equal(a["image"], b["image"])  # 同一 (seed, epoch, index) 必须一致
        # 翻转/旋转只重排像素：正像素数、有效像素数、逐通道和都应保持
        assert int(a["mask"].sum()) == int(arrays[idx][1].sum())
        assert int(a["valid"].sum()) == int(arrays[idx][2].sum())
        expected_sum = float(((arrays[idx][0].astype(np.float32) - mean) / std).sum())
        assert abs(float(a["image"].sum()) - expected_sum) < 1e-2


def test_collate_shapes(tmp_path):
    records, _, stats = _make_cache(tmp_path, 3)
    ds = S2SegDataset(records, stats)
    batch = collate([ds[i] for i in range(3)])
    assert tuple(batch["image"].shape) == (3, 10, 128, 128)
    assert tuple(batch["mask"].shape) == (3, 128, 128)
    assert tuple(batch["valid"].shape) == (3, 128, 128)
    assert batch["sample_id"] == ["s0", "s1", "s2"]


def test_filter_split(tmp_path):
    records, _, _ = _make_cache(tmp_path, 4)
    assert [r["sample_id"] for r in filter_split(records, "train")] == ["s0", "s2"]
    assert [r["sample_id"] for r in filter_split(records, "val")] == ["s1", "s3"]
    assert filter_split(records, "test") == []


def test_missing_manifest_raises(tmp_path):
    from landslide_seg.data.segformer_dataset import read_manifest

    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "nope.jsonl")

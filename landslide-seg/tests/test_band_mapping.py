"""波段契约测试：顺序、缺失检测、严格/合成两种策略。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.data.band_mapping import (  # noqa: E402
    ANY_SAT_S2_BANDS,
    L4S_BANDS,
    S2_L2A_BANDS,
    MissingBandError,
    normalize_anysat,
    to_anysat,
    to_terramind,
)


def _l4s_tensor(value=1.0):
    return np.full((len(L4S_BANDS), 8, 8), value, dtype=np.float32)


def test_band_tables():
    assert len(S2_L2A_BANDS) == 12
    assert "B8A" in S2_L2A_BANDS and "B10" not in S2_L2A_BANDS
    assert len(ANY_SAT_S2_BANDS) == 10
    assert "B8A" in ANY_SAT_S2_BANDS
    assert len(L4S_BANDS) == 14
    assert "B10" in L4S_BANDS and "B8A" not in L4S_BANDS


def test_strict_refuses_missing_b8a():
    with pytest.raises(MissingBandError) as e:
        to_terramind(_l4s_tensor(), L4S_BANDS, policy="strict")
    assert e.value.missing == ["B8A"]

    with pytest.raises(MissingBandError) as e2:
        to_anysat(_l4s_tensor(), L4S_BANDS, policy="strict")
    assert e2.value.missing == ["B8A"]


def test_smoke_policy_marks_synthetic():
    r = to_terramind(_l4s_tensor(), L4S_BANDS, policy="synthetic_smoke")
    assert r.tensor.shape == (12, 8, 8)
    assert r.synthetic_bands == {"B8A": "copied_from:B08"}
    assert not r.is_fully_observed
    # 12 通道顺序必须与官方一致
    assert r.target_bands == S2_L2A_BANDS


def test_terramind_channel_order_is_not_identity():
    """L4S 有 B10 没有 B8A，重排后通道不能等于原顺序（否则说明映射写错了）。"""
    arr = np.stack([np.full((2, 2), i, dtype=np.float32) for i in range(14)])
    r = to_terramind(arr, L4S_BANDS, policy="synthetic_smoke")
    # 通道 8 应为 B8A 的近似（来自 B08=索引7），通道 9 为 B09（索引8）
    assert r.tensor[8].mean() == pytest.approx(7.0)
    assert r.tensor[9].mean() == pytest.approx(8.0)
    # 原 B10（索引9）不应出现在 TiM 输入里
    assert r.tensor[9].mean() != pytest.approx(9.0)


def test_anysat_selection_and_norm():
    arr = np.stack([np.full((2, 2), float(i), dtype=np.float32) for i in range(14)])
    r = to_anysat(arr, L4S_BANDS, policy="synthetic_smoke")
    assert r.tensor.shape[0] == 10
    # B02 = 索引1
    assert r.tensor[0].mean() == pytest.approx(1.0)

    x = normalize_anysat(np.full((10, 2, 2), 4000.0, dtype=np.float32))
    assert x.shape == (10, 2, 2)
    assert np.isfinite(x).all()

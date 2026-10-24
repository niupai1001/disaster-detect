"""固定跨度窗口：只取参考日期前 N 天内的帧，且与数量组共用同一套选择规则。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.data.temporal_v6 import select_frame_indices  # noqa: E402

spec = importlib.util.spec_from_file_location("research_preflight", ROOT / "scripts/check_research_protocol.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

DAYS = np.asarray([0, 100, 200, 300, 400])
ANCHOR = 4


def test_window_keeps_only_days_inside_span():
    idx, pos = select_frame_indices(DAYS, ANCHOR, "history", 3, window_days=365)
    assert list(DAYS[idx]) == [100, 300, 400]
    assert idx[pos] == ANCHOR


def test_window_too_short_reports_not_eligible():
    with pytest.raises(ValueError, match="NOT_ELIGIBLE"):
        select_frame_indices(DAYS, ANCHOR, "history", 3, window_days=180)


def test_window_is_history_only():
    with pytest.raises(ValueError, match="history"):
        select_frame_indices(DAYS, ANCHOR, "multi", 3, window_days=365)


def test_window_config_passes_preflight_with_shared_members():
    result = preflight.audit_config("configs/segformer/v4r2_temporal_window/window180.yaml")
    assert result["counts"] == {"train": 2108, "dev": 669}
    assert result["temporal"]["window_days"] == 180
    assert result["temporal"]["requested_n_frames"] == 3
    assert result["budget"]["stop_updates"] == 2000

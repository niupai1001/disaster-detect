import numpy as np
import pytest
from landslide_seg.data.temporal_v6 import select_frame_indices


@pytest.mark.parametrize("n", [1, 3, 5, 7, 15])
def test_count_has_anchor_no_duplicates_and_order(n):
    days = np.arange(15)[::-1] * 11
    idx, pos = select_frame_indices(days, 8, "multi", n)
    assert len(idx) == len(set(idx)) == n
    assert idx[pos] == 8
    assert np.all(np.diff(days[idx]) >= 0)
    if n == 1:
        assert idx.tolist() == [8]
    if n == 15:
        assert set(idx) == set(range(15))


def test_history_count_never_reads_future_and_fails_if_short():
    days = np.array([50, 10, 40, 20, 30])
    idx, pos = select_frame_indices(days, 2, "history", 3)
    assert idx[pos] == 2 and max(days[idx]) == 40 and len(idx) == 3
    with pytest.raises(ValueError, match="NOT_ELIGIBLE"):
        select_frame_indices(days, 1, "history", 3)


@pytest.mark.parametrize("n", [0, -1, 2.5, True])
def test_reject_invalid_count(n):
    with pytest.raises(ValueError):
        select_frame_indices(np.arange(15), 0, "multi", n)

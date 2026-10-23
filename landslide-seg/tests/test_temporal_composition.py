import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/audit_temporal_composition.py"
spec = importlib.util.spec_from_file_location("temporal_composition", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def frames():
    return [{"frame_id": str(i), "date": f"2020-01-{i+1:02d}", "anchor": str(i == 3)} for i in range(7)]


def test_post_controls_keep_anchor_shared_subset_and_dates():
    rows = frames()
    five = module.select_post(rows, 5)
    three = module.select_post(rows, 3)
    assert len(set(five)) == 5 and len(set(three)) == 3
    assert 3 in three and set(three) <= set(five)
    assert module.select_post(rows[::-1], 5) == five


def test_post_reference_must_be_present():
    with pytest.raises(ValueError):
        module.select_post([r for r in frames() if r["anchor"] == "False"], 3)

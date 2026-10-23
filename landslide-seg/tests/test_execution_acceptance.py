import copy
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/audit_research_execution.py"
spec = importlib.util.spec_from_file_location("execution_acceptance", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def report():
    return {"checkpoint_sha256": "weights", "val_event_mean_iou": .3,
            "val": {"n_samples": 2, "by_event": {"hiroshima": {"pixel": {"iou": .2}},
                                                 "thrissur": {"pixel": {"iou": .4}}}}}


def test_valid_members_and_event_average():
    audit.validate_temporal_evaluation(report(), "weights", {"a", "b"},
                                       [{"sample_id": "a"}, {"sample_id": "b"}])


@pytest.mark.parametrize("change", ["hash", "members", "duplicate", "average", "event"])
def test_reject_same_count_wrong_identity(change):
    ev = copy.deepcopy(report())
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]
    if change == "hash":
        ev["checkpoint_sha256"] = "other"
    elif change == "members":
        rows[1]["sample_id"] = "c"
    elif change == "duplicate":
        rows[1]["sample_id"] = "a"
    elif change == "average":
        ev["val_event_mean_iou"] = .4
    else:
        ev["val"]["by_event"]["other"] = ev["val"]["by_event"].pop("thrissur")
    with pytest.raises(AssertionError):
        audit.validate_temporal_evaluation(ev, "weights", {"a", "b"}, rows)

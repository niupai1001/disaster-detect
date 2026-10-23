import importlib.util
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/evaluate_temporal.py"
spec = importlib.util.spec_from_file_location("temporal_cost", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_best_checkpoint_elapsed_is_not_total_training_cost(tmp_path):
    (tmp_path / "train_summary.json").write_text(json.dumps({"complete": True, "seconds": 1000}))
    d = module.training_cost(tmp_path, {"elapsed_seconds": 150})
    assert d["train_seconds"] == 1000
    assert d["selected_checkpoint_elapsed_seconds"] == 150


def test_incomplete_training_does_not_claim_full_cost(tmp_path):
    (tmp_path / "train_summary.json").write_text(json.dumps({"complete": False, "seconds": 500}))
    assert module.training_cost(tmp_path, {"elapsed_seconds": 150})["train_seconds"] is None


def test_missing_summary_keeps_selected_checkpoint_time_only(tmp_path):
    d = module.training_cost(tmp_path, {"elapsed_seconds": 150})
    assert d["train_seconds"] is None and d["selected_checkpoint_elapsed_seconds"] == 150

import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def module(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("frame_count_summary", scripts / "summarize_frame_count.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "outputs/segformer/v4r1_temporal_frame_count")
    assets = tmp_path / "docs/assets/research"
    assets.mkdir(parents=True)
    (assets / "execution_audit.json").write_text(json.dumps({"runs": [{
        "run_dir": "outputs/segformer/v4r1_temporal_control/SegFormer-B0-Temporal-Single/1234",
        "status": "COMPLETED_VALID_FOR_AUDITED_SCOPE", "checkpoints": {}}]}))
    return module


def test_complete_label_without_actual_evaluations_is_not_formal(module):
    rows = module.collect()["runs"]
    assert [r["n_frames"] for r in rows] == [1, 3, 5, 7, 15]
    assert not any(r["formal_result"] for r in rows)
    assert rows[0]["status"] == "评价缺失或变动，待重新验收"


def test_live_partial_json_line_does_not_destroy_progress(module):
    run = module.OUT / "SegFormer-B0-Temporal-Date3/1234"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text('{"global_step":1200}\n{"global_step":')
    row = next(r for r in module.collect()["runs"] if r["n_frames"] == 3)
    assert row["global_step"] == 1200
    assert not row["formal_result"]

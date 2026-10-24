"""组成对照：逐样本帧计划的成员、共享帧关系与数据集取帧。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.data.segformer_dataset import load_stats  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6  # noqa: E402

LOCK = json.loads((ROOT / "manifests/v4r3/subset_protocol_composition.json").read_text(encoding="utf-8"))
PLANS = {name: json.loads((ROOT / meta["path"]).read_text(encoding="utf-8"))
         for name, meta in LOCK["plans"].items()}
ROWS = [json.loads(x) for x in
        (ROOT / "manifests/v4r3/pool_composition.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()]


def test_shared_members_and_frame_relations():
    assert LOCK["counts"] == {"train": 1141, "dev": 669}
    for sid, post5 in PLANS["post5"].items():
        assert set(PLANS["post3"][sid]) <= set(post5)
        assert set(PLANS["pre2_post3"][sid]) & set(post5) == set(PLANS["post3"][sid])
        assert len(set(PLANS["pre2_post3"][sid]) - set(PLANS["post3"][sid])) == 2


def test_plan_declares_retrospective_future_frames():
    assert LOCK["plans"]["post5"]["allows_future"] is True
    assert LOCK["plans"]["post5"]["n_future_frames"] > 0


def _dataset(mode="multi", plan="manifests/v4r3/plans/post3.json"):
    return TemporalV6(ROWS[:2], load_stats(ROOT / "manifests/v4r1/stats_core25.json"),
                      mode=mode, plan=plan, root="data/s2_temporal_v4r1",
                      quality="data/s2_temporal_v4r1_none")


def test_dataset_uses_plan_frames():
    ds = _dataset()
    item = ds[0]
    expected = PLANS["post3"][ROWS[0]["sample_id"]]
    assert item["image"].shape[0] == len(expected)
    assert sorted(item["frame_index"].tolist()) == sorted(expected)


def test_history_mode_rejects_future_plan_frames(tmp_path):
    index = json.loads((ROOT / "data/s2_temporal_v4r1/index.json").read_text(encoding="utf-8"))
    dates = {r["sample_id"]: r["dates"] for r in index["rows"]}
    sid = ROWS[0]["sample_id"]
    anchor = int(ROWS[0]["time_index"])
    later = next(i for i, d in enumerate(dates[sid]) if d > ROWS[0]["date"])
    plan = {r["sample_id"]: [int(r["time_index"])] for r in ROWS[:2]}
    plan[sid] = [anchor, later]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="参考日期之后"):
        _dataset(mode="history", plan=str(path))[0]


def _preflight():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "research_preflight", ROOT / "scripts/check_research_protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_rejects_plan_with_wrong_length():
    with pytest.raises(ValueError, match="帧计划长度"):
        _preflight().audit_config("configs/segformer/v4r3_composition/post5.yaml",
                                  ["segformer.temporal.plan=manifests/v4r3/plans/post3.json"])


def test_preflight_requires_multi_mode_for_future_frames():
    with pytest.raises(ValueError, match="mode=multi"):
        _preflight().audit_config("configs/segformer/v4r3_composition/post5.yaml",
                                  ["segformer.temporal.mode=history"])

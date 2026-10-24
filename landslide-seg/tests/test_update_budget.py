"""预算适配：停止步数与学习率日程分开，且 2000 步确实是旧 3600 步日程的前缀。"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.training.budget import poly_lr_scale, resolve_update_budget  # noqa: E402

spec = importlib.util.spec_from_file_location("research_preflight", ROOT / "scripts/check_research_protocol.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

CUTOFF7 = "configs/segformer/v4r2_temporal_cutoff/frames7.yaml"


def _scales(budget, warmup_frac=0.05, steps=2000):
    warmup = budget.warmup_updates(warmup_frac)
    return [poly_lr_scale(k, schedule_updates=budget.schedule_updates,
                          warmup_updates=warmup) for k in range(steps)]


def test_short_budget_is_prefix_of_old_schedule():
    old = resolve_update_budget({"max_steps": 3600, "eval_every_updates": 100})
    new = resolve_update_budget({"max_steps": 2000, "stop_updates": 2000,
                                 "schedule_updates": 3600, "eval_every_updates": 100})
    assert old.warmup_updates(0.05) == new.warmup_updates(0.05) == 180
    assert _scales(old) == _scales(new)


def test_defaults_keep_legacy_behaviour():
    budget = resolve_update_budget({"max_steps": 3600, "eval_every_updates": 100})
    assert (budget.stop_updates, budget.schedule_updates) == (3600, 3600)
    assert budget.report_steps == (3600,)


@pytest.mark.parametrize("train", [
    {"max_steps": 2000, "stop_updates": 2000, "schedule_updates": 1800},          # 日程短于停止
    {"max_steps": 3600, "stop_updates": 2000, "schedule_updates": 3600},          # 停止与 max_steps 矛盾
    {"max_steps": 2000, "schedule_updates": 3600, "report_steps": [1000, 2050]},  # 超出停止步
    {"max_steps": 2000, "schedule_updates": 3600, "report_steps": [1050]},        # 不是评估间隔倍数
])
def test_inconsistent_budget_rejected(train):
    with pytest.raises(ValueError):
        resolve_update_budget({**train, "eval_every_updates": 100})


def test_report_steps_always_include_stop():
    budget = resolve_update_budget({"max_steps": 2000, "schedule_updates": 3600,
                                    "eval_every_updates": 100, "report_steps": [1000]})
    assert budget.report_steps == (1000, 2000)


def test_cutoff_subset_config_passes_preflight():
    result = preflight.audit_config(CUTOFF7)
    assert result["counts"] == {"train": 2007, "dev": 669}
    assert result["subset"]["min_frames_at_or_before_reference"] == 7
    assert result["budget"]["stop_updates"] == 2000
    assert result["budget"]["schedule_updates"] == 3600
    assert result["budget"]["warmup_updates"] == 180


def test_subset_cannot_point_at_other_manifest():
    with pytest.raises(ValueError, match="Configured manifest"):
        preflight.audit_config(CUTOFF7, ["segformer.data.manifest=manifests/v4r1/pool_100.jsonl"])

"""U-TAE 时间对照的验收汇总：用已验收的 15 期日期组校准口径。"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_utae_time", ROOT / "scripts/summarize_utae_time.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_reference_run_passes_and_reports_event_mean():
    cfg = summary.load_cfg(ROOT / "configs/utae/v4r1/multi.yaml")
    record = summary.check_run(ROOT / "outputs/utae_v4r1/v4r1/multi/1234", cfg)
    assert record["formal_result"], record["errors"]
    best = record["evals"]["best"]
    # 圆整后与台账的 0.3706（事件等权）一致；像素汇总口径是 0.3692，不能混用
    assert round(best["event_mean_iou"], 4) == 0.3706
    assert round(best["pooled_iou"], 4) == 0.3692
    assert set(best["by_event"]) == {"hiroshima", "thrissur"}


def test_unstarted_run_is_not_formal():
    record = summary.check_run(
        ROOT / "outputs/utae_v4r1_controls/does_not_exist/v4r1/multi/1234",
        {"train": {"epochs": 30}}, None)
    assert not record["formal_result"]
    assert any("missing" in e for e in record["errors"])

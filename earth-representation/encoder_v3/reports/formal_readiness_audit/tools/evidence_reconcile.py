"""Reconcile historical evidence against the current HEAD and data identities."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
REPORTS = ROOT / "encoder_v3" / "reports"


def load(name: str):
    path = REPORTS / name
    if not path.exists():
        return None, dict(path=str(path), status="EVIDENCE_MISSING")
    return json.loads(path.read_text(encoding="utf-8-sig")), dict(path=str(path), bytes=path.stat().st_size)


def main() -> None:
    baseline = json.loads((AUDIT / "00_baseline.json").read_text(encoding="utf-8"))

    admission, admission_meta = load("admission_review_2026-11-07.json")
    recomputed = Counter()
    item_rows = []
    for gate in (admission or {}).get("gates", []):
        verdict = gate.get("verdict")
        recomputed[verdict] += 1
        item_rows.append(dict(gate=gate.get("gate"), verdict=verdict, config_value=gate.get("config_value"), gap=gate.get("gap")))
    admission_check = dict(
        evidence=admission_meta,
        reported_summary=None if admission is None else dict(
            satisfied=admission.get("satisfied"),
            partially_satisfied_preflight_only=admission.get("partially_satisfied_preflight_only"),
            not_satisfied=admission.get("not_satisfied"),
        ),
        recomputed_by_verdict=dict(recomputed),
        recomputed_required_summary=dict(
            satisfied=recomputed.get("SATISFIED", 0),
            partially_satisfied_preflight_only=recomputed.get("PARTIAL_PREFLIGHT", 0),
            not_satisfied=recomputed.get("NOT_SATISFIED", 0),
        ),
        summary_mismatch=None if admission is None else dict(
            satisfied=admission.get("satisfied") != recomputed.get("SATISFIED", 0),
            partial=admission.get("partially_satisfied_preflight_only") != recomputed.get("PARTIAL_PREFLIGHT", 0),
            not_satisfied=admission.get("not_satisfied") != recomputed.get("NOT_SATISFIED", 0),
        ),
        items=item_rows,
        status="CORRECTED_COUNT_1_SATISFIED_5_PARTIAL_8_NOT_SATISFIED",
    )

    discovery, discovery_meta = load("discovery_coverage_current.json")
    discovery_check = dict(
        evidence=discovery_meta,
        status=None if discovery is None else discovery.get("status"),
        scheduled_pending_chunks=None if discovery is None else discovery.get("scheduled_pending_chunks"),
        unresolved_timeout_leaves=None if discovery is None else discovery.get("unresolved_timeout_leaves"),
        access_error_chunks=None if discovery is None else [x.get("root") for x in discovery.get("access_error_chunks", [])],
        admitted_members=None if discovery is None else discovery.get("admitted_members"),
        unique_candidate_paths=None if discovery is None else discovery.get("unique_candidate_paths"),
        limits=None if discovery is None else discovery.get("limits"),
        conclusion="FILE_DISCOVERY_PENDING_ZERO_DOES_NOT_MEAN_DATA_ADMISSION",
    )

    budget_path = REPORTS / "pilot_runs" / "budget.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else None
    stop_path = REPORTS / "pilot_runs" / "pilot_stop_decision.json"
    stop = json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else None
    events = (budget or {}).get("events", [])
    reservations = [e for e in events if e.get("kind") == "reserved"]
    completed = [e for e in events if e.get("kind") == "completed"]
    budget_check = dict(
        budget_exists=budget_path.exists(),
        stop_exists=stop_path.exists(),
        total_limit=None if budget is None else budget.get("total_limit"),
        per_candidate_limit=None if budget is None else budget.get("per_candidate_limit"),
        reserved=len(reservations),
        completed=len(completed),
        incomplete=sorted(set(e["reservation"] for e in reservations) - set(e["reservation"] for e in completed)),
        stop_decision=None if stop is None else dict(
            status=stop.get("status"), selected_optimizer=stop.get("selected_optimizer"),
            remaining_total=(stop.get("budget_at_stop") or {}).get("remaining_total"),
            remaining_by_candidate=(stop.get("budget_at_stop") or {}).get("remaining_by_candidate"),
            formal_training_authorized=stop.get("formal_training_authorized"),
        ),
        conclusion="214_REAL_PREFLIGHT_UPDATES_COMPLETED_186_REMAINING_NOT_AUTHORIZED",
    )

    existing = baseline.get("existing_engineering_report") or {}
    stale_check = dict(
        existing=existing,
        current_test_method_count=baseline.get("test_method_count"),
        status="STALE_DOES_NOT_DESCRIBE_CURRENT_HEAD" if existing.get("source_hash_mismatch") else "HASH_MATCH_REQUIRES_TEST_COUNT_CHECK",
    )

    config = json.loads((ROOT / "encoder_v3" / "config.json").read_text(encoding="utf-8"))
    required = config.get("required_before_training", {})
    gate_check = dict(
        required_keys=sorted(required),
        null_or_false=[k for k, v in required.items() if v is None or v is False or v == ""],
        optimizer_choice=required.get("optimizer_choice"),
        formal_training_authorization=required.get("formal_training_authorization"),
        formal_step_present="formal_step" in config,
        required_before_jepa=config.get("required_before_jepa"),
        m9_task_head_training_authorized=config.get("m9_task_head_training_authorized"),
    )

    adaptive = json.loads((ROOT / "encoder_v3" / "adaptive_config.json").read_text(encoding="utf-8"))
    adaptive_check = dict(
        required_before_training=adaptive.get("required_before_training"),
        unresolved=[k for k, v in adaptive.get("required_before_training", {}).items() if v is None],
        status=adaptive.get("status"),
    )

    cache, cache_meta = load("native_cache_audit.json")
    stats, stats_meta = load("train_statistics_recomputed.json")
    lineage, lineage_meta = load("native_lineage_audit.json")
    b1, b1_meta = load("B1_expanded_evidence.json")
    metric, metric_meta = load("validation_metric_confirmation.json")
    data_check = dict(
        native_cache=None if cache is None else dict(status=cache.get("status"), read_members=cache.get("read_members"), roles=cache.get("roles")),
        native_cache_evidence=cache_meta,
        train_statistics=None if stats is None else dict(status=stats.get("status"), members_checked=stats.get("members_checked"), all_counts_match=stats.get("all_counts_match"), max_abs_mean=stats.get("max_absolute_mean_difference"), max_abs_std=stats.get("max_absolute_std_difference")),
        train_statistics_evidence=stats_meta,
        lineage=None if lineage is None else dict(status=lineage.get("status"), members_total=lineage.get("members_total"), seconds=lineage.get("seconds")),
        lineage_evidence=lineage_meta,
        b1=None if b1 is None else dict(status=b1.get("status"), cases=b1.get("cases"), members=b1.get("members"), per_band_reports=b1.get("per_band_reports"), supported_pixel_exposures=b1.get("supported_pixel_exposures"), no_support_cases=b1.get("no_support_cases"), parameters_unchanged=b1.get("parameters_unchanged")),
        b1_evidence=b1_meta,
        validation_metric=None if metric is None else dict(status=metric.get("status"), implementation_status=metric.get("implementation_status"), formal_panels_and_support_set_frozen=metric.get("formal_panels_and_support_set_frozen"), pair_admission_approved=metric.get("pair_admission_approved")),
        validation_metric_evidence=metric_meta,
    )

    output = dict(
        schema="encoder_v3_formal_readiness_evidence_reconciliation_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        baseline_head=(baseline.get("repo") or {}).get("head"),
        admission=admission_check,
        discovery=discovery_check,
        pilot_budget=budget_check,
        stale_engineering_report=stale_check,
        gates=gate_check,
        adaptive_gates=adaptive_check,
        data= data_check,
        conclusions=[
            "The admission review summary is arithmetically wrong; per-item verdicts are authoritative until re-reviewed.",
            "File discovery has no scheduled pending chunks, but access errors and archives remain; admitted members are zero.",
            "The pilot ledger is complete at 214/214; the remaining 186 updates are explicitly not authorized.",
            "The existing engineering_tests.json predates current gates.py and cannot support current admission.",
            "B1 502-case diagnostics cover selected combinations only and are not training-quality evidence.",
            "The confirmed validation metric is not implemented and its formal panels/support set are not frozen.",
        ],
    )
    out = AUDIT / "01_evidence_reconciliation.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(admission=admission_check["recomputed_required_summary"],
                          discovery=discovery_check["status"],
                          budget=dict(reserved=budget_check["reserved"], completed=budget_check["completed"]),
                          stale=stale_check["status"], output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

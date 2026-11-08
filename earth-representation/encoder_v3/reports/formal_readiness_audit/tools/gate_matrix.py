"""Build the 14-gate + extra/JEPA admission matrix; no approval is granted."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from encoder_v3.gates import ADAPTIVE_IDS, GATE_IDS, JEPA_IDS, payload_sha256, require_training

ROOT = Path(__file__).resolve().parents[4]
ENC = ROOT / "encoder_v3"
AUDIT = ENC / "reports" / "formal_readiness_audit"
REPORTS = ENC / "reports"


def research_verdict():
    return {
        "optimizer_choice": ("SATISFIED", "AdamW selected; pilot stop decision records the user instruction.", "reports/pilot_runs/pilot_stop_decision.json"),
        "merge_configuration": ("PARTIAL_PREFLIGHT", "Single-round B and native/merged scales confirmed; GSD->r curve is preflight-only and the P/mask relation is unresolved.", "reports/merge_B_confirmation.json; reports/merge_levels_confirmation.json; reports/merge_rate_confirmation.json"),
        "hypernetwork_hidden_structure": ("PARTIAL_PREFLIGHT", "34->64->320/GELU/log-GSD frozen for preflight only.", "reports/hypernetwork_preflight_review.json"),
        "frozen_data_manifest": ("NOT_SATISFIED", "No approved formal manifest; release has 7,395 TRAIN/383 VAL, no TEST/GUARD; QA policy for all members not approved.", "reports/native_cache_audit.json; reports/cross_role_split_audit.json; reports/discovery_coverage_current.json"),
        "pair_admission": ("NOT_SATISFIED", "Admitted B2 pairs are zero; offsets, calibration isolation, registration/change thresholds are unapproved.", "reports/pair_admission_evidence.json; reports/pair_spatial_holdout_disjoint.json"),
        "mask_rectangle_distribution": ("PARTIAL_PREFLIGHT", "Mask rule is preflight-only; rectangle-vs-Merge-unit semantics and P relation unresolved.", "reports/spatial_mask_preflight_review.json"),
        "validation_panel": ("PARTIAL_PREFLIGHT", "Metric confirmed and now implemented; formal panels/support set are not frozen; B2 absent and B1 multi counts incomplete.", "reports/validation_metric_confirmation.json; reports/formal_readiness_audit/04_task_audit.json"),
        "improvement_threshold": ("NOT_SATISFIED", "No approved numeric improvement threshold.", "none"),
        "stability_window": ("NOT_SATISFIED", "No approved stability window or validation-uncertainty rule.", "none"),
        "early_stop": ("NOT_SATISFIED", "No formal early-stop rule; only the stopped pilot record exists.", "reports/pilot_runs/pilot_stop_decision.json"),
        "precision_acceptance": ("PARTIAL_PREFLIGHT", "Only the prior FP32 single-GPU preflight limits are approved; new Participation path not accepted.", "reports/fp32_equivalence_tolerance_review.json"),
        "resource_limit": ("NOT_SATISFIED", "No formal update/time/memory/storage cap or failure budget.", "none"),
        "execution_equivalence": ("NOT_SATISFIED", "Corrected D10 (2 unique updates + 1 resume) is within old FP32 preflight bounds, but formal new-architecture tolerance/approval is absent; earlier overrun remains documented.", "reports/formal_readiness_audit/08_execution_audit.json; reports/formal_readiness_audit/evidence/controlled_update/incident.json"),
        "formal_training_authorization": ("NOT_SATISFIED", "Must be explicitly issued by the user; it is not.", "config.json"),
    }


def main():
    config = json.loads((ENC / "config.json").read_text(encoding="utf-8"))
    adaptive = json.loads((ENC / "adaptive_config.json").read_text(encoding="utf-8"))
    admission = json.loads((REPORTS / "admission_review_2026-11-07.json").read_text(encoding="utf-8"))
    verdicts = research_verdict()
    try:
        require_training(config)
        structured = "APPROVED"
    except ValueError as exc:
        structured = "NOT_APPROVED: " + str(exc)
    rows = []
    for gate in sorted(GATE_IDS):
        research, note, evidence = verdicts[gate]
        value = config.get("required_before_training", {}).get(gate)
        structured_gate = "APPROVED" if isinstance(value, dict) and value.get("status") == "APPROVED" and value.get("reviewer") == "user" else "NOT_APPROVED"
        rows.append(dict(gate=gate, configured_value=value,
                         research_verdict=research, structured_gate_status=structured_gate,
                         evidence=evidence, note=note, blocks_formal_training=True))
    structured_approved = sum(row["structured_gate_status"] == "APPROVED" for row in rows)
    extra = []
    for gate in sorted(ADAPTIVE_IDS):
        value = adaptive.get("required_before_training", {}).get(gate)
        extra.append(dict(gate=gate, configured_value=value,
                          status="NOT_APPROVED_OR_PREFLIGHT_ONLY",
                          blocks_formal_training=True))
    jepa = [dict(gate=gate, configured_value=(config.get("required_before_jepa") or {}).get(gate),
                 status="NOT_APPROVED", blocks_formal_training=True) for gate in sorted(JEPA_IDS)]
    counts = {"SATISFIED": 0, "PARTIAL_PREFLIGHT": 0, "NOT_SATISFIED": 0}
    for row in rows:
        counts[row["research_verdict"]] += 1
    matrix = dict(
        schema="encoder_v3_formal_readiness_gate_matrix_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        code=dict(branch=__import__("subprocess").run(["git", "-C", str(ROOT), "branch", "--show-current"], capture_output=True, text=True).stdout.strip(),
                  head=__import__("subprocess").run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                  config_sha256=__import__("hashlib").sha256((ENC / "config.json").read_bytes()).hexdigest(),
                  config_payload_sha256=payload_sha256(config),
                  adaptive_config_sha256=__import__("hashlib").sha256((ENC / "adaptive_config.json").read_bytes()).hexdigest()),
        admission_review_reported=dict(satisfied=admission.get("satisfied"),
                                       partial=admission.get("partially_satisfied_preflight_only"),
                                       not_satisfied=admission.get("not_satisfied")),
        admission_review_recomputed=counts,
        summary=dict(gate_count=14, research_satisfied=counts["SATISFIED"], partial=counts["PARTIAL_PREFLIGHT"],
                     not_satisfied=counts["NOT_SATISFIED"], structured_approved=structured_approved,
                     formal_training_authorized=config.get("required_before_training", {}).get("formal_training_authorization"),
                     task_head_authorized=config.get("m9_task_head_training_authorized")),
        gates=rows, adaptive_gates=extra, jepa_gates=jepa,
        structured_gate_check=structured,
        formal_step_present="formal_step" in config,
        verdict="NOT_READY",
        blockers=[
            "No formal data manifest or full-release QA approval", "B2 admitted pairs = 0",
            "Architecture/Merge/mask are formally approved; formal validation panel is not frozen",
            "No approved improvement/stability thresholds or formal resource policy",
            "Corrected D10 is within old FP32 preflight tolerance only; formal equivalence remains unapproved",
            "JEPA five conditions unapproved",
            "No formal training authorization",
            "Single-route runner and candidate schedule/panel builders exist; automatic two-route/JEPA orchestration remains missing",
        ],
    )
    out = AUDIT / "09_gate_matrix.json"
    out.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(counts=counts, structured=structured, verdict=matrix["verdict"], output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

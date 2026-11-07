"""Read-only collection of the evidence behind each training-admission gate.

Nothing is written back to the existing configuration or reports; this only
gathers what the repository already claims, so the review can be based on files
that exist rather than on memory.
"""

import json
from pathlib import Path

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation\encoder_v3")
REPORTS = ROOT / "reports"

EVIDENCE = {
    "merge_configuration": [
        "merge_B_confirmation.json", "merge_levels_confirmation.json",
        "merge_rate_confirmation.json", "merge_participation_confirmation.json",
        "participation_merge_confirmation.json",
    ],
    "hypernetwork_hidden_structure": [
        "hypernetwork_preflight_review.json", "condition_encoding_review.json",
    ],
    "frozen_data_manifest": [
        "physical_patch_role_review.json", "frozen_pair_source_review.json",
        "discovery_coverage_after_leaf_followup.json",
    ],
    "pair_admission": [
        "B2_QA_policy_confirmation.json", "B2_geometry_mechanism_confirmation.json",
        "B2_pair_geometry_proposal.md", "pair_spatial_holdout_disjoint.json",
        "B2_calibration_use_review.json",
    ],
    "mask_rectangle_distribution": [
        "spatial_mask_preflight_review.json",
    ],
    "validation_panel": [
        "pilot_members_panels_review.json", "validation_panels_review.json",
        "validation_metric_confirmation.json", "validation_metric_review.json",
    ],
    "improvement_threshold": [
        "validation_metric_confirmation.json", "pilot_stop_decision.json",
    ],
    "stability_window": [],
    "early_stop": [
        "pilot_runs/pilot_stop_decision.json",
    ],
    "precision_acceptance": [
        "fp32_equivalence_tolerance_review.json",
    ],
    "resource_limit": [
        "budget.json", "resource_limit_review.json", "gpu_update_execution_review.json",
    ],
    "execution_equivalence": [
        "fp32_equivalence_tolerance_review.json", "cuda_fault_event_evidence.json",
        "gpu_update_execution_review.json", "route_step_runtime_incident.json",
    ],
    "optimizer_choice": [
        "pilot_runs/pilot_stop_decision.json", "optimizer_decision_review.json",
    ],
    "formal_training_authorization": [
        "formal_training_authorization.json",
    ],
}


def peek(path):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_type": "non-json", "size": path.stat().st_size}
    keys = list(d)[:6] if isinstance(d, dict) else [f"list[{len(d)}]"]
    snippet = {}
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        snippet[k] = v if isinstance(v, (str, int, float, bool, type(None))) else type(v).__name__
    return snippet


def main():
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    gates = cfg["required_before_training"]
    print(f"gates={len(gates)}  satisfied={sum(1 for v in gates.values() if v not in (None, False, ''))}")
    for gate, value in gates.items():
        state = "SET" if value not in (None, False, "") else "EMPTY"
        print(f"\n[{gate}] config={state} value={value!r}")
        for name in EVIDENCE.get(gate, []):
            p = REPORTS / name
            info = peek(p)
            if info is None:
                print(f"   - {name}: MISSING")
            else:
                print(f"   - {name}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

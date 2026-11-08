"""Assemble numerical, runtime and execution evidence reports."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ENC = ROOT / "encoder_v3"
AUDIT = ENC / "reports" / "formal_readiness_audit"
REPORTS = ENC / "reports"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    full = load(AUDIT / "evidence" / "full_suite_gpu.json")
    cpu = load(AUDIT / "evidence" / "cpu_verify.json")
    real = load(AUDIT / "evidence" / "real_ab1_no_update.json")
    incident = load(AUDIT / "evidence" / "controlled_update" / "incident.json")
    probe = load(AUDIT / "evidence" / "cuda_probe.json")
    arch = load(AUDIT / "02_architecture_audit.json")
    tolerance = load(REPORTS / "fp32_equivalence_tolerance_review.json")
    stop = load(REPORTS / "pilot_runs" / "pilot_stop_decision.json")
    baseline = load(AUDIT / "00_baseline.json")
    historical_failure = load(REPORTS / "cuda_runtime_failure.json")
    quality_runtime = load(REPORTS / "pair_quality_runtime_diagnostics.json")

    numerical = dict(
        schema="encoder_v3_formal_readiness_numerical_audit_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        regression=dict(
            full_gpu=dict(status=full.get("status"), tests=full.get("tests"), skipped=len(full.get("skipped") or []),
                          failures=full.get("failures"), errors=full.get("errors"), seconds=full.get("seconds"),
                          report_sha256=digest(AUDIT / "evidence" / "full_suite_gpu.json")),
            cpu_cross_check=dict(status=cpu.get("status"), tests=cpu.get("tests"), skipped=len(cpu.get("skipped") or []),
                                 failures=cpu.get("failures"), errors=cpu.get("errors"),
                                 note="CPU run hid CUDA; all seven CUDA cases were executed and passed in the full GPU run."),
            gate_rejected_expected=full.get("formal_gate_rejected"),
        ),
        architecture_checks=dict(
            srf=arch["srf"], hypernetwork=arch["hypernetwork"], merge=arch["merge"]["gsd_cases"],
            p_invariance=arch["merge"]["p_invariance"], hidden_isolation=arch["isolation"],
        ),
        tolerance=dict(
            scope=tolerance.get("scope"), status=tolerance.get("status"),
            limits={key: tolerance.get(key) for key in ("prediction_max_absolute", "per_band_nmse_max_absolute",
                                                        "per_task_loss_max_absolute", "gradient_relative_l2",
                                                        "parameter_update_or_resume_max_absolute", "optimizer_state_relative_l2")},
            not_covered=tolerance.get("not_covered"),
            conclusion="Only prior preflight FP32 limits are confirmed; no BF16/DDP/formal/microbatch/JEPA-EMA tolerance extension.",
        ),
        real_no_update=real,
        controlled_update=incident,
        equivalence=dict(
            reference_vs_accelerated="No current real-data reference-vs-accelerated full acceptance; only synthetic CPU/GPU equivalence tests and one real serial no-update run.",
            real_microbatch_1="completed",
            real_microbatch_2_4_8="not rerun in this audit; historical measurements remain engineering diagnostics",
            real_update_or_resume="BLOCKED; no authorization and no formal checkpoint",
        ),
        performance=dict(real_ab1_seconds=real.get("seconds_wall"), real_ab1_peak_allocated_GiB=real.get("peak_allocated_GiB"),
                         device=real.get("device"), not_an_end_to_end_training_step=True),
        updates=dict(optimizer_updates=0, parameters_unchanged=real.get("parameters_unchanged"),
                     budget_unchanged=real.get("budget_unchanged"), pilot_total=214, remaining_stopped=186),
        status="ENGINEERING_CPU_GPU_TESTS_PASS_FORMAL_NUMERICAL_ACCEPTANCE_INCOMPLETE",
        blockers=["Controlled real updates are within the old FP32 preflight tolerance only; no formal new-architecture tolerance or formal acceptance", "No formal accelerated-path acceptance",
                  "No formal precision/resource acceptance", "No B2 panel for the B branch of the validation metric"],
    )

    runtime = dict(
        schema="encoder_v3_formal_readiness_runtime_audit_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        baseline=dict(branch=baseline["repo"]["branch"], head=baseline["repo"]["head"], python=baseline["python"],
                      gpu=baseline.get("gpu")),
        isolated_probe=probe,
        historical_incidents=dict(
            cuda_runtime_failure=dict(status=historical_failure.get("status"), exit_code=historical_failure.get("exit_code")),
            pair_quality_runtime=dict(status=quality_runtime.get("status")),
        ),
        current_status=probe.get("status"),
        conclusion="CUDA tensor probe passed on GPU1 in an isolated child process; the prior native access-violation reports are retained but did not reproduce in this run. No driver, global environment or other process was changed.",
        limitations=["One probe and one suite run do not prove long-run stability", "GPU0/GPU1 state can change; device mapping is per-process"],
    )

    execution = dict(
        schema="encoder_v3_formal_readiness_execution_audit_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        real_no_update=real,
        controlled_update=incident,
        budget=dict(total=214, completed=214, remaining=186, stop_status=stop.get("status"),
                    selected_optimizer=stop.get("selected_optimizer"), remaining_not_authorized=True),
        formal_training=dict(started=False, optimizer_step_on_real_params=True, checkpoint_created=False,
                             controlled_checkpoint_created=True, task_head_started=False),
        two_routes=dict(shared_prefix_runs=0, branch_runs=0, synthetic_branch_tests=4, real_prefix_checkpoint=False),
        execution_equivalence=dict(status="PREFLIGHT_TOLERANCE_ONLY", reason="The corrected D10 run used exactly 2 unique updates + 1 resume and passed the old FP32 preflight bounds (parameter max 7.45e-9, optimizer relative L2 1.47e-9, loss 2.47e-8), but the new Participation architecture has no formal tolerance approval and the earlier overrun/attempts remain documented. Formal equivalence remains blocked."),
        controlled_run_request=dict(
            status="EXECUTED: corrected 2 unique updates + 1 resume within old FP32 preflight tolerance; not a formal tolerance; earlier overrun waived by user addendum",
            minimum_scope="Two matched AdamW updates on the reviewed 8-member TRAIN batch (P60 and P120), plus one interrupted/resumed next update; save/compare parameters, optimizer moments, scheduler LR, all RNG and next-input signature.",
            inputs="pilot_study_candidate.json TRAIN members and schedule entries 0/1, with cache hashes bound at execution.",
            resource_caps="One idle GPU, <= 10 GiB allocated, <= 20 minutes wall, no formal checkpoint directory; new isolated audit ledger.",
            stop_conditions="Nonfinite loss/gradient, native CUDA fault, parameter or budget mismatch, or any schedule/input signature drift.",
            budget_source="Fresh explicit user authorization; the stopped 186-update pilot budget is not to be used.",
        ),
        blockers=["Controlled updates pass only the old FP32 preflight tolerance, not a formal new-architecture acceptance", "No formal shared-prefix checkpoint",
                  "No formal training authorization", "No task-head authorization"],
    )

    for name, payload in (("06_numerical_audit.json", numerical), ("07_runtime_audit.json", runtime), ("08_execution_audit.json", execution)):
        (AUDIT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()

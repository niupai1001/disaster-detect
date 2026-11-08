"""Read-only training pipeline, recovery and route audit."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ENC = ROOT / "encoder_v3"
AUDIT = ENC / "reports" / "formal_readiness_audit"


def calls(pattern: str):
    hits = []
    for path in sorted(ENC.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        for match in re.finditer(pattern, path.read_text(encoding="utf-8")):
            hits.append(dict(file=path.name, line=path.read_text(encoding="utf-8")[:match.start()].count("\n") + 1, match=match.group(0)))
    return hits


def main():
    config = json.loads((ENC / "config.json").read_text(encoding="utf-8"))
    schedule = json.loads((ENC / "reports" / "pilot_study_candidate.json").read_text(encoding="utf-8"))
    real = json.loads((AUDIT / "evidence" / "real_ab1_no_update.json").read_text(encoding="utf-8"))
    incident = json.loads((AUDIT / "evidence" / "controlled_update" / "incident.json").read_text(encoding="utf-8"))
    validation = json.loads((AUDIT / "03_data_audit.json").read_text(encoding="utf-8"))
    route_step = (ENC / "route_step.py").read_text(encoding="utf-8")
    route_control = (ENC / "route_control.py").read_text(encoding="utf-8")
    route_branch = (ENC / "route_branch.py").read_text(encoding="utf-8")

    report = dict(
        schema="encoder_v3_formal_readiness_pipeline_audit_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_hashes={name: __import__("hashlib").sha256((ENC / name).read_bytes()).hexdigest() for name in
                       ("route_step.py", "route_control.py", "route_branch.py", "training.py", "scheduled_data.py", "validation_panels.py", "validation_metric.py")},
        runner=dict(
            formal_runner_module=(ENC / "formal_runner.py").exists(),
            execute_route_update_callers=calls(r"execute_route_update\("),
            initialize_route_branch_callers=calls(r"initialize_route_branch\("),
            frozen_schedule_loader_callers=calls(r"FrozenABSchedule\("),
            frozen_panel_loader_callers=calls(r"FrozenValidationPanels\("),
            implemented="formal_runner.run_route composes schedule.prepare -> move_views -> execute_route_update -> next input signature; it enforces explicit max_updates/max_wall_seconds and stops without replay.",
            still_missing="No automatic shared-prefix-to-two-route orchestration, no JEPA runner path, and no approved formal schedule/panel bound to a frozen manifest; loader-compatible candidate schedule/panel builders exist.",
            conclusion="Single-route production control loop and candidate schedule/panel builders exist and are tested with synthetic/injected inputs; real formal integration remains blocked by data/schedule/panel/authorization decisions.",
        ),
        schedule=dict(
            pilot_status=schedule.get("status"), pilot_scope=schedule.get("scope"),
            pilot_updates=len(schedule.get("schedule", [])), pilot_members=len(schedule.get("training_members", [])),
            formal_schedule_sha_in_config=None if not config.get("required_before_training") else config["required_before_training"].get("frozen_data_manifest"),
            formal_schedule_builder="formal_schedule.build_formal_candidate_schedule produces a loader-compatible candidate from explicit 8 members, P sequence and reviewed mask rule; formal_panels.build_validation_panel_document does the same for explicit validation cases; neither is an approved formal set.",
        ),
        scheduler=dict(
            route_schedule_explicit=True,
            route_step_requires=dict(
                precision="fp32", chunk_pixels="int", microbatch_observations="1/2/4/8", clip_norm="positive",
                peak_lr="explicit", plateau_threshold="explicit", weight_decay="explicit", betas="explicit",
                eps="explicit", ema_coefficient="explicit",
            ),
            route_schedule_controls=dict(warmup_updates="explicit 100 default", validation_interval="explicit 250 default", max_updates="explicit 10000 default", plateau_patience=4, lr_factor=.5, min_lr=1e-5),
            formal_step_present="formal_step" in config,
            improvement_threshold=config.get("required_before_training", {}).get("improvement_threshold"),
            stability_window=config.get("required_before_training", {}).get("stability_window"),
            early_stop=config.get("required_before_training", {}).get("early_stop"),
            resource_limit=config.get("required_before_training", {}).get("resource_limit"),
            conclusion="Warmup/validation/update limits and a finite-metric early-stop hook are explicit constructor/runner parameters; their values, the stability window and the formal resource policy remain unfrozen.",
        ),
        validation=dict(
            metric_implemented=(ENC / "validation_metric.py").exists(),
            panel_loader="FrozenValidationPanels requires formal_config_sha256, panel sha256 and exact VAL release; formal_panels.build_validation_panel_document builds a candidate document from explicit cases but no formal panel is frozen.",
            b2_available=validation["existing_evidence"]["b2"]["admitted_pairs"] > 0 if validation.get("existing_evidence") else None,
            formal_scalar_status="B2_UNAVAILABLE unless an admitted B2 panel exists; strict B1 multi-count coverage required.",
        ),
        checkpoint=dict(
            save_fields=["modules", "optimizer", "scheduler", "logical_update", "sampler", "frozen_ids", "rng"],
            route_metadata=["_route_progress", "_route_optimizer_names", "_route_last_metrics"],
            atomic_write="temp + fsync + os.replace; exclusive lock",
            synthetic_tests="route/control/branch/transaction/runner and schedule/panel-builder tests pass; current full suite is 222 tests on GPU1 with zero skips",
            real_update_resume="Corrected D10 run (2 unique updates + 1 resume) is within the old FP32 preflight tolerance; not a formal new-architecture acceptance. Earlier overrun/attempts remain documented in evidence/controlled_update/incident.json.",
        ),
        routes=dict(
            shared_prefix_requirement="RouteProgress.fork requires a certified shared checkpoint at >=2000 updates",
            two_route_tests="synthetic branch tests preserve Encoder/Decoder/optimizer/scheduler/RNG and prevent A/B stream changes",
            real_prefix="No approved shared-prefix checkpoint exists",
            jepa="five required_before_jepa entries are null; JEPA branch must remain disabled",
        ),
        execution=dict(
            real_no_update=real,
            controlled_update=incident,
            runtime="Isolated CUDA tensor probe passed on GPU1; prior native access-violation reports retained as historical; no driver/global changes made.",
            update_budget="214/214 completed; 186 remaining explicitly stopped and not used.",
        ),
        blockers=[
            "No automatic two-route orchestration or JEPA runner path; real integration still needs an approved schedule/panel prefix.",
            "No approved formal schedule/panel; candidate loader-compatible builders exist but no formal set is frozen.",
            "Formal improvement/stability thresholds and resource policy are not frozen; early-stop and update/wall limits are explicit runner parameters.",
            "Corrected D10 run is within old FP32 preflight tolerance only; no formal new-architecture acceptance.",
            "No approved shared-prefix checkpoint or two-route formal run.",
        ],
    )
    out = AUDIT / "05_training_pipeline_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(runner=report["runner"]["conclusion"], blockers=report["blockers"], output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

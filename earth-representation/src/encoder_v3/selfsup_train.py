"""Gated E0 training entry; it refuses to step while the optical gates are open.

    python -m encoder_v3.selfsup_train --help
    python -m encoder_v3.selfsup_train --config encoder_v3/config_optical_selfsup_v1.json --mode check
    python -m encoder_v3.selfsup_train --config ... --mode train --device cuda:1 --directory <run dir>

`train` requires require_optical_selfsup_training(); `check` only verifies the
configuration, data identity, schedule, panel and environment. No mode here is
invoked automatically by the preflight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .formal_runner import panel_validator, run_route
from .gates import require_optical_selfsup_training
from .route_control import RouteProgress, frozen_identity, inspect_route_checkpoint, restore_route_checkpoint
from .selfsup_preflight import build_context, configuration_sha256, gate_report, parameter_digest


def _control(config):
    control = config["control"]
    settings = config["formal_step"]
    early = control["early_stop"]
    if int(control["plateau_patience"]) != 4 or float(control["lr_factor"]) != 0.5 or float(control["min_lr"]) != 1e-5:
        raise ValueError("E0 scheduler candidate differs from the reviewed RouteSchedule constants")
    return dict(max_updates=int(control["max_updates"]), max_wall_seconds=float(control["max_wall_seconds"]),
                cell_indexed=bool(config["model"]["cell_indexed_decoder"]), retain_layouts=bool(control["retain_layouts"]),
                early_stop=dict(patience=int(early["patience"]), min_delta=float(early["min_delta"])))


def summarize(context):
    config = context["config"]
    return dict(
        mode="check",
        status="TRAIN_ENTRY_CONFIGURED",
        experiment=config["experiment"],
        configuration_sha256=configuration_sha256(context["config_path"]),
        frozen_identity=frozen_identity(context["frozen_ids"]),
        schedule_updates=len(context["schedule"]),
        panel_cases=len(context["panels"]),
        parameters=sum(parameter.numel() for module in context["modules"].values() for parameter in module.parameters()),
        control=_control(config),
        gates=gate_report(config),
        required_real_update_acceptance="reports/optical_selfsup_closure/09_remaining_decisions.md",
        optimizer_updates=0,
    )


def train(context, *, directory, device, resume=None):
    config = context["config"]
    require_optical_selfsup_training(config)
    control = _control(config)
    settings = config["formal_step"]
    modules, optimizer, scheduler = context["modules"], context["optimizer"], context["scheduler"]
    panels = context["panels"]
    gate = require_optical_selfsup_training
    validation = panel_validator(panels, cell_indexed=bool(config["validation"]["cell_indexed"]),
                                 retain_layouts=control["retain_layouts"], metric_key="S_intra")
    if resume is None:
        progress = RouteProgress.begin(context["frozen_ids"])
        expected_input_signature = None
        restored_state = None
    else:
        state, progress = restore_route_checkpoint(resume, modules=modules, optimizer=optimizer, scheduler=scheduler,
                                                   frozen_ids=context["frozen_ids"], phase="shared",
                                                   prefix_checkpoint_sha256=None)
        expected_input_signature = state["sampler"]["next_input_signature"]
        restored_state = state["sampler"].get("runner_state")
    if progress.phase != "shared":
        raise ValueError("E0 is a single shared A+B1 route; branch checkpoints are out of scope")
    if modules["encoder"].merge_recipe != context["frozen_ids"]["merge_recipe"]:
        raise ValueError("Encoder Merge recipe differs from the frozen configuration")
    result = run_route(config=config, frozen_ids=context["frozen_ids"], progress=progress, modules=modules,
                       optimizer=optimizer, scheduler=scheduler, schedule=context["schedule"], directory=Path(directory),
                       device=device, control=control, validation=validation,
                       expected_input_signature=expected_input_signature, gate=gate,
                       restored_early_stop=restored_state)
    summary = dict(experiment=config["experiment"], configuration_sha256=configuration_sha256(context["config_path"]),
                   frozen_identity=frozen_identity(context["frozen_ids"]), status=result["status"],
                   reason=result["reason"], updates=result["updates"],
                   logical_update=result["progress"].logical_update, wall_seconds=result["wall_seconds"],
                   early_stop_state=result["early_stop_state"], optimizer_steps=result["updates"],
                   parameter_sha256=parameter_digest(modules), device=str(device))
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _main(argv=None):
    parser = argparse.ArgumentParser(description="EncoderV3 E0 optical intra-observation training entry (fail-closed)")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_optical_selfsup_v1.json"))
    parser.add_argument("--mode", choices=("check", "train"), default="check")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "reports" / "optical_selfsup_runs" / "optical_intra_selfsup_v1")
    parser.add_argument("--resume", type=Path, default=None, help="complete uNNNNNN.pt route checkpoint to resume from")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args(argv)
    try:
        context = build_context(args.config, device=args.device)
        if args.mode == "check":
            report = summarize(context)
        else:
            report = train(context, directory=args.directory, device=args.device, resume=args.resume)
    except ValueError as exc:
        print(f"TRAINING_ENTRY_BLOCKED: {exc}")
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(payload if args.print else json.dumps({key: report.get(key) for key in ("mode", "status", "reason", "updates", "optimizer_steps") if key in report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

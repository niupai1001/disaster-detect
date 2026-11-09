"""E0 preflight: fail-closed identity checks plus a real-data no-update diagnosis.

    python -m encoder_v3.selfsup_preflight --config encoder_v3/config_optical_selfsup_v1.json --mode check
    python -m encoder_v3.selfsup_preflight --config ... --mode no-update

`check` reads configuration, release identity, sampling recipe, fixed panel,
model construction and environment, then never touches a band. `no-update`
additionally runs the real A+B1 forward/backward of the first logical updates
and the fixed validation panel, without constructing any optimizer step:
parameter digests must be identical before and after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from .adaptive_joint import accumulate_adaptive_joint
from .data import NativeOpticalDataset
from .formal_schedule import build_optical_selfsup_schedule, optical_train_pool
from .gates import GATE_IDS, OPTICAL_IDS, optical_identity_only, payload_sha256, require_optical_selfsup_training
from .model import ReflectanceDecoder
from .participation_decoder import MatrixADecoder
from .participation_encoder import ParticipationEncoder, merge_recipe
from .participation_layout import AreaEquivalentHillRate
from .pilot_protocol import optimizer_groups
from .plan_prefetch import move_views
from .route_control import frozen_identity
from .route_step import RouteSchedule, update_signature
from .scheduled_data import OpticalSelfsupSchedule
from .validation_metric import aggregate_intra_validation
from .validation_panels import FrozenValidationPanels

FORMAL_STEP_KEYS = {"precision", "chunk_pixels", "microbatch_observations", "clip_norm", "peak_lr",
                    "plateau_threshold", "weight_decay", "betas", "eps", "ema_coefficient"}


def configuration_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_digest(modules):
    digest = hashlib.sha256()
    for name, module in sorted(modules.items()):
        for key, value in sorted(module.state_dict().items()):
            if not torch.is_tensor(value):
                digest.update(f"{name}:{key}".encode()); continue
            array = value.detach().cpu().contiguous()
            digest.update(f"{name}:{key}:{array.dtype}:{tuple(array.shape)}".encode())
            digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def gradient_norms(modules):
    per_module = {}
    for name, module in sorted(modules.items()):
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().double().square().sum())
        per_module[name] = math.sqrt(total)
    total = math.sqrt(sum(value ** 2 for value in per_module.values()))
    return per_module, total


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {"schema", "experiment", "scope", "B2_training_required", "JEPA_training_required",
                "m9_task_head_training_authorized", "model", "data", "sampling", "validation",
                "control", "formal_step", "required_before_training"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("E0 configuration missing: " + ", ".join(missing))
    if config["schema"] != "encoder_v3_optical_selfsup_config_v1" or config["experiment"] != "optical_intra_selfsup_v1":
        raise ValueError("Not the optical_intra_selfsup_v1 configuration")
    if config["B2_training_required"] is not False or config["JEPA_training_required"] is not False:
        raise ValueError("E0 configuration must declare B2 and JEPA out of scope")
    if config["m9_task_head_training_authorized"] is not False:
        raise ValueError("E0 configuration must keep task-head training unauthorized")
    if set(config["formal_step"]) != FORMAL_STEP_KEYS:
        raise ValueError("Complete formal_step settings required; no candidate defaults")
    if set(config["required_before_training"]) != OPTICAL_IDS:
        raise ValueError("E0 gate block must list exactly the independent optical gate ids")
    if GATE_IDS & set(config["required_before_training"]):
        raise ValueError("E0 configuration cannot reuse historical gate ids")
    return config


def build_context(config_path, *, device="cpu"):
    config = load_config(config_path)
    data = config["data"]
    train = NativeOpticalDataset(Path(data["release"]), Path(data["contracts"]), role="TRAIN")
    validation = NativeOpticalDataset(Path(data["release"]), Path(data["contracts"]), role="VAL")
    curve = AreaEquivalentHillRate(**{key: float(value) for key, value in data["merge_curve"].items()})
    selection_seed = data["selection_seed"]
    recipe_path = Path(config_path).parent / config["sampling"]["recipe"]
    panel_path = Path(config_path).parent / config["validation"]["panel"]
    recipe_sha = config["sampling"]["recipe_sha256"]
    panel_sha = config["validation"]["panel_sha256"]
    frozen = dict(
        formal_config_sha256=frozen_identity(config),
        source_identities=train.identities,
        merge_recipe=merge_recipe(curve, selection_seed),
        sampling_recipe_sha256=recipe_sha,
        validation_panel_sha256=panel_sha,
    )
    pool = optical_train_pool(train)
    schedule = OpticalSelfsupSchedule(recipe_path, config=config, frozen_ids=frozen, dataset=train,
                                     rate_curve=curve, selection_seed=selection_seed, gate=optical_identity_only)
    panels = FrozenValidationPanels(panel_path, config=config, frozen_ids=frozen, dataset=validation,
                                    rate_curve=curve, selection_seed=selection_seed, gate=optical_identity_only,
                                    metric=aggregate_intra_validation)
    if not frozen["source_identities"] == validation.identities:
        raise ValueError("TRAIN and VAL must come from the same frozen release")
    model = config["model"]
    torch.manual_seed(int(model["init_seed"]))
    encoder = ParticipationEncoder(rate_curve=curve, selection_seed=selection_seed,
                                   hyper_hidden=int(model["hypernetwork_hidden"]),
                                   local_batch_size=int(model["local_batch_size"]),
                                   packed_local_output=model["packed_local_output"])
    decoder_type = "matrix" if model["matrix_a_decoder"] else "reference"
    decoder = (ReflectanceDecoder(indexed=True, cell_indexed=bool(model["cell_indexed_decoder"]))
               if decoder_type == "reference" else
               MatrixADecoder(indexed=True, cell_indexed=bool(model["cell_indexed_decoder"]), bulk_geometry=False))
    modules = dict(encoder=encoder, decoder=decoder)
    for module in modules.values():
        module.to(device)
    optimizer = torch.optim.AdamW(optimizer_groups(list(modules.values()), config["formal_step"]["weight_decay"]),
                                  lr=config["formal_step"]["peak_lr"], betas=tuple(config["formal_step"]["betas"]),
                                  eps=config["formal_step"]["eps"])
    control = config["control"]
    scheduler = RouteSchedule(optimizer, peak_lr=config["formal_step"]["peak_lr"],
                              threshold=config["formal_step"]["plateau_threshold"],
                              warmup_updates=int(control["warmup_updates"]),
                              validation_interval=int(control["validation_interval"]),
                              max_updates=int(control["max_updates"]))
    return dict(config=config, config_path=Path(config_path), train=train, validation=validation, curve=curve,
                selection_seed=selection_seed, pool=pool, recipe_path=recipe_path, panel_path=panel_path,
                recipe_sha256=recipe_sha, panel_sha256=panel_sha, frozen_ids=frozen, schedule=schedule,
                panels=panels, modules=modules, optimizer=optimizer, scheduler=scheduler,
                decoder_type=decoder_type, device=device)


def gate_report(config):
    report = {}
    try:
        require_optical_selfsup_training(config)
        report["optical_training_authorized"] = True
    except ValueError as exc:
        report["optical_training_authorized"] = False
        report["optical_blockers"] = [item.strip() for item in str(exc).replace("Optical self-supervised training blocked:", "").split(",") if item.strip()]
    from .gates import require_training
    try:
        require_training(config)
        report["historical_gate_authorized"] = True
    except ValueError:
        report["historical_gate_authorized"] = False
    report["gate_entries"] = {key: ("APPROVED" if value is not None else "OPEN") for key, value in config["required_before_training"].items()}
    return report


def check(context):
    config = context["config"]
    pool = context["pool"]
    schedule = context["schedule"]
    band_counts = defaultdict(lambda: defaultdict(int))
    for _, family, bands, _ in pool["members"]:
        band_counts[family][bands] += 1
    samples = {}
    for index in sorted({0, 1, len(schedule) // 2, len(schedule) - 1}):
        patch, slots = schedule.members(index)
        samples[str(index)] = dict(P_m=patch, members=[identifier for _, identifier in slots],
                                   families={family: sum(1 for slot_family, _ in slots if slot_family == family) for family in ("S2", "Landsat")})
    document = json.loads(context["recipe_path"].read_text(encoding="utf-8-sig"))
    rebuilt = build_optical_selfsup_schedule(
        context["train"], updates=document["updates"], seed=document["sampling"]["seed"],
        slots_per_source=document["sampling"]["slots_per_source"], mask_rule=document["mask_rule"],
        p_values=tuple(document["P_values"]))
    recipe_reproduces = rebuilt == document
    panel_rows = context["panels"].document["panels"]  # read-only identity/coverage summary; no band is touched
    cases_by_observation = defaultdict(lambda: defaultdict(int))
    for case in panel_rows:
        cases_by_observation[case["source_id"]][case["kind"]] += 1
    hidden_counts = defaultdict(list)
    for case in panel_rows:
        if case["kind"] == "B1":
            hidden_counts[case["source_id"]].append(len(case["hidden_band_ids"]))
    parameters = sum(parameter.numel() for module in context["modules"].values() for parameter in module.parameters())
    device = context["device"]
    environment = dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                       device=device, device_name=torch.cuda.get_device_name(torch.device(device)) if str(device).startswith("cuda") else "cpu",
                       peak_allocated_bytes_so_far=int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") else None)
    return dict(
        mode="check",
        status="CHECK_COMPLETE" if recipe_reproduces else "RECIPE_DOES_NOT_REPRODUCE",
        experiment=config["experiment"],
        configuration_sha256=configuration_sha256(context["config_path"]),
        configuration_payload_sha256=payload_sha256(config),
        frozen_identity=frozen_identity(context["frozen_ids"]),
        data=dict(source_identities=context["train"].identities, train_members=len(context["train"]),
                  val_members=len(context["validation"]), release=config["data"]["release"]),
        pool=dict(members=len(pool["members"]), sha256=pool["sha256"],
                  families={family: len(values) for family, values in pool["families"].items()},
                  band_counts={family: dict(sorted(counts.items())) for family, counts in band_counts.items()},
                  excluded=pool["excluded"]),
        schedule=dict(updates=len(schedule), recipe_sha256=context["recipe_sha256"], seed=schedule.seed, recipe_reproduces=recipe_reproduces,
                      slots_per_source=schedule.slots_per_source, sample_updates=samples),
        panel=dict(sha256=context["panel_sha256"], cases=len(panel_rows), observations=sorted(cases_by_observation),
                   cases_by_observation={key: dict(value) for key, value in cases_by_observation.items()},
                   b1_hidden_counts={key: sorted(value) for key, value in hidden_counts.items()},
                   pair_admissions=len(context["panels"].document["pair_admissions"])),
        model=dict(architecture="participation_selected_windows_v1", decoder=context["decoder_type"],
                   parameters=parameters, precision=config["formal_step"]["precision"],
                   merge_recipe=context["modules"]["encoder"].merge_recipe, fp32_only=True),
        environment=environment,
        gates=gate_report(config),
        optimizer_updates=0,
    )


def _hierarchical(rows, kind, group):
    per_observation = defaultdict(list)
    for row in rows:
        if row["kind"] != kind or row["result"].get("loss") is None:
            continue
        per_observation[(row["source"], row["observation"], group(row))].append(row["result"]["loss"])
    grouped = defaultdict(list)
    for (source, observation, key), values in per_observation.items():
        grouped[(source, key)].append(sum(values) / len(values))
    result = defaultdict(dict)
    for (source, key), values in grouped.items():
        result[key][source] = sum(values) / len(values)
    return {str(key): dict(sorted(sources.items())) for key, sources in sorted(result.items(), key=lambda item: str(item[0]))}


def research_metrics(rows):
    def mean_over_sources(values):
        return {key: (sum(source.values()) / len(source)) for key, source in values.items()}
    bands = defaultdict(list)
    for row in rows:
        for band in row["result"].get("bands") or []:
            if band.get("nmse") is not None:
                bands[band["band"]].append(band["nmse"])
    return dict(
        A_by_source=_hierarchical(rows, "A", lambda row: "all"),
        B1_by_source=_hierarchical(rows, "B1", lambda row: "all"),
        A_by_P=_hierarchical(rows, "A", lambda row: row["P_m"]),
        B1_by_P=_hierarchical(rows, "B1", lambda row: row["P_m"]),
        B1_by_hidden_count=_hierarchical(rows, "B1", lambda row: row["hidden_count"]),
        B1_by_band={band: sum(values) / len(values) for band, values in sorted(bands.items())},
        tokens=[dict(panel_id=row["panel_id"], kind=row["kind"], hidden_count=row["hidden_count"], P_m=row["P_m"],
                     tokens=row["result"].get("tokens"), supervised=sum(band.get("count", 0) for band in row["result"].get("bands") or []),
                     unsupported=sum(band.get("unsupported", 0) for band in row["result"].get("bands") or [])) for row in rows],
    )


def no_update(context, updates):
    config = context["config"]
    settings = config["formal_step"]
    device = context["device"]
    encoder, decoder = context["modules"]["encoder"], context["modules"]["decoder"]
    for module in context["modules"].values():
        module.train()
    before = parameter_digest(context["modules"])
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    results = []
    for index in range(updates):
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        packet = context["schedule"].prepare(index, cell_indexed=config["model"]["cell_indexed_decoder"],
                                             retain_layouts=False)
        prepared = time.perf_counter()
        views = move_views(packet["views"], device)
        moved = time.perf_counter()
        result = accumulate_adaptive_joint(encoder, decoder, views, logical_update=index,
                                           chunk_pixels=settings["chunk_pixels"], clip_norm=None,
                                           microbatch_observations=settings["microbatch_observations"])
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        finished = time.perf_counter()
        per_module, total = gradient_norms(context["modules"])
        clipped = float(torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()),
                                                      float(settings["clip_norm"]), error_if_nonfinite=True))
        _, after_clip = gradient_norms(context["modules"])
        supervised = sum(band.get("count", 0) for row in result["AB"] for band in row["result"].get("bands") or [])
        tokens = sum(row["result"].get("tokens", 0) for row in result["AB"])
        results.append(dict(
            logical_update=index,
            P_m=packet["physical_patch_m"],
            members=packet["source_members"],
            AB_input_signature=packet["AB_input_signature"],
            tasks={kind: sum(1 for view in packet["views"] if view.task.kind == kind) for kind in ("A", "B1")},
            absent_tasks=packet["absent_tasks"],
            active_views=sum(1 for row in result["AB"] if row["result"].get("loss") is not None),
            tokens=int(tokens), supervised_pixels=int(supervised),
            losses=dict(AB_weighted=result["AB_weighted_loss"], weighted=result["weighted_loss"],
                        per_view=[dict(observation=row["observation"], kind=row["kind"], loss=row["result"].get("loss")) for row in result["AB"]]),
            gradient_norm_before_clip=total,
            gradient_norm_per_module=per_module,
            clip_norm=settings["clip_norm"],
            clipped_norm_reported=clipped,
            gradient_norm_after_clip=after_clip,
            clip_ratio=(clipped / settings["clip_norm"] if clipped > settings["clip_norm"] else 1.0),
            all_gradients_finite=all(bool(torch.isfinite(parameter.grad).all()) for module in context["modules"].values() for parameter in module.parameters() if parameter.grad is not None),
            timing_seconds=dict(prepare=prepared - started, transfer=moved - prepared,
                                forward_backward=finished - moved, total=finished - started),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") else None,
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if str(device).startswith("cuda") else None,
        ))
        for module in context["modules"].values():
            module.zero_grad(set_to_none=True)
    after = parameter_digest(context["modules"])
    panel_started = time.perf_counter()
    panels = context["panels"].collect(encoder, decoder, cell_indexed=bool(config["validation"]["cell_indexed"]),
                                       retain_layouts=False)
    panel_seconds = time.perf_counter() - panel_started
    rows = panels["rows"]
    final = parameter_digest(context["modules"])
    return dict(
        mode="no-update",
        status="REAL_DATA_NO_UPDATE_COMPLETE" if before == after == final else "PARAMETERS_CHANGED_ABORT",
        experiment=config["experiment"],
        configuration_sha256=configuration_sha256(context["config_path"]),
        frozen_identity=frozen_identity(context["frozen_ids"]),
        parameter_sha256_before=before,
        parameter_sha256_after_updates=after,
        parameter_sha256_after_validation=final,
        optimizer_steps=0,
        updates=results,
        panel=dict(metric=panels["metric"], metric_status=panels["metric_status"], rows=len(rows),
                   seconds_wall=panel_seconds, optimizer_updates=0,
                   research=research_metrics(rows)),
    )


def _main(argv=None):
    parser = argparse.ArgumentParser(description="EncoderV3 E0 optical self-supervision preflight (never steps an optimizer)")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_optical_selfsup_v1.json"))
    parser.add_argument("--mode", choices=("check", "no-update"), default="check")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--updates", type=int, default=2, help="real logical updates materialized in no-update mode")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print", action="store_true", help="print the JSON evidence to stdout")
    args = parser.parse_args(argv)
    if args.mode == "no-update" and not 1 <= args.updates <= 4:
        parser.error("--updates must be between 1 and 4 for an unbudgeted preflight")
    try:
        context = build_context(args.config, device=args.device)
        report = check(context) if args.mode == "check" else no_update(context, args.updates)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"PREFLIGHT_FAILED: {type(exc).__name__}: {exc}")
        return 2
    if args.mode == "no-update" and report["parameter_sha256_before"] != report["parameter_sha256_after_updates"]:
        print("PREFLIGHT_FAILED: parameters changed during a no-update diagnosis")
        return 3
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.print or args.output is None:
        print(payload)
    else:
        summary = {key: report[key] for key in ("mode", "status", "optimizer_steps") if key in report}
        summary["updates"] = len(report.get("updates", [])) if isinstance(report.get("updates"), list) else None
        summary["panel_metric"] = (report.get("panel") or {}).get("metric")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

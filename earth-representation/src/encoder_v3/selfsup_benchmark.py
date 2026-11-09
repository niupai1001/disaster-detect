"""Bounded end-to-end benchmark for E0 execution candidates; never steps an optimizer.

The measured unit is the real frozen A+B1 logical update (eight observations,
one A and one B1 each). Part A changes one execution variable at a time on
update 0 (P60) and re-checks the adopted candidate on update 1 (P120). Part B
changes only the observation microbatch on update 0.

    python -m encoder_v3.selfsup_benchmark --config encoder_v3/config_optical_selfsup_v1.json --output ...
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from .adaptive_joint import accumulate_adaptive_joint
from .model import ReflectanceDecoder
from .participation_decoder import MatrixADecoder
from .participation_encoder import ParticipationEncoder
from .plan_prefetch import move_views
from .selfsup_preflight import build_context, parameter_digest

CANDIDATES = (
    dict(name="reference_B1_readout_local64_packed_off_microbatch1", cell_indexed=False, local_batch=64, packed=False, matrix=False, microbatch=1, repeats=1),
    dict(name="cell_indexed_B1_readout_local64_packed_off_microbatch1", cell_indexed=True, local_batch=64, packed=False, matrix=False, microbatch=1, repeats=2),
    dict(name="cell_indexed_matrix_A_readout_local64", cell_indexed=True, local_batch=64, packed=False, matrix=True, microbatch=1, repeats=1),
    dict(name="cell_indexed_local256_packed_off", cell_indexed=True, local_batch=256, packed=False, matrix=False, microbatch=1, repeats=1),
    dict(name="cell_indexed_local512_packed_off", cell_indexed=True, local_batch=512, packed=False, matrix=False, microbatch=1, repeats=1),
    dict(name="cell_indexed_local1024_packed_off", cell_indexed=True, local_batch=1024, packed=False, matrix=False, microbatch=1, repeats=1),
    dict(name="cell_indexed_local64_packed_on", cell_indexed=True, local_batch=64, packed=True, matrix=False, microbatch=1, repeats=1),
    dict(name="cell_indexed_microbatch2", cell_indexed=True, local_batch=64, packed=False, matrix=False, microbatch=2, repeats=1),
    dict(name="cell_indexed_microbatch4", cell_indexed=True, local_batch=64, packed=False, matrix=False, microbatch=4, repeats=1),
    dict(name="cell_indexed_microbatch8", cell_indexed=True, local_batch=64, packed=False, matrix=False, microbatch=8, repeats=1),
)


def _modules(context, *, local_batch, packed, matrix, cell_indexed):
    config = context["config"]
    torch.manual_seed(int(config["model"]["init_seed"]))
    encoder = ParticipationEncoder(rate_curve=context["curve"], selection_seed=context["selection_seed"],
                                   hyper_hidden=int(config["model"]["hypernetwork_hidden"]),
                                   local_batch_size=local_batch, packed_local_output=packed)
    decoder = (MatrixADecoder(indexed=True, cell_indexed=cell_indexed) if matrix
               else ReflectanceDecoder(indexed=True, cell_indexed=cell_indexed))
    modules = dict(encoder=encoder, decoder=decoder)
    for module in modules.values():
        module.to(context["device"]).train()
    return modules


def _gradient_vector(modules):
    return torch.cat([parameter.grad.detach().reshape(-1) for module in modules.values()
                      for parameter in module.parameters() if parameter.grad is not None])


def _run(context, modules, views, *, microbatch):
    device = context["device"]
    settings = context["config"]["formal_step"]
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = accumulate_adaptive_joint(modules["encoder"], modules["decoder"], views, logical_update=0,
                                       chunk_pixels=settings["chunk_pixels"], clip_norm=None,
                                       microbatch_observations=microbatch)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    gradient = _gradient_vector(modules).double()
    per_observation = defaultdict(int)
    for row in result["AB"]:
        per_observation[row["observation"]] += row["result"].get("tokens", 0)
    report = dict(seconds=seconds, tokens=int(sum(per_observation.values())),
                  sum_n_squared=int(sum(value * value for value in per_observation.values())),
                  max_tokens_per_observation=int(max(per_observation.values(), default=0)),
                  peak_allocated_mb=(torch.cuda.max_memory_allocated() / 2**20 if str(device).startswith("cuda") else None),
                  peak_reserved_mb=(torch.cuda.max_memory_reserved() / 2**20 if str(device).startswith("cuda") else None),
                  losses={f"{row['observation']}:{row['kind']}": row["result"].get("loss") for row in result["AB"]},
                  counts={f"{row['observation']}:{row['kind']}": [band.get("count") for band in row["result"].get("bands") or []] for row in result["AB"]},
                  gradient=gradient, weighted_loss=result["weighted_loss"])
    for module in modules.values():
        module.zero_grad(set_to_none=True)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def _compare(candidate, baseline):
    keys = [key for key in baseline["losses"] if baseline["losses"][key] is not None]
    loss_diffs = {key: abs(candidate["losses"][key] - baseline["losses"][key]) / max(1e-12, abs(baseline["losses"][key])) for key in keys}
    scale = baseline["gradient"].norm().item()
    return dict(max_loss_relative_difference=max(loss_diffs.values()) if loss_diffs else None,
                gradient_relative_l2=(candidate["gradient"] - baseline["gradient"]).norm().item() / scale if scale else None,
                supports_match=candidate["counts"] == baseline["counts"], tokens_match=candidate["tokens"] == baseline["tokens"])


def _small_members(context):
    """Eight smallest pool members: a full eight-observation update at bounded cost."""
    rows = context["train"].rows
    pool = {member[0] for member in context["pool"]["members"]}
    candidates = [(max(max(band["shape"]) for band in row["bands"]), row["id"]) for row in rows if row["id"] in pool]
    return sorted(candidates)[:8]


def _small_views(context):
    from .formal_schedule import _mask_rule, _rectangle_in, observation_bounds
    from .joint import PlannedView
    from .participation_encoder import prepare_participation_task
    from .tasks import fixed_spectral_task, hidden_band_indices, spatial_task
    config, curve = context["config"], context["curve"]
    indices = {row["id"]: index for index, row in enumerate(context["train"].rows)}
    views = []
    for _, identifier in _small_members(context):
        observation = context["train"][indices[identifier]]
        left, bottom, right, top = observation_bounds(observation)
        hidden = sorted(hidden_band_indices(len(observation.bands), seed=context["schedule"].seed, update=0, observation_id=identifier))
        hidden_ids = [observation.bands[index].record_id for index in hidden]
        stds = context["train"].engineering_train_stds(observation)
        rectangle = _rectangle_in(left, bottom, right, top, 60.0, seed=context["schedule"].seed, update=0, identity=identifier,
                                  stream="A_rectangle_optical_selfsup_v1", rule=_mask_rule(config["sampling"]["mask_rule"]))
        for task in (spatial_task(observation, rectangle), fixed_spectral_task(observation, hidden_ids)):
            plan = prepare_participation_task(task, rate_curve=curve, selection_seed=context["selection_seed"],
                                              origin_xy=(left, bottom), chunk_pixels=config["formal_step"]["chunk_pixels"],
                                              cell_indexed=True, retain_layouts=False, physical_patch_m=60.0)
            views.append(PlannedView(task, plan, stds, observation.metadata["sensor"]))
    return views


def _case(context, views):
    return dict(index="bounded-small-eight", P_m=60.0, members=sorted({view.task.metadata["id"] for view in views}),
                tasks=len(views), views=move_views(views, context["device"]), signature=None)


def _prepare(context, index, *, cell_indexed):
    packet = context["schedule"].prepare(index, cell_indexed=cell_indexed, retain_layouts=False)
    return dict(index=index, P_m=packet["physical_patch_m"], members=packet["source_members"],
                tasks=len(packet["views"]), views=move_views(packet["views"], context["device"]),
                signature=packet["AB_input_signature"])


def _benchmark_case(context, case, candidates):
    results, baseline = [], None
    for candidate in candidates:
        timings, record = [], None
        for _ in range(candidate["repeats"]):
            modules = _modules(context, local_batch=candidate["local_batch"], packed=candidate["packed"],
                               matrix=candidate["matrix"], cell_indexed=candidate["cell_indexed"])
            before = parameter_digest(modules)
            run = _run(context, modules, case["views"], microbatch=candidate["microbatch"])
            if before != parameter_digest(modules):
                raise RuntimeError("Parameters changed during a no-update benchmark")
            timings.append(run["seconds"])
            if record is None or run["seconds"] < record["seconds"]:
                record = run
            del modules
        summary = dict(name=candidate["name"], cell_indexed=candidate["cell_indexed"], local_batch=candidate["local_batch"],
                       packed=candidate["packed"], matrix_decoder=candidate["matrix"], microbatch=candidate["microbatch"],
                       repeats=candidate["repeats"], seconds=timings, seconds_median=statistics.median(timings),
                       seconds_min=min(timings), seconds_max=max(timings), tokens=record["tokens"],
                       sum_n_squared=record["sum_n_squared"], max_tokens_per_observation=record["max_tokens_per_observation"],
                       peak_allocated_mb=record["peak_allocated_mb"], peak_reserved_mb=record["peak_reserved_mb"],
                       weighted_loss=record["weighted_loss"], optimizer_updates=0, parameter_sha256=before)
        if baseline is None:
            baseline = record
            summary["relative_to_baseline"] = dict(max_loss_relative_difference=0.0, gradient_relative_l2=0.0,
                                                   supports_match=True, tokens_match=True)
        else:
            summary["relative_to_baseline"] = _compare(record, baseline)
            summary["speedup_vs_baseline"] = results[0]["seconds_median"] / summary["seconds_median"]
            summary["peak_memory_ratio_vs_baseline"] = (summary["peak_allocated_mb"] / results[0]["peak_allocated_mb"]
                                                        if summary["peak_allocated_mb"] and results[0]["peak_allocated_mb"] else None)
        results.append(summary)
    return dict(update=case["index"], P_m=case["P_m"], members=case["members"], tasks=case["tasks"],
                AB_input_signature=case["signature"], candidates=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded E0 execution benchmark (no optimizer step)")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_optical_selfsup_v1.json"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--part", choices=("updates", "cell", "sweep", "all"), default="all")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    context = build_context(args.config, device=args.device)
    updates = []
    adopted = dict(CANDIDATES[1], name="adopted_defaults_cell_indexed_local1024_microbatch1",
                   cell_indexed=bool(context["config"]["model"]["cell_indexed_decoder"]),
                   local_batch=int(context["config"]["model"]["local_batch_size"]),
                   packed=bool(context["config"]["model"]["packed_local_output"]),
                   matrix=bool(context["config"]["model"]["matrix_a_decoder"]),
                   microbatch=int(context["config"]["formal_step"]["microbatch_observations"]),
                   repeats=2)
    if args.part in ("cell", "updates", "all"):
        chosen = ((0, (adopted,)), (1, (dict(adopted, name="adopted_defaults_P120", repeats=1),))) if args.part == "cell" else (
            (0, (CANDIDATES[0], CANDIDATES[1])),
            (1, (dict(CANDIDATES[1], name="cell_indexed_P120_reference_check", repeats=1),)),
        )
        for index, candidates in chosen:
            case = _prepare(context, index, cell_indexed=True)
            updates.append(_benchmark_case(context, case, candidates))
    sweep_candidates = [
        dict(CANDIDATES[1], name="bounded_reference_baseline_local64_packed_off_microbatch1", cell_indexed=True, repeats=2),
        dict(CANDIDATES[1], name="bounded_local256_packed_off", local_batch=256, repeats=3),
        dict(CANDIDATES[1], name="bounded_local512_packed_off", local_batch=512, repeats=3),
        dict(CANDIDATES[1], name="bounded_local1024_packed_off", local_batch=1024, repeats=3),
        dict(CANDIDATES[1], name="bounded_local64_packed_on", packed=True, repeats=1),
        dict(CANDIDATES[1], name="bounded_matrix_A_readout", matrix=True, repeats=1),
        dict(CANDIDATES[1], name="bounded_microbatch2", microbatch=2, repeats=1),
        dict(CANDIDATES[1], name="bounded_microbatch4", microbatch=4, repeats=1),
        dict(CANDIDATES[1], name="bounded_microbatch8", microbatch=8, repeats=1),
    ]
    sweep = _benchmark_case(context, _case(context, _small_views(context)), sweep_candidates) if args.part in ("sweep", "all") else None
    report = dict(experiment=context["config"]["experiment"], device=args.device, torch=torch.__version__,
                  device_name=(torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "cpu"),
                  parameter_count=sum(parameter.numel() for module in context["modules"].values() for parameter in module.parameters()),
                  configuration_sha256=context["frozen_ids"]["formal_config_sha256"],
                  status="BOUNDED_REAL_BENCHMARK_NO_OPTIMIZER_STEP",
                  updates=updates, bounded_execution_sweep=sweep,
                  method="one execution variable at a time on the real frozen A+B1 logical update (eight observations); gradient/support/token equality against the reference readout; no optimizer.step")
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload if args.output is None else json.dumps({key: report[key] for key in ("experiment", "device", "device_name", "status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

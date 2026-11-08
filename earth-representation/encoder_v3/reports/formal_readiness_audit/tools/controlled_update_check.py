"""Authorized controlled AdamW update + exact resume check; no formal training.

Reads eight TRAIN members only, performs two unique updates (P60/P120), saves a
checkpoint, repeats the second update from the checkpoint, and compares state.
The stopped pilot budget is never touched.
"""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from encoder_v3.adaptive_joint import accumulate_adaptive_joint
from encoder_v3.data import NativeOpticalDataset, sha256
from encoder_v3.joint import PlannedView
from encoder_v3.model import ReflectanceDecoder
from encoder_v3.participation_encoder import ParticipationEncoder, prepare_participation_task
from encoder_v3.participation_layout import AreaEquivalentHillRate
from encoder_v3.pilot_protocol import optimizer_groups
from encoder_v3.plan_prefetch import move_views
from encoder_v3.prepare_pilot import bounds
from encoder_v3.route_step import RouteSchedule
from encoder_v3.tasks import fixed_spectral_task, spatial_task
from encoder_v3.training import restore_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
STUDY = ROOT / "encoder_v3" / "reports" / "pilot_study_candidate.json"
AUTH = AUDIT / "approvals" / "D10_authorization.json"
BUDGET = ROOT / "encoder_v3" / "reports" / "pilot_runs" / "budget.json"
OUT = AUDIT / "evidence" / "controlled_update"
CURVE = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if torch.is_tensor(value):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def equal_state(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal_state(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def ordered_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(ordered_tensors(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(ordered_tensors(item))
        return result
    return []


def max_abs_between(left, right):
    pairs = list(zip(ordered_tensors(left), ordered_tensors(right), strict=True))
    return max((float((a.detach().cpu() - b.detach().cpu()).abs().max()) for a, b in pairs), default=0.0)


def relative_l2_between(left, right):
    a = torch.cat([tensor.detach().cpu().flatten().double() for tensor in ordered_tensors(left)])
    b = torch.cat([tensor.detach().cpu().flatten().double() for tensor in ordered_tensors(right)])
    return float((a - b).norm() / b.norm().clamp_min(torch.finfo(b.dtype).tiny))


def main():
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if (auth.get("authorized_by") != "user" or auth.get("unique_logical_updates") != 2
            or auth.get("resume_checks") != 1 or auth.get("max_wall_seconds") > 1200
            or auth.get("max_allocated_GiB") > 10 or auth.get("use_stopped_pilot_budget") is not False
            or auth.get("formal_training_authorized") is not False):
        raise ValueError("Controlled-update authorization is missing or outside the approved caps")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    dataset = NativeOpticalDataset(Path(study["release"]), Path(study["contracts"]), role="TRAIN")
    rows = {row["id"]: row for row in dataset.rows}
    by_id = {row["id"]: index for index, row in enumerate(dataset.rows)}

    def build_views(entry, patch_m):
        views = []
        for task_row in entry["tasks"]:
            identifier = task_row["observation"]
            observation = dataset[by_id[identifier]]
            left, bottom, _, _ = bounds(rows[identifier])
            train_stds = dataset.engineering_train_stds(observation)
            for task in (spatial_task(observation, task_row["A_rectangle"]), fixed_spectral_task(observation, task_row["B1_hidden"])):
                plan = prepare_participation_task(task, rate_curve=CURVE, selection_seed=0, origin_xy=(left, bottom),
                                                 chunk_pixels=512, cell_indexed=True, retain_layouts=False, physical_patch_m=patch_m)
                views.append(PlannedView(task, plan, train_stds, observation.metadata["sensor"]))
        return views

    entry0, entry1 = study["schedule"][0], study["schedule"][1]
    views0_cpu, views1_cpu = build_views(entry0, 60.0), build_views(entry1, 120.0)
    device = torch.device("cuda:0")
    encoder = ParticipationEncoder(rate_curve=CURVE, selection_seed=0, hyper_hidden=64, local_batch_size=64).to(device)
    decoder = ReflectanceDecoder(indexed=True, cell_indexed=True).to(device)
    optimizer = torch.optim.AdamW(optimizer_groups([encoder, decoder], .04), lr=3e-4, betas=(.9, .999), eps=1e-8)
    scheduler = RouteSchedule(optimizer, peak_lr=3e-4, threshold=0., warmup_updates=100, validation_interval=250, max_updates=10000)
    frozen = dict(kind="controlled_update_check", study_sha256=sha256(STUDY), authorization_sha256=sha256(AUTH),
                  source_identities=dataset.identities, merge_recipe=encoder.merge_recipe, unique_updates=2, resume_checks=1)
    budget_before = sha256(BUDGET)
    before = {name: parameter_hash(module) for name, module in (("encoder", encoder), ("decoder", decoder))}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    views0 = move_views(views0_cpu, device)
    scheduler.before_update(0)
    result0 = accumulate_adaptive_joint(encoder, decoder, views0, logical_update=0, chunk_pixels=512, clip_norm=3.0,
                                        microbatch_observations=1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after0 = {name: parameter_hash(module) for name, module in (("encoder", encoder), ("decoder", decoder))}
    checkpoint = OUT / "u000001.pt"
    save_checkpoint(checkpoint, modules={"encoder": encoder, "decoder": decoder}, optimizer=optimizer,
                    scheduler=scheduler, logical_update=1,
                    sampler=dict(next_update=1, next_input_signature=hashlib.sha256(json.dumps(entry1, sort_keys=True).encode()).hexdigest()),
                    frozen_ids=frozen)
    checkpoint_sha = sha256(checkpoint)
    del views0

    views1 = move_views(views1_cpu, device)
    scheduler.before_update(1)
    rng_after_restore = None
    result1 = accumulate_adaptive_joint(encoder, decoder, views1, logical_update=1, chunk_pixels=512, clip_norm=3.0,
                                        microbatch_observations=1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after1 = {name: parameter_hash(module) for name, module in (("encoder", encoder), ("decoder", decoder))}
    snapshot1 = {name: {key: value.detach().clone() for key, value in module.state_dict().items() if torch.is_tensor(value)} for name, module in (("encoder", encoder), ("decoder", decoder))}
    optimizer_snapshot = deepcopy(optimizer.state_dict())
    uninterrupted = dict(parameters=after1, optimizer=optimizer_snapshot, loss=result1["AB_weighted_loss"])
    del views1

    restore_checkpoint(checkpoint, modules={"encoder": encoder, "decoder": decoder}, optimizer=optimizer, scheduler=scheduler, frozen_ids=frozen)
    rng_after_restore = torch.get_rng_state().clone()
    views1 = move_views(views1_cpu, device)
    scheduler.before_update(1)
    result1_resumed = accumulate_adaptive_joint(encoder, decoder, views1, logical_update=1, chunk_pixels=512, clip_norm=3.0,
                                                microbatch_observations=1)
    optimizer.step()
    after_resume = {name: parameter_hash(module) for name, module in (("encoder", encoder), ("decoder", decoder))}
    snapshot_resume = {name: {key: value.detach().clone() for key, value in module.state_dict().items() if torch.is_tensor(value)} for name, module in (("encoder", encoder), ("decoder", decoder))}
    resumed = dict(parameters=after_resume, loss=result1_resumed["AB_weighted_loss"])
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    budget_after = sha256(BUDGET)
    parameter_max_abs = max_abs_between(snapshot1, snapshot_resume)
    optimizer_max_abs = max_abs_between(optimizer_snapshot, optimizer.state_dict())
    optimizer_relative_l2 = relative_l2_between(optimizer_snapshot, optimizer.state_dict())
    loss_difference = abs(result1["AB_weighted_loss"] - result1_resumed["AB_weighted_loss"])
    exact = after1 == after_resume and result1["AB_weighted_loss"] == result1_resumed["AB_weighted_loss"] and equal_state(optimizer_snapshot, optimizer.state_dict())
    preflight_only = parameter_max_abs <= 5e-7 and optimizer_relative_l2 <= 2e-6 and loss_difference <= 1e-6
    status = "CONTROLLED_UPDATE_AND_RESUME_EXACT" if exact else "CONTROLLED_UPDATE_AND_RESUME_WITHIN_PREFLIGHT_TOLERANCE_NOT_FORMAL" if preflight_only else "CONTROLLED_UPDATE_CHECK_FAILED"
    report = dict(
        schema="encoder_v3_controlled_update_check_v1",
        status=status,
        authorization=dict(path=str(AUTH), sha256=sha256(AUTH), unique_updates=2, resume_checks=1),
        study_sha256=sha256(STUDY), release_identities=dataset.identities,
        updates=dict(update0_loss=result0["AB_weighted_loss"], update1_loss=result1["AB_weighted_loss"],
                     update0_param_hashes=after0, update1_param_hashes=after1,
                     update0_gradient_norm=result0["gradient_norm_before_clip"], update1_gradient_norm=result1["gradient_norm_before_clip"]),
        resume=dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                    parameters_equal=after1 == after_resume, loss_equal=result1["AB_weighted_loss"] == result1_resumed["AB_weighted_loss"],
                    optimizer_state_equal=equal_state(optimizer.state_dict(), uninterrupted["optimizer"]),
                    parameter_max_abs=parameter_max_abs, optimizer_state_max_abs=optimizer_max_abs,
                    optimizer_state_relative_l2=optimizer_relative_l2, loss_abs_difference=loss_difference,
                    within_preflight_tolerance_only=preflight_only,
                    tolerance_scope="old FP32 preflight limits; not a new formal tolerance",
                    rng_restored=rng_after_restore is not None,
                    resumed_loss=result1_resumed["AB_weighted_loss"]),
        parameters_changed=before != after1,
        peak_allocated_GiB=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        seconds_wall=seconds, device=torch.cuda.get_device_name(device),
        budget=dict(before=budget_before, after=budget_after, unchanged=budget_before == budget_after,
                    pilot_reserved=214, remaining_not_used=186),
        formal_training=False, task_head_training=False,
        limitations=["Two unique updates on eight preflight TRAIN members; not a formal route.",
                     "No B2, JEPA, validation panel, scheduler plateau or early stop.",
                     "Random initialization fixture; not model-quality evidence."],
    )
    if not (report["budget"]["unchanged"] and report["parameters_changed"] and report["peak_allocated_GiB"] <= 10 and seconds <= 1200):
        report["status"] = "CONTROLLED_UPDATE_CHECK_FAILED_CAP_OR_BUDGET"
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "updates", "resume", "parameters_changed", "peak_allocated_GiB", "seconds_wall", "budget")}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()


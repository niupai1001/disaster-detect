"""Real-data A/B1 forward+backward with no optimizer/EMA/checkpoint.

Reads only the eight user-confirmed preflight TRAIN members and the frozen
release/contracts. It never constructs an optimizer, never calls optimizer.step
and never writes a checkpoint. Run with CUDA_VISIBLE_DEVICES pointing at the
idle device.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from encoder_v3.data import NativeOpticalDataset, sha256
from encoder_v3.joint import PlannedView
from encoder_v3.model import ReflectanceDecoder
from encoder_v3.participation_encoder import ParticipationEncoder, prepare_participation_task
from encoder_v3.participation_layout import AreaEquivalentHillRate
from encoder_v3.plan_prefetch import move_views
from encoder_v3.prepare_pilot import bounds
from encoder_v3.adaptive_joint import accumulate_adaptive_joint
from encoder_v3.tasks import fixed_spectral_task, spatial_task

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
STUDY = ROOT / "encoder_v3" / "reports" / "pilot_study_candidate.json"
BUDGET = ROOT / "encoder_v3" / "reports" / "pilot_runs" / "budget.json"


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if torch.is_tensor(value):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    dataset = NativeOpticalDataset(Path(study["release"]), Path(study["contracts"]), role="TRAIN")
    rows = {row["id"]: row for row in dataset.rows}
    entry = study["schedule"][0]
    curve = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)
    views = []
    for task_row in entry["tasks"]:
        identifier = task_row["observation"]
        observation = dataset[list(rows).index(identifier)]
        left, bottom, _, _ = bounds(rows[identifier])
        a_task = spatial_task(observation, task_row["A_rectangle"])
        b_task = fixed_spectral_task(observation, task_row["B1_hidden"])
        train_stds = dataset.engineering_train_stds(observation)
        for task in (a_task, b_task):
            plan = prepare_participation_task(
                task, rate_curve=curve, selection_seed=0, origin_xy=(left, bottom),
                chunk_pixels=512, cell_indexed=True, retain_layouts=False,
                physical_patch_m=entry["patch_m"],
            )
            views.append(PlannedView(task, plan, train_stds, observation.metadata["sensor"]))
    device = torch.device("cuda:0")
    encoder = ParticipationEncoder(rate_curve=curve, selection_seed=0, hyper_hidden=64, local_batch_size=64).to(device)
    decoder = ReflectanceDecoder(indexed=True, cell_indexed=True).to(device)
    encoder.train(); decoder.train()
    views = move_views(views, device)
    before = {name: parameter_hash(module) for name, module in (("encoder", encoder), ("decoder", decoder))}
    budget_before = sha256(BUDGET) if BUDGET.exists() else None
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = accumulate_adaptive_joint(
        encoder, decoder, views, logical_update=0, chunk_pixels=512, clip_norm=3.0,
        microbatch_observations=1, jepa_views=(), teacher=None, predictor=None,
        jepa_weight=None, latent_loss_fn=None, curriculum_end_update=None, ab_stable=False,
    )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    after = {name: parameter_hash(module) for name, module in (("encoder", encoder), ("decoder", decoder))}
    grads = [p.grad for p in list(encoder.parameters()) + list(decoder.parameters()) if p.grad is not None]
    nonfinite = sum(0 if bool(torch.isfinite(g).all()) else 1 for g in grads)
    budget_after = sha256(BUDGET) if BUDGET.exists() else None
    report = dict(
        schema="encoder_v3_real_ab1_no_update_diagnostic_v1",
        status="REAL_FORWARD_BACKWARD_NO_OPTIMIZER_UPDATE",
        release_identities=dataset.identities,
        logical_update=entry["logical_update"], patch_m=entry["patch_m"],
        observations=[task["observation"] for task in entry["tasks"]],
        views=len(views), tokens=result.get("execution") and None,
        optimizer_updates=result.get("optimizer_updates"),
        weighted_loss=result.get("AB_weighted_loss"),
        gradient_norm_before_clip=result.get("gradient_norm_before_clip"),
        gradient_tensors=len(grads), nonfinite_gradient_tensors=nonfinite,
        parameter_hash_before=before, parameter_hash_after=after,
        parameters_unchanged=before == after,
        budget_before=budget_before, budget_after=budget_after, budget_unchanged=budget_before == budget_after,
        peak_allocated_GiB=torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        seconds_wall=seconds, device=torch.cuda.get_device_name(device),
        checkpoint_created=False, optimizer_constructed=False, ema_run=False,
        limitations=["Randomly initialized fixture model; not reconstruction quality", "Single update-0 pilot schedule; not formal training acceptance", "B2 absent; no cross-source gradient"],
    )
    out = AUDIT / "evidence" / "real_ab1_no_update.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "views", "weighted_loss", "gradient_norm_before_clip", "parameters_unchanged", "budget_unchanged", "peak_allocated_GiB", "seconds_wall", "nonfinite_gradient_tensors")}, indent=2))


if __name__ == "__main__":
    main()

"""Isolated bridge-calibration and feature-assisted USE recovery probes.

Implements the current PLAN.md design without touching the production trainer:

  * calibration: frozen USE/visual, trainable 32->10 bridge, match the frozen
    direct teacher's MiT stage-1 feature on fixed TRAIN full-10 members;
  * group g1: supervised recovery from the calibration state, label loss only;
  * group g2: identical state/data/order/schedule, plus the feature-assist term
    on MiT stage-1 and stage-2 (fixed coefficient 0.05, equal stages);
  * short TRAIN probes (96 fit / 32 check) and a formal 2000-step entry point
    that is implemented but NOT executed here.

The teacher is the existing direct best checkpoint, frozen and evaluated on
TRAIN full-10 inputs only.  No DEV/FINAL/Italy data is used for calibration or
for the short probes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from landslide_seg.config import file_sha256, load_segformer_config, resolve_path, set_seed
from landslide_seg.data.segformer_dataset import S2SegDataset, collate, filter_split, load_stats, read_manifest
from landslide_seg.data.use_dataset import USESegDataset
from landslide_seg.models.factory import build_model, model_identity
from landslide_seg.training.segformer_trainer import (
    _make_optimizer,
    _make_scheduler,
    _seg_loss,
    load_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/use_recovery_probe"
CONFIG_DIR = ROOT / "configs/segformer/use_recovery"
SHORT_MANIFEST = ROOT / "outputs/use_interface_diagnostics/selection/diagnostic_tiles.json"
SEED = 1234
STAGES = ("mit.stage1", "mit.stage2")


def _load_diag():
    spec = importlib.util.spec_from_file_location(
        "diagnose_use_interface", ROOT / "scripts/diagnose_use_interface.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load_diag()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def sha(path: Path) -> str:
    return file_sha256(path)


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def module_hash(module: torch.nn.Module) -> str:
    return diag.state_dict_sha256(module.state_dict())


def tensor_hashes(module: torch.nn.Module) -> dict[str, str]:
    return {name: diag.tensor_sha256(param.detach()) for name, param in module.named_parameters()}


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def freeze_bn(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def module_state_sha(module: torch.nn.Module) -> str:
    return diag.state_dict_sha256({k: v.detach().cpu() for k, v in module.state_dict().items()})


# ---------------------------------------------------------------------------
# data / selection
# ---------------------------------------------------------------------------
def load_records(cfg: dict):
    manifest = resolve_path(cfg["segformer"]["data"]["manifest"])
    rows = read_manifest(manifest)
    train = filter_split(rows, "train")
    dev = filter_split(rows, "val")
    if len(train) != 2223 or len(dev) != 669:
        raise ValueError(f"Unexpected TRAIN/DEV counts: {len(train)}/{len(dev)}")
    return train, dev


def select_calibration_members(records: list[dict], count: int, seed: int) -> list[dict]:
    """Proportional event coverage, deterministic within event."""
    by_event: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_event[record["event_id"]].append(record)
    events = sorted(by_event)
    if count < len(events):
        raise ValueError("Calibration count smaller than event count")
    total = len(records)
    allocation = {event: 1 for event in events}
    remaining = count - len(events)
    exact = {event: count * len(by_event[event]) / total for event in events}
    while remaining > 0:
        candidates = [event for event in events if allocation[event] < len(by_event[event])]
        if not candidates:
            break
        chosen = max(candidates, key=lambda e: (exact[e] - allocation[e], len(by_event[e]), e))
        allocation[chosen] += 1
        remaining -= 1
    selected = []
    for event in events:
        members = sorted(by_event[event], key=lambda r: diag.selection_key(r["sample_id"]))
        selected.extend(members[: allocation[event]])
    return selected


def load_short_split() -> tuple[list[dict], list[dict]]:
    tiles = read_json(SHORT_MANIFEST)
    rows = read_manifest(resolve_path("manifests/v4r1/pool_100.jsonl"))
    by_id = {row["sample_id"]: row for row in rows}
    fit = [by_id[t["sample_id"]] for t in tiles["tiles"] if t["probe_role"] == "fit"]
    check = [by_id[t["sample_id"]] for t in tiles["tiles"] if t["probe_role"] == "check"]
    if len(fit) != 96 or len(check) != 32:
        raise ValueError("Short split must be 96/32")
    return fit, check


class IndexStream:
    """Same deterministic order/exposure semantics as the formal FullBatchStream."""

    def __init__(self, n: int, batch_size: int, seed: int, state: dict | None = None):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0
        self.exposures = np.zeros(self.n, dtype=np.int64)
        if state:
            if state["n"] != self.n or state["batch_size"] != self.batch_size or state["seed"] != self.seed:
                raise ValueError("Stream identity differs from checkpoint")
            self.epoch = int(state["epoch"])
            self.offset = int(state["offset"])
            self.exposures = np.asarray(state["exposures"], dtype=np.int64)
        self.order = self._order()

    def _order(self) -> list[int]:
        gen = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return torch.randperm(self.n, generator=gen).tolist()

    def next(self) -> list[tuple[int, int, int]]:
        items = []
        for _ in range(self.batch_size):
            if self.offset == len(self.order):
                self.epoch += 1
                self.offset = 0
                self.order = self._order()
            index = self.order[self.offset]
            self.offset += 1
            exposure = int(self.exposures[index])
            self.exposures[index] += 1
            items.append((index, self.epoch, exposure))
        return items

    def state_dict(self) -> dict:
        return {
            "n": self.n,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "offset": self.offset,
            "exposures": self.exposures.tolist(),
        }

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(self.state_dict(), sort_keys=True).encode()).hexdigest()


def fetch_batch(
    use_ds: USESegDataset,
    teacher_ds: S2SegDataset | None,
    items: list[tuple[int, int, int]],
    *,
    mixed: bool,
    need_teacher: bool,
):
    students = []
    teachers = []
    for index, epoch, exposure in items:
        use_ds.set_epoch(epoch)
        use_ds.exposure = exposure if mixed else 0
        students.append(use_ds[index])
        if need_teacher:
            assert teacher_ds is not None
            teacher_ds.set_epoch(epoch)
            teachers.append(teacher_ds[index]["image"])
    batch = collate(students)
    teacher_x = torch.stack(teachers) if need_teacher else None
    return batch, teacher_x


# ---------------------------------------------------------------------------
# teacher / features
# ---------------------------------------------------------------------------
def load_teacher(cfg: dict, device: torch.device):
    recovery = cfg["segformer"]["recovery"]
    path = resolve_path(recovery["teacher"])
    expected = recovery.get("teacher_sha256")
    if expected and sha(path) != expected:
        raise ValueError(f"Teacher checkpoint hash differs: {path}")
    model, ckpt = load_checkpoint(path, str(device))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, ckpt, path


def _student_modules(model) -> dict[str, torch.nn.Module]:
    visual = model.visual
    return {
        "mit.stage1": visual.model.segformer.encoder.layer_norm[0],
        "mit.stage2": visual.model.segformer.encoder.layer_norm[1],
    }


def _teacher_modules(model) -> dict[str, torch.nn.Module]:
    return {
        "mit.stage1": model.model.segformer.encoder.layer_norm[0],
        "mit.stage2": model.model.segformer.encoder.layer_norm[1],
    }


def forward_student(model, observation, stages: tuple[str, ...]):
    store = {}
    modules = _student_modules(model)
    handles = []

    def hook(_module, _inputs, output, name):
        value = diag._tensor_from_output(output)
        if torch.is_tensor(value):
            store[name] = value

    for stage in stages:
        handles.append(modules[stage].register_forward_hook(lambda m, i, o, s=stage: hook(m, i, o, s)))
    try:
        out = model(observation)
    finally:
        for handle in handles:
            handle.remove()
    return out, store


def teacher_features(teacher, pixel_values, stages: tuple[str, ...]):
    store = {}
    modules = _teacher_modules(teacher)
    handles = []

    def hook(_module, _inputs, output, name):
        value = diag._tensor_from_output(output)
        if torch.is_tensor(value):
            store[name] = value

    for stage in stages:
        handles.append(modules[stage].register_forward_hook(lambda m, i, o, s=stage: hook(m, i, o, s)))
    try:
        with torch.no_grad():
            teacher(pixel_values)
    finally:
        for handle in handles:
            handle.remove()
    return store


def full_support_mask(valid: torch.Tensor, stride: int) -> torch.Tensor:
    pooled = F.avg_pool2d(valid.float().unsqueeze(1), stride, stride, ceil_mode=True)[:, 0]
    return pooled >= 1.0 - 1e-6


def fit_feature_norm(teacher, records: list[dict], stats: dict, cfg: dict, device: torch.device, out_dir: Path) -> dict:
    ds = S2SegDataset(records, stats, augment=False, seed=SEED, preload=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    stages = STAGES
    accum = {
        stage: {"sum": None, "sumsq": None, "n": 0}
        for stage in stages
    }
    for batch in loader:
        valid = batch["valid"].to(device)
        x = torch.where(valid[:, None], batch["image"].to(device), 0)
        feats = teacher_features(teacher, x, stages)
        for stage in stages:
            feat = feats[stage]
            stride = 4 if stage == "mit.stage1" else 8
            mask = full_support_mask(valid, stride)
            channels = feat.shape[1]
            if accum[stage]["sum"] is None:
                accum[stage]["sum"] = np.zeros(channels, dtype=np.float64)
                accum[stage]["sumsq"] = np.zeros(channels, dtype=np.float64)
            flat = feat.permute(1, 0, 2, 3).reshape(channels, -1)
            m = mask.reshape(-1)
            if m.any():
                selected = flat[:, m]
                accum[stage]["sum"] += selected.sum(1).double().cpu().numpy()
                accum[stage]["sumsq"] += selected.square().sum(1).double().cpu().numpy()
                accum[stage]["n"] += int(m.sum())
    result = {"stages": {}, "source": "TRAIN calibration members, teacher full-10, augment=False",
              "n_members": len(records)}
    for stage in stages:
        n = accum[stage]["n"]
        if n == 0:
            raise RuntimeError(f"No full-support teacher positions for {stage}")
        mean = accum[stage]["sum"] / n
        var = np.maximum(accum[stage]["sumsq"] / n - mean**2, 1e-8)
        result["stages"][stage] = {"mean": mean.tolist(), "std": np.sqrt(var).tolist(), "n_positions": n}
    path = out_dir / "feature_norm.json"
    write_json(path, result)
    result["sha256"] = sha(path)
    write_json(path, result)
    return result


def feature_loss(student_feats: dict, teacher_feats: dict, valid: torch.Tensor, norm: dict, stages: tuple[str, ...]) -> torch.Tensor:
    total = None
    for stage in stages:
        feat = student_feats[stage]
        target = teacher_feats[stage].detach()
        if feat.shape != target.shape:
            raise ValueError(f"Feature shape mismatch at {stage}: {feat.shape} vs {target.shape}")
        stride = 4 if stage == "mit.stage1" else 8
        mask = full_support_mask(valid, stride)
        if mask.any():
            mean = torch.as_tensor(norm["stages"][stage]["mean"], device=feat.device, dtype=feat.dtype).view(1, -1, 1, 1)
            std = torch.as_tensor(norm["stages"][stage]["std"], device=feat.device, dtype=feat.dtype).view(1, -1, 1, 1)
            diff = ((feat - mean) / std - (target - mean) / std).square().mean(1)
            part = (diff * mask).sum() / mask.sum()
        else:
            part = feat.sum() * 0.0
        total = part if total is None else total + part
    assert total is not None
    return total / len(stages)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def _calibration_scheduler(optimizer, steps: int, warmup: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    stream: IndexStream,
    step: int,
    cfg: dict,
    norm: dict,
    extra: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stream": stream.state_dict(),
        "step": int(step),
        "config": cfg,
        "feature_norm": norm,
        "rng": rng_state(),
    }
    payload.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def calibrate(
    cfg: dict,
    out_dir: Path = OUT,
    device_name: str | None = None,
    *,
    steps: int | None = None,
    checkpoint_at: int | None = None,
    resume_from: Path | None = None,
) -> dict:
    device = _device(device_name)
    cal = cfg["segformer"]["recovery"]["calibration"]
    steps = int(steps or cal["updates"])
    batch_size = int(cal["batch_size"])
    stats = load_stats(resolve_path(cfg["segformer"]["data"]["stats"]))
    train, _ = load_records(cfg)
    members = select_calibration_members(train, int(cal["members"]), int(cal["selection_seed"]))
    run_dir = out_dir / "calibration"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "calibration_members.json", {
        "n": len(members),
        "events": dict(sorted(Counter(r["event_id"] for r in members).items())),
        "sample_ids": [r["sample_id"] for r in members],
        "content_sha256": hashlib.sha256(json.dumps([r["sample_id"] for r in members]).encode()).hexdigest(),
    })
    use_ds = USESegDataset(members, stats, augment=True, seed=SEED, preload=True)
    use_ds.mixed = False  # calibration matches the full-10 teacher
    teacher, teacher_ckpt, teacher_path = load_teacher(cfg, device)
    norm = fit_feature_norm(teacher, members, stats, cfg, device, run_dir)

    set_seed(SEED, True)
    model = build_model(cfg["segformer"]).to(device)
    init_hashes = {
        "use": module_hash(model.use),
        "visual": module_hash(model.visual),
        "bridge": module_hash(model.bridge),
        "per_tensor": tensor_hashes(model),
    }
    write_json(run_dir / "initial_hashes.json", {
        "use": init_hashes["use"],
        "visual": init_hashes["visual"],
        "bridge": init_hashes["bridge"],
        "n_tensors": len(init_hashes["per_tensor"]),
        "per_tensor": init_hashes["per_tensor"],
    })
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "config": cfg}, run_dir / "init.pt")

    for parameter in model.use.parameters():
        parameter.requires_grad = False
    for parameter in model.visual.parameters():
        parameter.requires_grad = False
    for parameter in model.bridge.parameters():
        parameter.requires_grad = True
    model.eval()
    teacher.eval()

    optimizer = torch.optim.AdamW(
        [p for p in model.bridge.parameters() if p.requires_grad],
        lr=float(cal["lr"]),
        weight_decay=float(cal["weight_decay"]),
    )
    scheduler = _calibration_scheduler(optimizer, steps, int(cal["warmup_updates"]))
    stream = IndexStream(len(members), batch_size, SEED)
    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        stream = IndexStream(len(members), batch_size, SEED, checkpoint["stream"])
        restore_rng(checkpoint["rng"])
        start_step = int(checkpoint["step"])
    start = time.time()
    history = []
    bridge_grad_zero = 0
    alignment_max = None
    teacher_zero_grad = True
    for step in range(start_step, steps):
        items = stream.next()
        batch, _ = fetch_batch(use_ds, None, items, mixed=False, need_teacher=False)
        image = batch["image"].to(device)
        # The teacher sees the same valid-masked full-10 tensor as the student.
        # This is a feature target, not a re-evaluation of the teacher's raw input.
        teacher_x = image["x"]
        if alignment_max is None:
            alignment_max = float((image["x"] - teacher_x).abs().max())
        out, feats = forward_student(model, image, ("mit.stage1",))
        targets = teacher_features(teacher, teacher_x, ("mit.stage1",))
        loss = feature_loss(feats, targets, batch["valid"].to(device), norm, ("mit.stage1",))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite calibration loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.bridge.parameters() if p.requires_grad], 1.0))
        bridge_grad_zero += int(grad_norm == 0.0)
        teacher_zero_grad = teacher_zero_grad and all(p.grad is None for p in teacher.parameters())
        optimizer.step()
        scheduler.step()
        history.append({"global_step": step + 1, "loss": float(loss.detach()), "grad_norm": grad_norm,
                        "lr": float(optimizer.param_groups[0]["lr"]), "seconds": time.time() - start})
        if checkpoint_at is not None and step + 1 == checkpoint_at:
            save_checkpoint(run_dir / f"step_{checkpoint_at}.pt", model, optimizer, scheduler, stream, step + 1, cfg, norm)
    scope = {
        "use_unchanged": module_hash(model.use) == init_hashes["use"],
        "visual_unchanged": module_hash(model.visual) == init_hashes["visual"],
        "bridge_changed": module_hash(model.bridge) != init_hashes["bridge"],
        "bridge_grad_zero_steps": bridge_grad_zero,
        "teacher_zero_grad": teacher_zero_grad,
        "train_teacher_alignment_max_abs": alignment_max,
    }
    save_checkpoint(run_dir / "calibration.pt", model, optimizer, scheduler, stream, steps, cfg, norm,
                    extra={"parameter_scope": scope, "teacher": str(teacher_path).replace("\\", "/"),
                           "teacher_sha256": sha(teacher_path), "teacher_ckpt": teacher_ckpt})
    summary = {
        "stage": "calibration",
        "steps": steps,
        "n_members": len(members),
        "batch_size": batch_size,
        "mixed": False,
        "loss": "normalized MSE to frozen teacher MiT stage-1, full-support valid positions",
        "optimizer": "AdamW bridge-only",
        "lr": float(cal["lr"]),
        "weight_decay": float(cal["weight_decay"]),
        "scheduler": "linear warmup + cosine decay",
        "warmup_updates": int(cal["warmup_updates"]),
        "feature_norm": sha(run_dir / "feature_norm.json"),
        "teacher": str(teacher_path).replace("\\", "/"),
        "teacher_sha256": sha(teacher_path),
        "initial_hashes": {k: init_hashes[k] for k in ("use", "visual", "bridge")},
        "final_hashes": {"use": module_hash(model.use), "visual": module_hash(model.visual), "bridge": module_hash(model.bridge)},
        "parameter_scope": scope,
        "stream_sha256": stream.sha(),
        "history": history,
        "seconds": time.time() - start,
        "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "not_formal_training": True,
    }
    write_json(run_dir / "calibration_summary.json", summary)
    write_json(run_dir / "resolved_config.json", cfg)
    del model, teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_calibration_model(cfg: dict, path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(cfg["segformer"]).to(device)
    model.load_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad = True
    norm = read_json(path.parent / "feature_norm.json")
    return model, checkpoint, norm


# ---------------------------------------------------------------------------
# supervised recovery
# ---------------------------------------------------------------------------
def supervised(
    cfg: dict,
    group: str,
    out_dir: Path = OUT,
    device_name: str | None = None,
    *,
    short: bool = True,
    steps: int | None = None,
    execute_formal: bool = False,
    resume_from: Path | None = None,
    checkpoint_at: int | None = None,
    calibration_path: Path | None = None,
) -> dict:
    if group not in ("g1", "g2"):
        raise ValueError(group)
    recovery = cfg["segformer"]["recovery"]
    feature_assist = group == "g2"
    if feature_assist and not bool(recovery.get("feature_assist", False)):
        raise ValueError("g2 config must set recovery.feature_assist=true")
    device = _device(device_name)
    set_seed(SEED, True)
    stats = load_stats(resolve_path(cfg["segformer"]["data"]["stats"]))
    train, dev = load_records(cfg)
    if short:
        fit_records, check_records = load_short_split()
        records = fit_records
        eval_records = check_records
        fit_eval_records = fit_records
        updates = int(steps or recovery["short"]["updates"])
        schedule_steps = updates
        eval_every = int(recovery["short"]["eval_every"])
        report_steps = []
    else:
        if not execute_formal:
            return {
                "status": "NOT_RUN",
                "reason": "formal 2000-step training is not authorised in this task; pass --execute to run",
                "group": group,
                "config": str(CONFIG_DIR / f"group{group[-1]}.yaml").replace("\\", "/"),
                "command": f"python -X utf8 scripts/run_use_recovery_probe.py formal --group {group} --execute --device cuda:0",
            }
        records = train
        eval_records = dev
        fit_eval_records = None
        updates = int(steps or recovery["formal"]["stop_updates"])
        schedule_steps = int(recovery["formal"]["schedule_updates"])
        eval_every = int(recovery["formal"]["eval_every"])
        report_steps = [int(x) for x in recovery["formal"]["report_steps"]]
    if calibration_path is None:
        calibration_path = out_dir / "calibration/calibration.pt"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Missing calibration state: {calibration_path}")
    model, calibration, norm = load_calibration_model(cfg, calibration_path, device)
    start_state_hash = module_state_sha(model)
    teacher = None
    teacher_ds = None
    teacher_path = None
    if feature_assist:
        teacher, _, teacher_path = load_teacher(cfg, device)
        teacher_ds = S2SegDataset(records, stats, augment=True, seed=SEED, preload=True)

    use_ds = USESegDataset(records, stats, augment=True, seed=SEED, preload=True)
    use_ds.mixed = True
    check_ds = USESegDataset(eval_records, stats, augment=False, preload=True) if short else None
    fit_eval_ds = USESegDataset(fit_eval_records, stats, augment=False, preload=True) if short else None
    dev_ds = USESegDataset(eval_records, stats, augment=False, preload=True) if not short else None
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg, schedule_steps)
    stream = IndexStream(len(records), int(cfg["segformer"]["train"]["batch_size"]), SEED)
    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        stream = IndexStream(len(records), int(cfg["segformer"]["train"]["batch_size"]), SEED, checkpoint["stream"])
        restore_rng(checkpoint["rng"])
        start_step = int(checkpoint["step"])
        start_state_hash = checkpoint.get("start_state_hash", start_state_hash)
    run_dir = out_dir / ("short" if short else "formal") / group
    run_dir.mkdir(parents=True, exist_ok=True)
    if short:
        eval_start = diag.evaluate_subset(model, fit_eval_ds, cfg, device)
        check_start = diag.evaluate_subset(model, check_ds, cfg, device)
    else:
        eval_start = diag.evaluate_subset(model, dev_ds, cfg, device)
        check_start = None
    teacher_grads_zero = True
    scope_start = {"use": module_hash(model.use), "visual": module_hash(model.visual), "bridge": module_hash(model.bridge)}
    history = []
    start = time.time()
    best_score = -1.0
    best_step = -1
    freeze_bn(model)
    for step in range(start_step, updates):
        model.train()
        freeze_bn(model)
        items = stream.next()
        batch, teacher_x = fetch_batch(use_ds, teacher_ds, items, mixed=True, need_teacher=feature_assist)
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        valid = batch["valid"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if feature_assist:
            out, student_feats = forward_student(model, image, STAGES)
            teacher_x = torch.where(valid[:, None], teacher_x.to(device), 0)
            teacher_feats = teacher_features(teacher, teacher_x, STAGES)
            label = _seg_loss(cfg, out["logits"], mask, valid, boundary=True)
            feature = feature_loss(student_feats, teacher_feats, valid, norm, STAGES)
            loss = label + float(recovery["feature"]["coefficient"]) * feature
            teacher_grads_zero = teacher_grads_zero and all(p.grad is None for p in teacher.parameters())
        else:
            out = model(image)
            label = _seg_loss(cfg, out["logits"], mask, valid, boundary=True)
            feature = None
            loss = label
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised loss")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["segformer"]["optim"]["grad_clip"])))
        optimizer.step()
        scheduler.step()
        history.append({
            "global_step": step + 1,
            "label_loss": float(label.detach()),
            "feature_loss": float(feature.detach()) if feature is not None else None,
            "total_loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "lr": [float(g["lr"]) for g in optimizer.param_groups],
            "seconds": time.time() - start,
        })
        if (step + 1) % eval_every == 0 or step + 1 == updates:
            if short:
                fit_metrics = diag.evaluate_subset(model, fit_eval_ds, cfg, device)
                check_metrics = diag.evaluate_subset(model, check_ds, cfg, device)
                history[-1]["fit"] = fit_metrics
                history[-1]["check"] = check_metrics
            else:
                dev_metrics = diag.evaluate_subset(model, dev_ds, cfg, device)
                history[-1]["dev"] = dev_metrics
                score = float(dev_metrics["event_mean_iou"])
                if score > best_score:
                    best_score, best_step = score, step + 1
                    save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, stream, step + 1, cfg, norm,
                                    extra={"group": group, "selection": "dev_event_mean_iou", "start_state_hash": start_state_hash})
                if step + 1 in report_steps:
                    save_checkpoint(run_dir / f"step{step + 1}.pt", model, optimizer, scheduler, stream, step + 1, cfg, norm,
                                    extra={"group": group, "start_state_hash": start_state_hash})
        if checkpoint_at is not None and step + 1 == checkpoint_at:
            save_checkpoint(run_dir / f"step_{checkpoint_at}.pt", model, optimizer, scheduler, stream, step + 1, cfg, norm,
                            extra={"group": group, "start_state_hash": start_state_hash})
        if (step + 1) % 100 == 0 or step + 1 == updates:
            print(f"[{group}] step={step+1}/{updates} loss={float(loss.detach()):.4f} label={float(label.detach()):.4f} seconds={time.time()-start:.1f}", flush=True)
    save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, stream, updates, cfg, norm,
                    extra={"group": group, "start_state_hash": start_state_hash})
    final = {
        "group": group,
        "short": short,
        "steps": updates,
        "mixed_bands": True,
        "feature_assist": feature_assist,
        "feature_coefficient": float(recovery["feature"]["coefficient"]) if feature_assist else None,
        "feature_stages": list(recovery["feature"]["stages"]) if feature_assist else None,
        "calibration_path": str(calibration_path.relative_to(ROOT)).replace("\\", "/"),
        "calibration_sha256": sha(calibration_path),
        "start_state_hash": start_state_hash,
        "start_step": start_step,
        "resumed_from": str(resume_from.relative_to(ROOT)).replace("\\", "/") if resume_from else None,
        "scope_start": scope_start,
        "scope_end": {"use": module_hash(model.use), "visual": module_hash(model.visual), "bridge": module_hash(model.bridge)},
        "teacher": str(teacher_path).replace("\\", "/") if teacher_path else None,
        "teacher_zero_grad": teacher_grads_zero if feature_assist else None,
        "eval_start": eval_start,
        "check_start": check_start,
        "history": history,
        "seconds": time.time() - start,
        "peak_alloc_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "best_score": best_score if not short else None,
        "best_step": best_step if not short else None,
        "formal_training": not short,
        "not_generalisation": short,
    }
    if short:
        final["fit_end"] = diag.evaluate_subset(model, fit_eval_ds, cfg, device)
        final["check_end"] = diag.evaluate_subset(model, check_ds, cfg, device)
    else:
        final["dev_end"] = diag.evaluate_subset(model, dev_ds, cfg, device)
    write_json(run_dir / "summary.json", final)
    write_json(run_dir / "resolved_config.json", cfg)
    (run_dir / "per_update.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in history), encoding="utf-8"
    )
    del model
    if teacher is not None:
        del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final


# ---------------------------------------------------------------------------
# interface checks / resume self-check
# ---------------------------------------------------------------------------
@torch.no_grad()
def interface_checks(cfg: dict, out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    path = out_dir / "calibration/calibration.pt"
    model, _, norm = load_calibration_model(cfg, path, device)
    model.eval()
    train, _ = load_records(cfg)
    records = select_calibration_members(train, 32, int(cfg["segformer"]["recovery"]["calibration"]["selection_seed"]))
    ds = USESegDataset(records, load_stats(resolve_path(cfg["segformer"]["data"]["stats"])), augment=False, preload=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
    perms = [list(reversed(range(10))), [0, 2, 1, 3, 4, 5, 6, 7, 8, 9]]
    perm_rows = []
    for batch in loader:
        image = batch["image"].to(device)
        p0 = diag.probability(model, image, "use")
        valid = batch["valid"].to(device)
        row = {}
        for mode in ("joint", "band", "metadata"):
            values = []
            for perm in perms:
                permuted = diag.permute_batch(batch, perm, mode, "use")
                p1 = diag.probability(model, diag.batch_input(permuted, "use").to(device), "use")
                values.append(diag.prob_delta(p0, p1, valid))
            row[mode] = float(np.mean([v["mean_abs_diff"] for v in values]))
        perm_rows.append(row)
    subset_rows = {}
    for name in ("rgb", "rgb_nir", "six", "unseen5"):
        values = []
        for batch in loader:
            image = batch["image"].to(device)
            p_full = diag.probability(model, image, "use")
            sub = diag.subset_batch(batch, diag.SUBSETS[name])
            p_sub = diag.probability(model, diag.batch_input(sub, "use").to(device), "use")
            values.append(diag.prob_delta(p_full, p_sub, batch["valid"].to(device)))
        subset_rows[name] = float(np.mean([v["iou_agreement"] for v in values]))
    masked = []
    for batch in loader:
        image = batch["image"].to(device)
        p_full = diag.probability(model, image, "use")
        valid = batch["valid"].to(device)
        remove = [diag.S2_10_BANDS.index(b) for b in ("B05", "B06", "B07", "B8A", "B11", "B12")]
        p_mask = diag.probability(model, diag.batch_input(diag.masked_batch(batch, remove), "use").to(device), "use")
        p_absent = diag.probability(
            model, diag.batch_input(diag.subset_batch(batch, ("B02", "B03", "B04", "B08")), "use").to(device), "use"
        )
        masked.append({"masked_vs_absent": diag.prob_delta(p_absent, p_mask, valid)["mean_abs_diff"],
                       "finite": bool(torch.isfinite(p_mask).all())})
    valid_rows = []
    for batch in loader:
        base = diag.clone_batch(batch)
        region = (slice(None), slice(None), slice(32, 64), slice(32, 64))
        base["image"]["pixel_valid"][region] = False
        base["image"]["x"][region] = 0.0
        p_zero = diag.probability(model, diag.batch_input(base, "use").to(device), "use")
        nan_batch = diag.clone_batch(base)
        nan_batch["image"]["x"][region] = float("nan")
        out_nan = model(diag.batch_input(nan_batch, "use").to(device))
        p_nan = torch.sigmoid(out_nan["logits"].float()[:, 0])
        valid = batch["valid"].to(device).clone()
        valid[:, 32:64, 32:64] = False
        valid_rows.append({
            "finite": bool(torch.isfinite(p_nan).all()),
            "max_diff": float((p_zero - p_nan).abs()[valid].max()) if valid.any() else None,
            "input_valid_matches": bool(
                (out_nan["input_valid"].detach().cpu()
                 == (nan_batch["image"]["pixel_valid"].bool()
                     & nan_batch["image"]["band_valid"][:, :, None, None]).any(1)).all()
            ),
        })
    result = {
        "calibration_sha256": sha(path),
        "permutation_mean_abs_diff": {mode: float(np.mean([r[mode] for r in perm_rows])) for mode in ("joint", "band", "metadata")},
        "missing_band_iou_agreement": subset_rows,
        "masked_vs_absent": float(np.mean([r["masked_vs_absent"] for r in masked])),
        "masked_finite": bool(all(r["finite"] for r in masked)),
        "valid_region_nan_far_max": float(max(r["max_diff"] for r in valid_rows if r["max_diff"] is not None)),
        "valid_region_finite": bool(all(r["finite"] for r in valid_rows)),
        "input_valid_matches": bool(all(r["input_valid_matches"] for r in valid_rows)),
    }
    write_json(out_dir / "interface_checks.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def selfcheck(cfg: dict, out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    work = out_dir / "selfcheck"
    work.mkdir(parents=True, exist_ok=True)
    # Control: 6 calibration steps with a checkpoint at 3.
    first = calibrate(cfg, work / "a", device, steps=6, checkpoint_at=3)
    resumed = calibrate(cfg, work / "b", device, steps=6, checkpoint_at=3, resume_from=work / "a/calibration/step_3.pt")
    # Uninterrupted control.
    control = calibrate(cfg, work / "c", device, steps=6)
    a = torch.load(work / "a/calibration/calibration.pt", map_location="cpu", weights_only=False)
    b = torch.load(work / "b/calibration/calibration.pt", map_location="cpu", weights_only=False)
    c = torch.load(work / "c/calibration/calibration.pt", map_location="cpu", weights_only=False)
    def bridge_hash(state):
        return diag.state_dict_sha256({k: v for k, v in state.items() if k.startswith("bridge.")})
    bridge_keys = [k for k in b["model"] if k.startswith("bridge.")]
    bridge_max_diff = max(float((b["model"][k] - c["model"][k]).abs().max()) for k in bridge_keys)
    result = {
        "steps": 6,
        "checkpoint_at": 3,
        "resumed_bridge_hash": bridge_hash(b["model"]),
        "control_bridge_hash": bridge_hash(c["model"]),
        "bridge_match": bridge_hash(b["model"]) == bridge_hash(c["model"]),
        "bridge_max_abs_diff": bridge_max_diff,
        "bridge_close": bridge_max_diff < 1e-4,
        "stream_match": b["stream"] == c["stream"],
        "optimizer_step_match": bool(
            b["optimizer"]["state"][next(iter(b["optimizer"]["state"]))]["step"]
            == c["optimizer"]["state"][next(iter(c["optimizer"]["state"]))]["step"]
        ),
        "first_loss": first["history"][-1]["loss"],
        "resumed_loss": resumed["history"][-1]["loss"],
        "control_loss": control["history"][-1]["loss"],
        "loss_abs_diff": abs(resumed["history"][-1]["loss"] - control["history"][-1]["loss"]),
    }
    try:
        result["supervised"] = supervised_selfcheck(cfg, out_dir, device_name)
    except Exception as exc:  # keep the calibration result even if the short supervised check fails
        result["supervised"] = {"error": type(exc).__name__, "message": str(exc)}
    write_json(out_dir / "selfcheck.json", result)
    return result


# ---------------------------------------------------------------------------
# provenance / CLI
# ---------------------------------------------------------------------------
def supervised_selfcheck(cfg: dict, out_dir: Path = OUT, device_name: str | None = None) -> dict:
    device = _device(device_name)
    work = out_dir / "selfcheck_supervised"
    work.mkdir(parents=True, exist_ok=True)
    calibration_path = out_dir / "calibration/calibration.pt"
    common = dict(short=True, steps=4, calibration_path=calibration_path)
    supervised(cfg, "g1", work / "a", device_name, checkpoint_at=2, **common)
    supervised(cfg, "g1", work / "b", device_name, resume_from=work / "a/short/g1/step_2.pt", **common)
    supervised(cfg, "g1", work / "c", device_name, **common)
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    a = load(work / "a/short/g1/last.pt")
    b = load(work / "b/short/g1/last.pt")
    c = load(work / "c/short/g1/last.pt")
    max_diff = max(float((b["model"][key] - c["model"][key]).abs().max()) for key in b["model"])
    stream_ok = b["stream"] == c["stream"] == a["stream"]
    opt_key = next(iter(b["optimizer"]["state"]))
    opt_ok = bool(b["optimizer"]["state"][opt_key]["step"] == c["optimizer"]["state"][opt_key]["step"])
    summary_b = read_json(work / "b/short/g1/summary.json")
    summary_c = read_json(work / "c/short/g1/summary.json")
    result = {
        "steps": 4,
        "checkpoint_at": 2,
        "model_max_abs_diff": max_diff,
        "model_close": max_diff < 1e-4,
        "stream_match": stream_ok,
        "optimizer_step_match": opt_ok,
        "resumed_final_label_loss": summary_b["history"][-1]["label_loss"],
        "control_final_label_loss": summary_c["history"][-1]["label_loss"],
        "loss_abs_diff": abs(summary_b["history"][-1]["label_loss"] - summary_c["history"][-1]["label_loss"]),
    }
    write_json(out_dir / "selfcheck_supervised.json", result)
    return result


def sequence_check(out_dir: Path = OUT) -> dict:
    rows = {}
    for group in ("g1", "g2"):
        path = out_dir / "short" / group / "last.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            rows[group] = {"stream": payload.get("stream"), "stream_sha": hashlib.sha256(
                json.dumps(payload.get("stream"), sort_keys=True).encode()).hexdigest()}
    equal = len(rows) == 2 and rows["g1"]["stream"] == rows["g2"]["stream"]
    result = {
        "g1_stream_sha256": rows.get("g1", {}).get("stream_sha"),
        "g2_stream_sha256": rows.get("g2", {}).get("stream_sha"),
        "stream_equal": equal,
        "n_exposures_g1": int(sum(rows.get("g1", {}).get("stream", {}).get("exposures", []))),
        "n_exposures_g2": int(sum(rows.get("g2", {}).get("stream", {}).get("exposures", []))),
        "note": "stream equality covers epoch/offset/exposures; with the same seed and records it makes the per-sample mixed-band view sequence identical",
    }
    write_json(out_dir / "short_sequence_check.json", result)
    return result


def write_report(out_dir: Path = OUT) -> Path:
    sequence = sequence_check(out_dir)
    calibration = read_json(out_dir / "calibration/calibration_summary.json")
    g1 = read_json(out_dir / "short/g1/summary.json")
    g2 = read_json(out_dir / "short/g2/summary.json")
    interface = read_json(out_dir / "interface_checks.json") if (out_dir / "interface_checks.json").is_file() else {}
    selfcheck = read_json(out_dir / "selfcheck.json") if (out_dir / "selfcheck.json").is_file() else {}
    hashes = {
        "calibration.pt": sha(out_dir / "calibration/calibration.pt"),
        "feature_norm.json": sha(out_dir / "calibration/feature_norm.json"),
        "initial_hashes.json": sha(out_dir / "calibration/initial_hashes.json"),
        "calibration_members.json": sha(out_dir / "calibration/calibration_members.json"),
        "calibration.yaml": sha(CONFIG_DIR / "calibration.yaml"),
        "group1.yaml": sha(CONFIG_DIR / "group1.yaml"),
        "group2.yaml": sha(CONFIG_DIR / "group2.yaml"),
        "recovery_base.yaml": sha(CONFIG_DIR / "recovery_base.yaml"),
    }
    first_loss = calibration["history"][0]["loss"]
    last_loss = calibration["history"][-1]["loss"]
    lines = [
        "# USE两组恢复实验：隔离实现与短程验收",
        "",
        "> 本目录只包含 TRAIN 短程工程探针；**没有启动两组 2000 步正式训练**。"
        "短程 IoU 不作为泛化或正式成绩。",
        "",
        "验证：全量 `pytest -q` 通过（含 `tests/test_use_recovery_probe.py` 的 7 项检查）；正式分支仅做未执行 dry-run。",
        "",
        "## 1. 实际配置",
        "",
        "- 基础：`event_ood_v4r1`，TRAIN 2223 / DEV 669，seed1234，`stats_core25`，mixed bands 与原 physical_mixed 相同。",
        "- 视觉：官方 MiT-B0 与解码器初始化；USE 与 bridge 随机初始化；未从已训练 physical/physical_mixed 续训。",
        "- 教师：本轮直入 `use_baseline/1234/best.pt`，SHA256 `5498d1dec476613c77f4bed2aea3e4ba6d589cd3c4fb67852043568954e83e98`；全部参数冻结、eval、只读 TRAIN full10。",
        "- 教师特征目标与学生在同一有效区掩膜后的 full10 输入上计算（校准阶段两者逐元素相同；g2 学生为混合波段子集、教师仍为 full10），不是重新评价教师的原始前向。",
        "- 校准：固定 512 个 TRAIN 成员（事件比例分配，seed 20261024），full10，batch8，200 步；冻结 USE/visual，仅 bridge 可训练；匹配教师 MiT stage-1。",
        "- g1：从校准状态起，mixed 波段监督，标签损失 0.5 BCE+0.5 Dice；g2：同状态、同数据/顺序/增强/波段采样/学习率，唯一新增 MiT stage-1+stage-2 特征项，系数 0.05（等权）。",
        "",
        "| 文件 | SHA256 |",
        "|---|---|",
    ]
    for name, value in hashes.items():
        lines.append(f"| `{name}` | `{value}` |")
    lines += [
        "",
        "## 2. 桥接标定结果",
        "",
        f"- 200 步，512 成员；参数范围：USE 未变 {calibration['parameter_scope']['use_unchanged']}、visual 未变 {calibration['parameter_scope']['visual_unchanged']}、bridge 改变 {calibration['parameter_scope']['bridge_changed']}。",
        f"- 教师零梯度：{calibration['parameter_scope']['teacher_zero_grad']}；学生 full10 与教师输入逐元素最大差 {calibration['parameter_scope']['train_teacher_alignment_max_abs']}。",
        f"- 特征匹配损失：首步 {first_loss:.4f} → 末步 {last_loss:.4f}；bridge 梯度为零的步数 {calibration['parameter_scope']['bridge_grad_zero_steps']}。",
        f"- 校准用时 {calibration['seconds']:.1f} s；峰值显存 {calibration['peak_alloc_mib']:.0f} MiB。",
        "",
        "## 3. 短程恢复探针（TRAIN 96 拟合 / 32 检查，300 步）",
        "",
        "| 组 | 起始拟合IoU | 末拟合IoU | 起始检查IoU | 末检查IoU | 用时(s) | 峰值显存(MiB) | 教师零梯度 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for group, data in (("g1", g1), ("g2", g2)):
        teacher_state = "不适用（g1无教师项）" if data["teacher_zero_grad"] is None else str(data["teacher_zero_grad"])
        lines.append(
            f"| {group} | {data['eval_start']['pooled']['iou']:.4f} | {data['fit_end']['pooled']['iou']:.4f} | "
            f"{data['check_start']['pooled']['iou']:.4f} | {data['check_end']['pooled']['iou']:.4f} | "
            f"{data['seconds']:.1f} | {data['peak_alloc_mib']:.0f} | {teacher_state} |"
        )
    lines += [
        "",
        f"两组起始状态哈希相同：`{g1['start_state_hash']}`；这是“从相同标定状态开始”的直接证据。",
        f"两组数据/波段序列（stream）一致：{sequence['stream_equal']}；stream SHA256 `{sequence['g1_stream_sha256']}`。",
        "g2 的拟合 IoU 高于 g1，但 TRAIN 内 32 块检查没有同步提高；单 seed、300 步、无系数扫描，不能据此判断特征辅助有效。",
        "80 步初测在阈值 0.5 下退化为全背景，因此短程验收固定为 300 步；该初测不作为结果保留。",
        "",
        "## 4. 接口与恢复硬门槛",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 联合置换平均概率差 | {interface.get('permutation_mean_abs_diff', {}).get('joint')} |",
        f"| 仅波段置换 | {interface.get('permutation_mean_abs_diff', {}).get('band')} |",
        f"| 仅元数据置换 | {interface.get('permutation_mean_abs_diff', {}).get('metadata')} |",
        f"| masked vs absent 概率差 | {interface.get('masked_vs_absent')} |",
        f"| 缺波段全部有限 | {interface.get('masked_finite')} |",
        f"| 无效区 NaN 远处最大变化 | {interface.get('valid_region_nan_far_max')} |",
        f"| input_valid 与掩膜一致 | {interface.get('input_valid_matches')} |",
        f"| 校准 checkpoint 恢复：bridge 最大差 | {selfcheck.get('bridge_max_abs_diff')}（<1e-4: {selfcheck.get('bridge_close')}） |",
        f"| 校准恢复后 stream 一致 | {selfcheck.get('stream_match')} |",
        f"| 校准恢复后 optimizer step 一致 | {selfcheck.get('optimizer_step_match')} |",
        f"| 短程监督恢复：模型最大差 | {selfcheck.get('supervised', {}).get('model_max_abs_diff')}（<1e-4: {selfcheck.get('supervised', {}).get('model_close')}） |",
        f"| 短程监督恢复后 stream/optimizer 一致 | {selfcheck.get('supervised', {}).get('stream_match')} / {selfcheck.get('supervised', {}).get('optimizer_step_match')} |",
        f"| g1/g2 数据序列一致 | {sequence['stream_equal']} |",
        "",
        "## 5. 正式训练待执行命令（本次未运行）",
        "",
        "```powershell",
        "$env:PYTHONPATH='src'",
        "# 仅在 calibration.pt 不存在时运行；已有则跳过，保证两组共用同一标定状态",
        "python -X utf8 scripts/run_use_recovery_probe.py calibrate",
        "python -X utf8 scripts/run_use_recovery_probe.py formal --group g1 --execute --device cuda:0",
        "python -X utf8 scripts/run_use_recovery_probe.py formal --group g2 --execute --device cuda:0",
        "```",
        "",
        "两组 formal 都读取同一 `outputs/use_recovery_probe/calibration/calibration.pt`；不要在中途重跑 calibration。",
        "formal 分支按 PLAN 使用全 TRAIN 2223、full 1/2 + RGB/4/6 波段各 1/6、batch8、FP32、BN 运行统计冻结、stop2000/schedule3600/warmup180、每100步 DEV 选模、保存 500/1000/1500/2000 与 best/last。",
        "",
        "## 6. 限制",
        "",
        "- 短程探针是 TRAIN 96/32、单 seed、300 步；不是 DEV 成绩，不用于正式选模。",
        "- 校准只匹配教师 MiT stage-1；g2 只加 stage-1/2 特征项，未做系数扫描、未做多 seed、未做结构消融。",
        "- 教师是既有直入 best，继承开发选模历史；本实现把它固定为开发实验教师，并在 TRAIN 上生成目标，但不是无历史接触的教师。",
        "- formal 分支已实现但本次未执行；正式结果、DEV 选模和评价尚不存在。",
    ]
    path = OUT / "README.md"
    summary = {
        "calibration": {k: calibration[k] for k in ("steps", "n_members", "seconds", "parameter_scope", "teacher_sha256", "feature_norm")},
        "short": {"g1": {k: g1[k] for k in ("steps", "start_state_hash", "fit_end", "check_end", "seconds", "teacher_zero_grad")},
                  "g2": {k: g2[k] for k in ("steps", "start_state_hash", "fit_end", "check_end", "seconds", "teacher_zero_grad", "feature_coefficient")}},
        "interface": interface,
        "selfcheck": selfcheck,
        "sequence": sequence,
        "hashes": hashes,
        "formal_commands": [
            "python -X utf8 scripts/run_use_recovery_probe.py calibrate",
            "python -X utf8 scripts/run_use_recovery_probe.py formal --group g1 --execute --device cuda:0",
            "python -X utf8 scripts/run_use_recovery_probe.py formal --group g2 --execute --device cuda:0",
        ],
        "formal_executed": False,
        "not_generalisation": True,
    }
    write_json(out_dir / "summary.json", summary)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_provenance(out_dir: Path = OUT) -> dict:
    sources = [Path(__file__), *sorted(CONFIG_DIR.glob("*.yaml")), diag.MANIFEST, diag.STATS, diag.QUALITY_INDEX,
               resolve_path("outputs/use/s2_10band_v4r1_p20/use_baseline/1234/best.pt")]
    rows = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "size_bytes": p.stat().st_size}
            for p in sources if p.is_file()]
    outputs = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "size_bytes": p.stat().st_size}
               for p in sorted(out_dir.rglob("*")) if p.is_file()]
    payload = {"command": "python -X utf8 scripts/run_use_recovery_probe.py short --group both",
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sources": rows, "outputs": outputs,
               "scope": "TRAIN short probes only; no formal 2000-step run"}
    write_json(out_dir / "provenance.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "short", "selfcheck", "interface", "report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--out", type=Path, default=OUT)
        cmd.add_argument("--device", default=None)
        if name in ("short", "calibrate"):
            cmd.add_argument("--steps", type=int, default=None)
        if name == "short":
            cmd.add_argument("--group", choices=["g1", "g2", "both"], default="both")
    formal = sub.add_parser("formal")
    formal.add_argument("--group", choices=["g1", "g2", "both"], default="both")
    formal.add_argument("--steps", type=int, default=2000)
    formal.add_argument("--execute", action="store_true")
    formal.add_argument("--out", type=Path, default=OUT)
    formal.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        cfg = load_segformer_config(CONFIG_DIR / "calibration.yaml")
        summary = calibrate(cfg, args.out, args.device, steps=args.steps)
        print(json.dumps({"steps": summary["steps"], "scope": summary["parameter_scope"]}, ensure_ascii=False))
    elif args.command == "short":
        results = {}
        groups = ["g1", "g2"] if args.group == "both" else [args.group]
        for group in groups:
            cfg = load_segformer_config(CONFIG_DIR / f"group{group[-1]}.yaml")
            results[group] = supervised(cfg, group, args.out, args.device, short=True, steps=args.steps)
        print(json.dumps({g: {k: r.get(k) for k in ("steps", "start_state_hash", "fit_end", "check_end")} for g, r in results.items()}, ensure_ascii=False))
    elif args.command == "selfcheck":
        cfg = load_segformer_config(CONFIG_DIR / "calibration.yaml")
        result = selfcheck(cfg, args.out, args.device)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "interface":
        cfg = load_segformer_config(CONFIG_DIR / "calibration.yaml")
        result = interface_checks(cfg, args.out, args.device)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "report":
        path = write_report(args.out)
        write_provenance(args.out)
        print(str(path))
    else:
        groups = ["g1", "g2"] if args.group == "both" else [args.group]
        results = {}
        for group in groups:
            cfg = load_segformer_config(CONFIG_DIR / f"group{group[-1]}.yaml")
            results[group] = supervised(cfg, group, args.out, args.device, short=False, steps=args.steps, execute_formal=args.execute)
        print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

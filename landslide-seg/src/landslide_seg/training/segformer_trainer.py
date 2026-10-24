"""SegFormer 训练器：单输出二分类 + masked BCE/Dice + 逐 epoch 验证。

设计要点（对应执行计划 v2 §7）：
- 损失直接复用 ``training/losses.py`` 里已按 ``valid`` 归一的 ``bce_loss`` / ``dice_loss``；
- 优化器三分组：编码器 / 解码器 / 新增层（首层新增通道、投影、Stem、融合头），
  两组学习率，其余超参全组一致；
- 每个 epoch 在验证集上按**前景 IoU** 保存 best，另存 last；
- checkpoint 里带完整配置、清单/统计/权重身份、种子与 RNG 状态，支持 resume；
- 数据增强的随机流由 ``(seed, epoch, index)`` 决定，与模型初始化无关，
  因此新增 Stem 不会改变样本顺序。

所有相对路径都相对项目根目录解析，脚本从哪个 cwd 启动都一样。
"""

from __future__ import annotations

import json
import math
import random
import importlib.metadata
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import (
    PROJECT_ROOT,
    file_sha256,
    manifest_identity,
    resolve_path,
    set_seed,
)
from ..data.segformer_dataset import (
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from ..models.factory import build_model, model_identity, model_type
from .budget import resolve_update_budget
from .losses import bce_loss, dice_loss
from .metrics import summarize

DEFAULT_PRETRAINED = "weights/segformer/segformer-b0-finetuned-ade-512-512"


# --------------------------------------------------------------------------
# 身份与工具
# --------------------------------------------------------------------------
def weight_identity(pretrained_dir: str | Path) -> dict:
    """记录权重目录的 revision、文件大小与**完整** SHA-256。"""
    p = resolve_path(pretrained_dir)
    out: dict[str, Any] = {
        "dir": str(p).replace("\\", "/"),
        "exists": p.is_dir(),
        "revision": None,
        "files": {},
    }
    if not p.is_dir():
        return out
    for name in ("model.safetensors", "config.json"):
        f = p / name
        if f.is_file():
            out["files"][name] = {"size_bytes": f.stat().st_size, "sha256": file_sha256(f)}
    cache = p / ".cache" / "huggingface" / "download"
    if cache.is_dir():
        for meta in sorted(cache.glob("*.metadata")):
            try:
                first = meta.read_text(encoding="utf-8").splitlines()[0].strip()
            except Exception:
                continue
            if len(first) >= 40:
                out["revision"] = first
                break
    return out


def _device(name: str | None) -> torch.device:
    if name:
        dev = torch.device(name)
    else:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA 设备，但 torch.cuda.is_available()=False")
    return dev


def _run_dir(cfg: dict) -> Path:
    s = cfg["segformer"]
    return resolve_path(s["output_dir"]) / s["data"]["version"] / s["experiment"] / str(s["seed"])


def _make_optimizer(model: SegFormerSeg, cfg: dict) -> torch.optim.Optimizer:
    """三分组 + 可选的 T1 首层适配学习率。

    ``optim.first_conv_lr`` 只作用于 direct10 的 patch_embeddings[0].proj
    （即被替换成 10 通道的那一层），其余编码器参数仍用 ``encoder_lr``。
    这是为了检验"新增通道学习率过高/过低"是否与 A/B 组的差异混杂。
    """
    optim = cfg["segformer"]["optim"]
    groups = model.param_groups()
    encoder = list(groups["encoder"])
    first_lr = optim.get("first_conv_lr")
    if first_lr is not None:
        if getattr(model, "adapter_type", None) != "direct10":
            raise ValueError(
                "optim.first_conv_lr 只适用于 direct10（project10to3 没有替换 patch_embeddings）"
            )
        first_ids = {
            id(p) for n, p in model.named_parameters() if "patch_embeddings.0.proj" in n
        }
        first = [p for p in encoder if id(p) in first_ids]
        rest = [p for p in encoder if id(p) not in first_ids]
        if not first:
            raise ValueError(
                "optim.first_conv_lr 只适用于 direct10（project10to3 没有 patch_embeddings 替换）"
            )
        params = [
            {"params": rest, "lr": float(optim["encoder_lr"])},
            {"params": first, "lr": float(first_lr)},
            {"params": groups["decoder"] + groups["new"], "lr": float(optim["head_lr"])},
        ]
    else:
        params = [
            {"params": encoder, "lr": float(optim["encoder_lr"])},
            {"params": groups["decoder"] + groups["new"], "lr": float(optim["head_lr"])},
        ]
    return torch.optim.AdamW(params, weight_decay=float(optim["weight_decay"]))


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, total_steps: int):
    optim = cfg["segformer"]["optim"]
    warmup = max(1, int(total_steps * float(optim["warmup_frac"])))
    power = float(optim.get("poly_power", 1.0))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, (1.0 - progress) ** power)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_loader(ds: S2SegDataset, batch_size: int, shuffle: bool, seed: int, epoch: int) -> DataLoader:
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(int(seed) * 100_003 + int(epoch))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate,
        generator=gen,
        drop_last=False,
    )


# --------------------------------------------------------------------------
# T2：事件 / 尺度均衡采样
# --------------------------------------------------------------------------
def _scale_bin(mask_px: int) -> str:
    if mask_px <= 0:
        return "empty"
    if mask_px < 50:
        return "small"
    if mask_px < 200:
        return "medium"
    return "large"


def _balanced_weights(
    records: list[dict],
    *,
    event_power: float = 0.5,
    scale_power: float = 0.5,
    max_weight: float = 4.0,
) -> np.ndarray:
    """权重 ∝ (1/事件频次)^a · (1/尺度档频次)^b，归一化并截断到 [1/max, max]。"""
    from collections import Counter

    if not records:
        return np.zeros(0, dtype=np.float64)
    ev = Counter(r["event_id"] for r in records)
    bins = [_scale_bin(int(r.get("mask_px", 0))) for r in records]
    bc = Counter(bins)
    w = np.asarray(
        [
            (1.0 / ev[r["event_id"]]) ** float(event_power)
            * (1.0 / bc[b]) ** float(scale_power)
            for r, b in zip(records, bins)
        ],
        dtype=np.float64,
    )
    w = w / max(w.mean(), 1e-12)
    w = np.clip(w, 1.0 / float(max_weight), float(max_weight))
    return w / w.sum()


def _balanced_indices(records: list[dict], cfg: dict, seed: int, epoch: int) -> list[int]:
    sampling = cfg["segformer"].get("sampling", {})
    w = _balanced_weights(
        records,
        event_power=float(sampling.get("event_power", 0.5)),
        scale_power=float(sampling.get("scale_power", 0.5)),
        max_weight=float(sampling.get("max_weight", 4.0)),
    )
    rng = np.random.default_rng((int(seed) * 1_000_003 + int(epoch) * 7_919) % (2**63 - 1))
    return rng.choice(len(records), size=len(records), replace=True, p=w).tolist()


def _make_train_loader(
    ds: S2SegDataset,
    records: list[dict],
    cfg: dict,
    batch_size: int,
    seed: int,
    epoch: int,
) -> DataLoader:
    mode = str(cfg["segformer"].get("sampling", {}).get("mode", "natural"))
    if mode == "balanced":
        indices = _balanced_indices(records, cfg, seed, epoch)
        return DataLoader(
            ds,
            batch_size=int(batch_size),
            sampler=indices,
            num_workers=0,
            collate_fn=collate,
            drop_last=False,
        )
    return _make_loader(ds, batch_size, True, seed, epoch)


# --------------------------------------------------------------------------
# 单步 / 单 epoch
# --------------------------------------------------------------------------
def _amp_context(cfg: dict, device: torch.device):
    train_cfg = cfg["segformer"]["train"]
    enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    dtype = torch.bfloat16 if str(train_cfg.get("amp_dtype", "bf16")) == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _boundary_weight(mask: torch.Tensor, width: int, beta: float) -> torch.Tensor:
    """GT 边界带权重：band = dilate(gt) - erode(gt)，返回 ``1 + beta*band``。"""
    g = mask.float().unsqueeze(1)
    k = 2 * int(width) + 1
    dil = F.max_pool2d(g, k, stride=1, padding=int(width))
    ero = -F.max_pool2d(-g, k, stride=1, padding=int(width))
    band = (dil - ero).clamp(0.0, 1.0)
    return (1.0 + float(beta) * band).squeeze(1)


def _seg_loss(
    cfg: dict,
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    boundary: bool = False,
) -> torch.Tensor:
    """valid-aware BCE+Dice；``boundary=True`` 时 BCE 按 GT 边界带加权。"""
    loss_cfg = cfg["segformer"]["loss"]
    logits = logits.float()
    if logits.ndim == 4:
        logits = logits[:, 0]
    bw = float(loss_cfg.get("boundary_weight", 0.0) or 0.0)
    if boundary and bw > 0:
        w = _boundary_weight(mask, int(loss_cfg.get("boundary_width", 2)), bw)
        v = valid.float()
        raw = F.binary_cross_entropy_with_logits(logits, mask.float(), reduction="none")
        bce = (raw * w * v).sum() / (w * v).sum().clamp_min(1.0)
    else:
        bce = bce_loss(logits, mask, valid, pos_weight=float(loss_cfg.get("pos_weight", 1.0) or 1.0))
    dice = dice_loss(logits, mask, valid)
    return float(loss_cfg["bce_weight"]) * bce + float(loss_cfg["dice_weight"]) * dice


def _loss(
    cfg: dict,
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    base_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """最终输出 + 可选粗定位辅助监督：``L = Lseg(F) + λc·Lseg(up(C))``。"""
    fine = _seg_loss(cfg, logits, mask, valid, boundary=True)
    coarse_w = float(cfg["segformer"]["loss"].get("coarse_weight", 0.0) or 0.0)
    parts = {"fine": fine, "coarse": None}
    if coarse_w > 0:
        if base_logits is None:
            raise RuntimeError("coarse_weight>0 但模型没有 base_logits")
        if base_logits is logits:
            raise RuntimeError("U-Net 没有独立粗输出，不能启用 coarse_weight")
        coarse = _seg_loss(cfg, base_logits, mask, valid)
        parts["coarse"] = coarse
        return fine + coarse_w * coarse, parts
    return fine, parts


def _train_epoch(
    model: SegFormerSeg,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
    device: torch.device,
) -> tuple[float, int, int]:
    model.train()
    if cfg['segformer']['train'].get('freeze_bn_running_stats', False):
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    total, seen, skipped, steps = 0.0, 0, 0, 0
    coarse_total = 0.0
    clip = float(cfg["segformer"]["optim"]["grad_clip"])
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        if int(valid.sum()) == 0:
            skipped += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(cfg, device):
            out = model(image)
        loss, parts = _loss(cfg, out["logits"], mask, valid, out.get("base_logits"))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        if parts["coarse"] is not None:
            coarse_total += float(parts["coarse"].detach()) * image.size(0)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        total += float(loss.detach()) * image.size(0)
        seen += image.size(0)
        steps += 1
    return total / max(seen, 1), skipped, steps, coarse_total / max(seen, 1)


@torch.no_grad()
def evaluate_loader(
    model: SegFormerSeg,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """按像素混淆计数聚合验证指标（无效像素真正剔除）。"""
    from .metrics import confusion

    model.eval()
    threshold = float(cfg["segformer"]["eval"]["threshold"])
    counters = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        logits = model(image)["logits"].float()
        prob = torch.sigmoid(logits[:, 0])
        pred = (prob >= threshold).cpu().numpy()
        gt = mask.cpu().numpy().astype(bool)
        v = valid.cpu().numpy().astype(bool)
        for i in range(pred.shape[0]):
            counters.append(confusion(pred[i], gt[i], v[i]))
    return summarize(counters)


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------
def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def _save_checkpoint(
    path: Path,
    model: SegFormerSeg,
    optimizer,
    scheduler,
    epoch: int,
    best_iou: float,
    best_epoch: int,
    cfg: dict,
    meta: dict,
    seed_info: dict,
    extra: dict | None = None,
) -> None:
    payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_val_iou": float(best_iou),
            "best_epoch": int(best_epoch),
            "config": cfg,
            "meta": meta,
            "seed_info": seed_info,
            "rng": _rng_state(),
            "params": model.count_params(),
            "adapter_report": model.adapter_report,
            "load_info": model.load_info,
        }
    payload.update(extra or {})
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str | None = None) -> tuple[SegFormerSeg, dict]:
    """从 checkpoint 重建模型（全新实例，用于评估/重载测试）。"""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    expected = ckpt.get("meta", {}).get("stats_sha256")
    if expected and file_sha256(resolve_path(cfg["segformer"]["data"]["stats"])) != expected:
        raise ValueError("Normalization statistics changed since checkpoint was saved")
    model = build_model(cfg["segformer"])
    model.load_state_dict(ckpt["model"])
    if hasattr(model, "probe_report"):
        model.probe_report = ckpt["meta"]["model_identity"]["probe"]
    dev = _device(device)
    model.to(dev)
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def train_one(
    cfg: dict,
    device: str | None = None,
    resume: bool = False,
    log=print,
) -> dict:
    """跑一次完整训练，返回摘要（同时把产物写到 run 目录）。"""
    s = cfg["segformer"]
    d = s["data"]
    seed = int(s["seed"])
    seed_info = set_seed(seed, bool(s.get("deterministic", True)))
    dev = _device(device)

    manifest = resolve_path(d["manifest"])
    stats_path = resolve_path(d["stats"])
    pretrained = resolve_path(s["model"].get("pretrained_dir", DEFAULT_PRETRAINED))

    if not manifest.is_file():
        raise FileNotFoundError(f"清单不存在: {manifest}（先跑 scripts/prepare_segformer_data.py）")
    if not stats_path.is_file():
        raise FileNotFoundError(f"统计文件不存在: {stats_path}")
    if model_type(s) in ("segformer", "single", "use") and not pretrained.is_dir():
        raise FileNotFoundError(f"预训练权重目录不存在: {pretrained}")
    if model_type(s) == "single" and float(s["loss"].get("coarse_weight", 0.0) or 0.0) > 0:
        raise ValueError("单头模型（type=single）没有粗输出，不能启用 loss.coarse_weight")

    records = read_manifest(manifest)
    stats = load_stats(stats_path)
    from .protocol import validate_protocol
    protocol_info = validate_protocol(cfg, records, stats)
    train_recs = filter_split(records, "train")
    val_recs = filter_split(records, "val")
    if not train_recs:
        raise RuntimeError("train split 为空：检查清单的 split 字段")
    if not val_recs:
        raise RuntimeError("val split 为空：检查 configs/segformer 的 val_event")

    run_dir = _run_dir(cfg)
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty; choose a new experiment or resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(s, pretrained_dir=pretrained).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    train_cfg = s["train"]
    preload = bool(d.get("preload", True))
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    max_steps = train_cfg.get("max_steps")
    steps_per_epoch = math.ceil(len(train_recs) / batch_size)
    # 更新制：停止步数与 poly 日程长度分开；日程长度可能大于停止步数。
    budget = resolve_update_budget(train_cfg) if str(train_cfg.get("budget_mode")) == "updates" else None
    total_steps = budget.schedule_updates if budget else (int(max_steps) if max_steps else steps_per_epoch * epochs)

    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg, total_steps)

    extra = str(s["model"].get("extra_channel", "none"))
    sidecar = d.get("extra_sidecar")
    dataset_class = S2SegDataset
    if model_type(s) == 'use':
        from ..data.use_dataset import USESegDataset
        dataset_class = USESegDataset
    train_ds = dataset_class(train_recs, stats, augment=True, seed=seed, preload=preload,
                            extra=extra, sidecar=sidecar)
    val_ds = dataset_class(val_recs, stats, augment=False, seed=seed, preload=preload,
                          extra=extra, sidecar=sidecar)
    if model_type(s) == 'use':
        train_ds.mixed = bool(d.get('mixed_bands', False))

    meta = {
        "experiment": s["experiment"],
        "seed": seed,
        "seed_info": seed_info,
        "data_version": d["version"],
        "manifest": manifest_identity(manifest),
        "stats_sha256": file_sha256(stats_path),
        "protocol": protocol_info,
        "source_sha256": {
            str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(p)
            for p in sorted((PROJECT_ROOT / "src/landslide_seg").rglob("*.py"))
        },
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "scipy", "PyYAML")},
        "use_quality_sha256": file_sha256(train_ds.quality_root / 'index.json') if hasattr(train_ds, 'quality_root') else None,
        "model_identity": model_identity(s, model),
        "weight": weight_identity(pretrained) if (model_type(s) in ("segformer", "single", "use") and pretrained.is_dir()) else None,
        "params": model.count_params(),
        "adapter_report": model.adapter_report,
        "load_info": model.load_info,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "budget": budget.as_dict() if budget else None,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "preload": preload,
        "extra_channel": extra,
        "extra_sidecar": sidecar,
        "sampling": s.get("sampling", {}),
        "first_conv_lr": s["optim"].get("first_conv_lr"),
        "train_samples": [r["sample_id"] for r in train_recs],
        "val_samples": [r["sample_id"] for r in val_recs],
    }
    if train_cfg.get("budget_mode") == "updates":
        from .update_training import run_updates
        return run_updates(cfg, model, optimizer, scheduler, train_ds, val_ds, train_recs, val_recs,
                           stats, meta, seed_info, run_dir, dev, resume, log)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    start_epoch = 0
    best_iou = -1.0
    best_epoch = -1
    history: list[dict] = []
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    if resume and last_path.is_file():
        ckpt = torch.load(str(last_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_iou = float(ckpt["best_val_iou"])
        best_epoch = int(ckpt["best_epoch"])
        rng = ckpt.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        log(f"[resume] {last_path} -> epoch {start_epoch}")

    log(
        f"[run] {s['experiment']} seed={seed} device={dev} "
        f"train={len(train_recs)} val={len(val_recs)} batch={batch_size} "
        f"epochs={epochs} steps/epoch={steps_per_epoch} total_steps={total_steps}"
    )
    log(f"[run] out={run_dir}")

    metrics_path = run_dir / "metrics.jsonl"
    if not resume:
        metrics_path.write_text("", encoding="utf-8")
    started = time.time()
    global_step = 0
    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        train_loader = _make_train_loader(train_ds, train_recs, cfg, batch_size, seed, epoch)
        val_loader = _make_loader(val_ds, batch_size, False, seed, 0)
        t0 = time.time()
        train_loss, skipped, steps, coarse_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, cfg, dev
        )
        global_step += steps
        val = evaluate_loader(model, val_loader, cfg, dev)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "coarse_loss": round(coarse_loss, 5) if coarse_loss else None,
            "skipped_batches": skipped,
            "steps": steps,
            "global_step": global_step,
            "lr": [round(float(g["lr"]), 8) for g in optimizer.param_groups],
            "val_iou": round(float(val.get("iou") or 0.0), 5),
            "val_dice": round(float(val.get("dice") or 0.0), 5),
            "val_precision": round(float(val.get("precision") or 0.0), 5),
            "val_recall": round(float(val.get("recall") or 0.0), 5),
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(
            f"[epoch {epoch:3d}] loss={row['train_loss']:.4f} "
            f"val_iou={row['val_iou']:.4f} val_dice={row['val_dice']:.4f} "
            f"({row['seconds']:.1f}s)"
        )

        _save_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best_iou, best_epoch,
            cfg, meta, seed_info,
        )
        if row["val_iou"] > best_iou:
            best_iou = row["val_iou"]
            best_epoch = epoch
            _save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best_iou, best_epoch,
                cfg, meta, seed_info,
            )
            log(f"  [best] val_iou={best_iou:.4f} @ epoch {epoch}")
        if max_steps and global_step >= int(max_steps):
            log(f"[stop] 达到 max_steps={max_steps}")
            break

    elapsed = time.time() - started
    summary = {
        "experiment": s["experiment"],
        "seed": seed,
        "run_dir": str(run_dir).replace("\\", "/"),
        "device": str(dev),
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "params": model.count_params(),
        "best_epoch": best_epoch,
        "best_val_iou": best_iou,
        "epochs_run": len(history),
        "global_step": global_step,
        "seconds": round(elapsed, 2),
        "seconds_per_epoch": round(elapsed / max(len(history), 1), 2),
        "best_checkpoint": str(best_path).replace("\\", "/"),
        "last_checkpoint": str(last_path).replace("\\", "/"),
        "peak_alloc_mib": round(torch.cuda.max_memory_allocated(dev) / 2**20, 1) if dev.type == "cuda" else None,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(dev) / 2**20, 1) if dev.type == "cuda" else None,
        "history": history,
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"[done] {s['experiment']} seed={seed} best_epoch={best_epoch} "
        f"best_val_iou={best_iou:.4f} ({elapsed/60:.1f} min)"
    )
    return summary

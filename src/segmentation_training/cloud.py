from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import csv
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import load_config
from .manifest import filter_records, load_model_input_manifest, split_band_reference, validate_record_raster_grids
from .metrics import (
    compute_confusion_matrix,
    decode_prediction,
    encode_label,
    per_class_metrics,
    prediction_summary_rows,
    probability_summary,
    summarize_area,
    threshold_sweep,
)
from .models import build_model
from .preview import write_contact_sheet, write_prediction_preview

P14_HARD_WINDOW_BASELINE_IOU = {
    "sample-000946": 0.0,
    "sample-000979": 0.0,
    "sample-001041": 0.0,
    "sample-001221": 0.0,
    "sample-001225": 0.0,
    "sample-001227": 0.0,
    "sample-001230": 0.0,
    "sample-001232": 0.022123893805309734,
    "sample-001267": 0.0,
    "sample-001268": 0.0,
    "sample-001281": 0.0,
    "sample-001282": 0.0,
    "sample-001315": 0.0,
    "sample-001350": 0.0,
}


def refuse_local_training() -> str:
    return (
        "Refusing to run training without --cloud-confirm. "
        "Full E1-E3 training is cloud-only for SegmentationTrainingBaseline v0.1."
    )


def prepare_cloud_run(
    *,
    config_path: Path,
    bundle_dir: Path,
    run_dir: Path,
    command_line: list[str],
) -> dict:
    config = load_config(config_path)
    manifest_path = bundle_dir / "manifests" / "cloud_model_input_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cloud manifest not found: {manifest_path}")
    records = load_model_input_manifest(manifest_path, required_channels=tuple(config["input_channels"]))
    scoped_records = filter_records(records, class_scope=config["class_scope"])
    selection_records = [record for record in scoped_records if record.split != "test"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "task_id": config["task_id"],
        "contract_task_id": "task-ef3a9c94ddb7",
        "experiment_id": config["experiment_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "bundle_dir": str(bundle_dir),
        "run_dir": str(run_dir),
        "seed": config.get("seed"),
        "model": config["model"],
        "input_channels": config["input_channels"],
        "class_scope": config["class_scope"],
        "split_counts": _split_counts(selection_records),
        "test_split_read": False,
        "command_line": command_line,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "cloud_hardware_summary": "record_on_cloud_before_training",
        "status": "prepared_cloud_training_harness",
    }
    resume_metadata = _resolve_resume_checkpoint(config)
    if resume_metadata:
        manifest.update(resume_metadata)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def run_cloud_training(
    *,
    config_path: Path,
    bundle_dir: Path,
    run_dir: Path,
    command_line: list[str],
) -> dict:
    import numpy as np
    import torch
    import rasterio
    from rasterio.windows import Window as RasterWindow

    from .experiments import trivial_background_prediction
    from .losses import build_segmentation_loss, class_weights_from_counts
    config = load_config(config_path)
    manifest = prepare_cloud_run(
        config_path=config_path,
        bundle_dir=bundle_dir,
        run_dir=run_dir,
        command_line=command_line,
    )
    records = load_model_input_manifest(
        bundle_dir / "manifests" / "cloud_model_input_manifest.csv",
        required_channels=tuple(config["input_channels"]),
    )
    scoped_records = filter_records(records, class_scope=config["class_scope"])
    train_records = _limit_records(
        [record for record in scoped_records if record.split == "train"],
        config.get("cloud", {}).get("max_train_samples"),
    )
    validation_records = [record for record in scoped_records if record.split == "validation"]
    if not validation_records:
        raise ValueError("No validation records available for cloud training")
    grid_errors = validate_record_raster_grids(
        [*train_records, *validation_records],
        contract_dir=bundle_dir,
        required_channels=tuple(config["input_channels"]),
    )
    if grid_errors:
        raise ValueError("Raster grid validation failed:\n" + "\n".join(grid_errors[:20]))
    progress = _TrainingProgressLogger(run_dir)
    performance_config = config.get("training", {}).get("performance", {})
    record_cache = None
    if bool(performance_config.get("cache_records", False)):
        progress.log(
            "caching train/validation rasters in memory "
            f"record_count={len(train_records) + len(validation_records)} channels={len(config['input_channels'])}"
        )
        record_cache = _build_record_cache(
            [*train_records, *validation_records],
            bundle_dir=bundle_dir,
            channels=tuple(config["input_channels"]),
            rasterio=rasterio,
            np=np,
        )
        cached_bytes = sum(
            int(item["channels"].nbytes) + int(item["label"].nbytes) for item in record_cache.values()
        )
        progress.log(f"cached records ready approx_bytes={cached_bytes}")

    source_class_ids = _source_class_ids(config)
    encoded_class_ids = list(range(len(source_class_ids)))
    max_samples = int(config.get("cloud", {}).get("max_validation_samples", 32))

    model_family = config["model"]["family"]
    progress.log(
        f"starting experiment={config['experiment_id']} model={model_family} "
        f"train_records={len(train_records)} validation_records={len(validation_records)} "
        f"max_validation_samples={max_samples}"
    )
    if model_family == "trivial":
        metrics_payload = _evaluate_trivial(
            validation_records[:max_samples],
            bundle_dir=bundle_dir,
            config=config,
            source_class_ids=source_class_ids,
            rasterio=rasterio,
            trivial_background_prediction=trivial_background_prediction,
            np=np,
        )
        metrics_payload["foreground_window_coverage"] = _foreground_window_coverage_rows(
            [*train_records, *validation_records],
            bundle_dir=bundle_dir,
            foreground_class_ids=[class_id for class_id in source_class_ids if class_id != 0],
            window_size=int(config.get("training", {}).get("window_size", 256)),
            rasterio=rasterio,
            np=np,
        )
        progress.log_epoch_metrics(
            epoch=1,
            max_epochs=1,
            train_loss=0.0,
            mean_iou=metrics_payload["mean_iou"],
            foreground_recall=metrics_payload["foreground_recall"],
        )
        _write_run_outputs(run_dir, metrics_payload, source_class_ids)
        _maybe_write_training_curves(run_dir)
        manifest["status"] = "completed_trivial_baseline"
        _write_manifest(run_dir, manifest)
        return manifest

    if not train_records:
        raise ValueError("No training records available for cloud training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_trainable_model(config, source_class_ids=source_class_ids)
    model.to(device)
    runtime = _configure_torch_runtime(torch, device=device, performance_config=performance_config)
    if runtime["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    resume_metadata = _resolve_resume_checkpoint(config)
    if resume_metadata:
        _load_resume_checkpoint(model, resume_metadata, device=device, torch=torch)
        progress.log(
            "loaded resume checkpoint "
            f"path={resume_metadata['resume_from_checkpoint']} "
            f"source_epoch={resume_metadata.get('resume_source_epoch', '')}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("training", {}).get("learning_rate", 0.001)))
    class_weights = None
    if config.get("training", {}).get("loss") in {"weighted_cross_entropy", "weighted_cross_entropy_dice"}:
        class_weights = class_weights_from_counts(
            _encoded_label_counts(
                train_records,
                bundle_dir=bundle_dir,
                source_class_ids=source_class_ids,
                rasterio=rasterio,
                np=np,
            ),
            max_weight=float(config.get("training", {}).get("class_weights", {}).get("max_weight", 20.0)),
        )
    loss_fn = build_segmentation_loss(
        loss_name=config.get("training", {}).get("loss", "cross_entropy"),
        ignore_index=int(config["metrics"].get("ignore_index", 255)),
        class_weights=class_weights,
        dice_weight=float(config.get("training", {}).get("dice_weight", 1.0)),
    )
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    max_epochs = int(config.get("training", {}).get("max_epochs", 1))
    batch_size = int(config.get("training", {}).get("batch_size", 4))
    window_size = int(config.get("training", {}).get("window_size", 256))
    log_interval = int(config.get("cloud", {}).get("log_interval_batches", 10))
    validation_interval = int(config.get("cloud", {}).get("validation_interval_epochs", 1))
    rng = np.random.default_rng(int(config.get("seed", 20260505)))
    total_batches = (len(train_records) + batch_size - 1) // batch_size
    best_checkpoint = _BestCheckpointTracker(run_dir, metric_name="mean_iou")

    losses: list[float] = []
    for epoch in range(1, max_epochs + 1):
        rng.shuffle(train_records)
        batch_x = []
        batch_y = []
        epoch_losses: list[float] = []
        batch_index = 0
        progress.log(f"epoch {epoch}/{max_epochs} started")
        for record in train_records:
            channels, label = _read_training_window(
                record,
                bundle_dir=bundle_dir,
                channels=tuple(config["input_channels"]),
                window_size=window_size,
                sampler_config=config.get("training", {}).get("sampler", {}),
                foreground_class_ids=[class_id for class_id in source_class_ids if class_id != 0],
                rng=rng,
                rasterio=rasterio,
                RasterWindow=RasterWindow,
                np=np,
                record_cache=record_cache,
            )
            channels, label = _apply_training_augmentation(
                channels,
                label,
                config.get("training", {}).get("augmentation", {}),
                rng=rng,
                np=np,
            )
            encoded_label = _encode_label(label, source_class_ids, np=np)
            if not _target_has_valid_pixels(encoded_label, np=np):
                continue
            batch_x.append(torch.from_numpy(channels.astype("float32")))
            batch_y.append(torch.from_numpy(encoded_label.astype("int64")))
            if len(batch_x) == batch_size:
                loss_value = _train_batch(
                    model,
                    optimizer,
                    loss_fn,
                    batch_x,
                    batch_y,
                    device,
                    torch,
                    use_amp=runtime["mixed_precision"],
                    channels_last=runtime["channels_last"],
                )
                losses.append(loss_value)
                epoch_losses.append(loss_value)
                batch_index += 1
                if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                    progress.log_batch(epoch, max_epochs, batch_index, total_batches, loss_value, epoch_losses)
                batch_x = []
                batch_y = []
        if batch_x:
            loss_value = _train_batch(
                model,
                optimizer,
                loss_fn,
                batch_x,
                batch_y,
                device,
                torch,
                use_amp=runtime["mixed_precision"],
                channels_last=runtime["channels_last"],
            )
            losses.append(loss_value)
            epoch_losses.append(loss_value)
            batch_index += 1
            progress.log_batch(epoch, max_epochs, batch_index, total_batches, loss_value, epoch_losses)
        if epoch % validation_interval == 0 or epoch == max_epochs:
            epoch_metrics = _evaluate_model(
                model,
                validation_records[:max_samples],
                bundle_dir=bundle_dir,
                config=config,
                source_class_ids=source_class_ids,
                device=device,
                rasterio=rasterio,
                RasterWindow=RasterWindow,
                torch=torch,
                np=np,
                use_amp=runtime["mixed_precision"],
                channels_last=runtime["channels_last"],
                record_cache=record_cache,
                preview_config={"max_items": 0, "worst_false_positive_items": 0},
            )
            progress.log_epoch_metrics(
                epoch=epoch,
                max_epochs=max_epochs,
                train_loss=float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                mean_iou=epoch_metrics["mean_iou"],
                foreground_recall=epoch_metrics["foreground_recall"],
            )
            if best_checkpoint.update(epoch=epoch, metrics=epoch_metrics, model=model, torch=torch):
                progress.log(
                    f"epoch {epoch}/{max_epochs} saved best_mean_iou="
                    f"{epoch_metrics['mean_iou']:.4f}"
                )

    metrics_payload = _evaluate_model(
        model,
        validation_records[:max_samples],
        bundle_dir=bundle_dir,
        config=config,
        source_class_ids=source_class_ids,
        device=device,
        rasterio=rasterio,
        RasterWindow=RasterWindow,
        torch=torch,
        np=np,
        use_amp=runtime["mixed_precision"],
        channels_last=runtime["channels_last"],
        record_cache=record_cache,
        preview_config=config.get("metrics", {}).get("preview", {}),
    )
    metrics_payload["foreground_window_coverage"] = _foreground_window_coverage_rows(
        [*train_records, *validation_records],
        bundle_dir=bundle_dir,
        foreground_class_ids=[class_id for class_id in source_class_ids if class_id != 0],
        window_size=window_size,
        rasterio=rasterio,
        np=np,
    )
    metrics_payload["train_loss_last"] = losses[-1] if losses else None
    _write_run_outputs(run_dir, metrics_payload, source_class_ids, preview_config=config.get("metrics", {}).get("preview", {}))
    _maybe_write_training_curves(run_dir)
    torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")
    manifest["status"] = "completed_cloud_training"
    manifest["cloud_hardware_summary"] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "mixed_precision": runtime["mixed_precision"],
        "channels_last": runtime["channels_last"],
        "cudnn_benchmark": runtime["cudnn_benchmark"],
    }
    _write_manifest(run_dir, manifest)
    return manifest


def run_channel_contribution(
    *,
    config_path: Path,
    bundle_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict:
    import numpy as np
    import torch
    import rasterio
    from rasterio.windows import Window as RasterWindow

    config = load_config(config_path)
    manifest_path = bundle_dir / "manifests" / "cloud_model_input_manifest.csv"
    records = load_model_input_manifest(manifest_path, required_channels=tuple(config["input_channels"]))
    scoped_records = filter_records(records, class_scope=config["class_scope"])
    validation_records = [record for record in scoped_records if record.split == "validation"]
    max_samples = int(config.get("cloud", {}).get("max_validation_samples", 32))
    validation_records = validation_records[:max_samples]
    grid_errors = validate_record_raster_grids(
        validation_records,
        contract_dir=bundle_dir,
        required_channels=tuple(config["input_channels"]),
    )
    if grid_errors:
        raise ValueError("Raster grid validation failed:\n" + "\n".join(grid_errors[:20]))

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_class_ids = [int(class_id) for class_id in config["metrics"]["class_ids"]]
    model = _build_trainable_model(config, source_class_ids=source_class_ids)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    performance_config = config.get("training", {}).get("performance", {})
    runtime = _configure_torch_runtime(torch, device=device, performance_config=performance_config)
    if runtime["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    record_cache = None
    if bool(performance_config.get("cache_records", False)):
        record_cache = _build_record_cache(
            validation_records,
            bundle_dir=bundle_dir,
            channels=tuple(config["input_channels"]),
            rasterio=rasterio,
            np=np,
        )

    preview_config = {"max_items": 0, "worst_false_positive_items": 0}
    baseline = _evaluate_model(
        model,
        validation_records,
        bundle_dir=bundle_dir,
        config=config,
        source_class_ids=source_class_ids,
        device=device,
        rasterio=rasterio,
        RasterWindow=RasterWindow,
        torch=torch,
        np=np,
        use_amp=runtime["mixed_precision"],
        channels_last=runtime["channels_last"],
        record_cache=record_cache,
        preview_config=preview_config,
    )
    contribution_config = config.get("metrics", {}).get("channel_contribution", {})
    mode = str(contribution_config.get("mode", "zero"))
    groups = _channel_contribution_groups(tuple(config["input_channels"]), contribution_config.get("groups"))
    rows = [_channel_contribution_row("baseline", [], baseline, baseline)]
    for name, channels in groups:
        payload = _evaluate_model(
            model,
            validation_records,
            bundle_dir=bundle_dir,
            config=config,
            source_class_ids=source_class_ids,
            device=device,
            rasterio=rasterio,
            RasterWindow=RasterWindow,
            torch=torch,
            np=np,
            use_amp=runtime["mixed_precision"],
            channels_last=runtime["channels_last"],
            record_cache=record_cache,
            preview_config=preview_config,
            channel_perturbation={"channels": channels, "mode": mode},
        )
        rows.append(_channel_contribution_row(name, channels, payload, baseline))

    _write_dict_rows(output_dir / "channel_contribution.csv", rows)
    manifest = {
        "status": "completed_channel_contribution",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "bundle_dir": str(bundle_dir),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "experiment_id": config["experiment_id"],
        "input_channels": config["input_channels"],
        "perturbation_mode": mode,
        "validation_records": len(validation_records),
        "test_split_read": False,
        "cloud_hardware_summary": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "mixed_precision": runtime["mixed_precision"],
            "channels_last": runtime["channels_last"],
            "cudnn_benchmark": runtime["cudnn_benchmark"],
        },
    }
    (output_dir / "channel_contribution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _build_trainable_model(config: dict, *, source_class_ids: list[int]):
    return build_model(
        config["model"],
        input_channels=len(config["input_channels"]),
        output_classes=len(source_class_ids),
    )


def _apply_training_augmentation(channels, label, augmentation_config: dict | None, *, rng, np):
    augmentation_config = augmentation_config or {}
    if not bool(augmentation_config.get("enabled", False)):
        return channels, label
    augmented_channels = channels
    augmented_label = label
    if rng.random() < float(augmentation_config.get("horizontal_flip_probability", 0.0)):
        augmented_channels = augmented_channels[:, :, ::-1]
        augmented_label = augmented_label[:, ::-1]
    if rng.random() < float(augmentation_config.get("vertical_flip_probability", 0.0)):
        augmented_channels = augmented_channels[:, ::-1, :]
        augmented_label = augmented_label[::-1, :]
    if rng.random() < float(augmentation_config.get("rotate90_probability", 0.0)):
        k = int(rng.integers(1, 4))
        augmented_channels = np.rot90(augmented_channels, k=k, axes=(-2, -1))
        augmented_label = np.rot90(augmented_label, k=k, axes=(0, 1))
    noise_std = float(augmentation_config.get("gaussian_noise_std", 0.0))
    if noise_std > 0:
        augmented_channels = augmented_channels + rng.normal(0.0, noise_std, size=augmented_channels.shape)
    return np.ascontiguousarray(augmented_channels), np.ascontiguousarray(augmented_label)


def _apply_channel_perturbation(channels, *, input_channels: tuple[str, ...], perturbation: dict | None, np):
    if not perturbation:
        return channels
    selected = set(perturbation.get("channels", []))
    if not selected:
        return channels
    mode = str(perturbation.get("mode", "zero"))
    perturbed = channels.copy()
    for index, channel in enumerate(input_channels):
        if channel not in selected:
            continue
        if mode == "zero":
            perturbed[index] = 0
        elif mode == "mean":
            perturbed[index] = float(np.mean(perturbed[index]))
        else:
            raise ValueError(f"Unsupported channel perturbation mode {mode!r}")
    return perturbed


def _channel_contribution_groups(input_channels: tuple[str, ...], groups_config) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    if isinstance(groups_config, dict):
        for name, channels in groups_config.items():
            selected = [channel for channel in channels if channel in input_channels]
            if selected:
                groups.append((str(name), selected))
    groups.extend((channel, [channel]) for channel in input_channels)
    seen = set()
    unique_groups = []
    for name, channels in groups:
        key = (name, tuple(channels))
        if key in seen:
            continue
        seen.add(key)
        unique_groups.append((name, channels))
    return unique_groups


def _channel_contribution_row(name: str, channels: list[str], payload: dict, baseline: dict) -> dict:
    metrics = _foreground_payload_metrics(payload)
    baseline_metrics = _foreground_payload_metrics(baseline)
    return {
        "group": name,
        "channels": ";".join(channels),
        "mean_iou": metrics["mean_iou"],
        "foreground_precision": metrics["foreground_precision"],
        "foreground_recall": metrics["foreground_recall"],
        "foreground_dice": metrics["foreground_dice"],
        "mean_iou_drop": baseline_metrics["mean_iou"] - metrics["mean_iou"],
        "foreground_dice_drop": baseline_metrics["foreground_dice"] - metrics["foreground_dice"],
        "pred_label_area_ratio": metrics["pred_label_area_ratio"],
    }


def _foreground_payload_metrics(payload: dict) -> dict[str, float]:
    foreground = [metric for metric in payload.get("per_class", []) if metric.class_id != 0]
    precision = float(sum(metric.precision for metric in foreground) / len(foreground)) if foreground else 0.0
    recall = float(sum(metric.recall for metric in foreground) / len(foreground)) if foreground else 0.0
    dice = float(sum(metric.dice for metric in foreground) / len(foreground)) if foreground else 0.0
    area = payload.get("area", {})
    label_pixels = sum(item.get("label_pixels", 0) for class_id, item in area.items() if int(class_id) != 0)
    predicted_pixels = sum(item.get("predicted_pixels", 0) for class_id, item in area.items() if int(class_id) != 0)
    return {
        "mean_iou": float(payload.get("mean_iou", 0.0)),
        "foreground_precision": precision,
        "foreground_recall": recall,
        "foreground_dice": dice,
        "pred_label_area_ratio": float(predicted_pixels / label_pixels) if label_pixels else 0.0,
    }


def _limit_records(records, max_samples):
    if max_samples is None or int(max_samples) <= 0:
        return records
    return records[: int(max_samples)]


class _TrainingProgressLogger:
    def __init__(self, run_dir: Path, *, emit: Callable[[str], None] = print):
        self.emit = emit
        self.path = run_dir / "logs" / "training_progress.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        self.emit(line)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log_batch(
        self,
        epoch: int,
        max_epochs: int,
        batch_index: int,
        total_batches: int,
        loss_value: float,
        epoch_losses: list[float],
    ) -> None:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        pct = batch_index / max(1, total_batches) * 100
        self.log(
            f"epoch {epoch}/{max_epochs} batch {batch_index}/{total_batches} "
            f"({pct:.1f}%) loss={loss_value:.4f} avg_loss={avg_loss:.4f}"
        )

    def log_epoch_metrics(
        self,
        *,
        epoch: int,
        max_epochs: int,
        train_loss: float,
        mean_iou: float,
        foreground_recall: float,
    ) -> None:
        self.log(
            f"epoch {epoch}/{max_epochs} validation "
            f"loss={train_loss:.4f} mean_iou={mean_iou:.4f} foreground_recall={foreground_recall:.4f}"
        )


class _BestCheckpointTracker:
    def __init__(self, run_dir: Path, *, metric_name: str):
        self.run_dir = run_dir
        self.metric_name = metric_name
        self.best_value: float | None = None

    def update(self, *, epoch: int, metrics: dict, model, torch) -> bool:
        value = float(metrics.get(self.metric_name, float("-inf")))
        if self.best_value is not None and value <= self.best_value:
            return False
        self.best_value = value
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"best_{self.metric_name}.pt")
        metadata = {
            "epoch": epoch,
            "metric_name": self.metric_name,
            "metric_value": value,
            "mean_iou": metrics.get("mean_iou"),
            "foreground_recall": metrics.get("foreground_recall"),
        }
        (checkpoint_dir / f"best_{self.metric_name}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return True


def _source_class_ids(config: dict) -> list[int]:
    scope = config["class_scope"]
    if scope == "binary_c2":
        return [0, 1]
    if scope == "binary_c5":
        return [0, 2]
    return [0, 1, 2]


def _configure_torch_runtime(torch, *, device, performance_config: dict | None = None) -> dict:
    performance_config = performance_config or {}
    cuda_enabled = str(device).startswith("cuda")
    cudnn_benchmark = bool(performance_config.get("cudnn_benchmark", False)) and cuda_enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return {
        "mixed_precision": bool(performance_config.get("mixed_precision", False)) and cuda_enabled,
        "channels_last": bool(performance_config.get("channels_last", False)) and cuda_enabled,
        "cudnn_benchmark": cudnn_benchmark,
    }


def _train_batch(
    model,
    optimizer,
    loss_fn,
    batch_x,
    batch_y,
    device,
    torch,
    *,
    use_amp: bool = False,
    channels_last: bool = False,
) -> float:
    model.train()
    inputs = torch.stack(batch_x).to(device)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    labels = torch.stack(batch_y).to(device)
    if not bool((labels != 255).any().item()):
        return 0.0
    optimizer.zero_grad()
    with _autocast(torch, enabled=use_amp):
        loss = loss_fn(model(inputs), labels)
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("Non-finite training loss detected; check raster normalization and labels")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _build_record_cache(records, *, bundle_dir: Path, channels: tuple[str, ...], rasterio, np) -> dict:
    cache = {}
    for record in records:
        mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
        with rasterio.open(mask_path) as mask_ds:
            label = mask_ds.read(1)
        band_arrays = []
        for channel in channels:
            band_path = _resolve_cloud_or_bundle_path(bundle_dir, record.input_band_paths[channel])
            band_arrays.append(_read_input_band(band_path, rasterio=rasterio, np=np))
        cache[record.sample_id] = {
            "channels": np.stack(band_arrays, axis=0),
            "label": label,
        }
    return cache


def _autocast(torch, *, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _read_training_window(
    record,
    *,
    bundle_dir: Path,
    channels: tuple[str, ...],
    window_size: int,
    sampler_config: dict | None = None,
    foreground_class_ids: list[int] | None = None,
    rng,
    rasterio,
    RasterWindow,
    np,
    record_cache: dict | None = None,
):
    from .sampler import choose_training_window

    sampler_config = sampler_config or {}
    foreground_class_ids = foreground_class_ids or []
    cached = record_cache.get(record.sample_id) if record_cache else None
    if cached is not None:
        label = cached["label"]
        height, width = label.shape
        size = _safe_window_size(height, width, window_size)
        if sampler_config.get("train_policy") == "foreground_biased":
            selected = choose_training_window(
                label,
                size=size,
                rng=rng,
                foreground_class_ids=foreground_class_ids,
                foreground_probability=float(sampler_config.get("foreground_probability", 0.75)),
                min_foreground_pixels=int(sampler_config.get("min_foreground_pixels", 1)),
            )
        else:
            selected = choose_training_window(
                label,
                size=size,
                rng=rng,
                foreground_class_ids=[],
                foreground_probability=0.0,
                min_foreground_pixels=0,
            )
        return _slice_cached_window(cached, selected, window_size, np=np)
    mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
    with rasterio.open(mask_path) as mask_ds:
        height = mask_ds.height
        width = mask_ds.width
        size = _safe_window_size(height, width, window_size)
        if sampler_config.get("train_policy") == "foreground_biased":
            full_label = mask_ds.read(1)
            selected = choose_training_window(
                full_label,
                size=size,
                rng=rng,
                foreground_class_ids=foreground_class_ids,
                foreground_probability=float(sampler_config.get("foreground_probability", 0.75)),
                min_foreground_pixels=int(sampler_config.get("min_foreground_pixels", 1)),
            )
            window = RasterWindow(selected.col, selected.row, selected.width, selected.height)
        else:
            row = int(rng.integers(0, max(1, height - size + 1)))
            col = int(rng.integers(0, max(1, width - size + 1)))
            window = RasterWindow(col, row, size, size)
        label = mask_ds.read(1, window=window)
    band_arrays = []
    for channel in channels:
        band_path = _resolve_cloud_or_bundle_path(bundle_dir, record.input_band_paths[channel])
        band_arrays.append(_read_input_band(band_path, window=window, rasterio=rasterio, np=np))
    channels_array = np.stack(band_arrays, axis=0)
    return _pad_window(channels_array, window_size, fill_value=0, np=np), _pad_window(
        label, window_size, fill_value=255, np=np
    )


def _read_validation_window(
    record,
    *,
    bundle_dir,
    channels,
    window_size,
    sampler_config: dict | None = None,
    foreground_class_ids: list[int] | None = None,
    rasterio,
    RasterWindow,
    np,
    record_cache: dict | None = None,
):
    from .sampler import choose_validation_window

    sampler_config = sampler_config or {}
    foreground_class_ids = foreground_class_ids or []
    cached = record_cache.get(record.sample_id) if record_cache else None
    if cached is not None:
        label = cached["label"]
        size = _safe_window_size(label.shape[0], label.shape[1], window_size)
        selected = choose_validation_window(
            label,
            size=size,
            foreground_class_ids=foreground_class_ids,
            policy=str(sampler_config.get("validation_policy", "center")),
        )
        return _slice_cached_window(cached, selected, window_size, np=np)
    mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
    with rasterio.open(mask_path) as mask_ds:
        size = _safe_window_size(mask_ds.height, mask_ds.width, window_size)
        full_label = mask_ds.read(1)
        selected = choose_validation_window(
            full_label,
            size=size,
            foreground_class_ids=foreground_class_ids,
            policy=str(sampler_config.get("validation_policy", "center")),
        )
        window = RasterWindow(selected.col, selected.row, selected.width, selected.height)
        label = mask_ds.read(1, window=window)
    band_arrays = []
    for channel in channels:
        band_path = _resolve_cloud_or_bundle_path(bundle_dir, record.input_band_paths[channel])
        band_arrays.append(_read_input_band(band_path, window=window, rasterio=rasterio, np=np))
    channels_array = np.stack(band_arrays, axis=0)
    return _pad_window(channels_array, window_size, fill_value=0, np=np), _pad_window(
        label, window_size, fill_value=255, np=np
    )


def _slice_cached_window(cached: dict, window, target_size: int, *, np):
    row_end = window.row + window.height
    col_end = window.col + window.width
    channels_array = cached["channels"][:, window.row : row_end, window.col : col_end]
    label = cached["label"][window.row : row_end, window.col : col_end]
    return _pad_window(channels_array, target_size, fill_value=0, np=np), _pad_window(
        label, target_size, fill_value=255, np=np
    )


def _safe_window_size(height: int, width: int, requested: int) -> int:
    size = min(height, width, requested)
    size = max(4, size - (size % 4))
    return size


def _pad_window(array, target_size: int, *, fill_value: int | float, np):
    if array.ndim == 2:
        height, width = array.shape
        output = np.full((target_size, target_size), fill_value, dtype=array.dtype)
        copy_h = min(height, target_size)
        copy_w = min(width, target_size)
        output[:copy_h, :copy_w] = array[:copy_h, :copy_w]
        return output
    if array.ndim == 3:
        channels, height, width = array.shape
        output = np.full((channels, target_size, target_size), fill_value, dtype=array.dtype)
        copy_h = min(height, target_size)
        copy_w = min(width, target_size)
        output[:, :copy_h, :copy_w] = array[:, :copy_h, :copy_w]
        return output
    raise ValueError(f"Unsupported window array shape {array.shape}")


def _normalize_band(array, *, np):
    arr = array.astype("float32")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype="float32")
    lo = float(np.percentile(arr[finite], 2))
    hi = float(np.percentile(arr[finite], 98))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1).astype("float32")
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0).astype("float32")


def _read_input_band(path_reference, *, rasterio, np, window=None):
    path_value, band_index = split_band_reference(str(path_reference))
    with rasterio.open(path_value) as band_ds:
        return _normalize_band(band_ds.read(band_index, window=window), np=np)


def _target_has_valid_pixels(target, *, np, ignore_index: int = 255) -> bool:
    return bool(np.any(target != ignore_index))


def _encode_label(label, source_class_ids: list[int], *, np):
    return encode_label(label, source_class_ids)


def _decode_prediction(encoded_prediction, source_class_ids: list[int], *, np):
    return decode_prediction(encoded_prediction, source_class_ids)


def _evaluate_model(
    model,
    records,
    *,
    bundle_dir,
    config,
    source_class_ids,
    device,
    rasterio,
    RasterWindow,
    torch,
    np,
    use_amp: bool = False,
    channels_last: bool = False,
    record_cache: dict | None = None,
    preview_config: dict | None = None,
    channel_perturbation: dict | None = None,
):
    model.eval()
    labels = []
    predictions = []
    probabilities = []
    sample_ids = []
    preview_items = []
    foreground_class_ids = [class_id for class_id in source_class_ids if class_id != 0]
    preview_limit = _preview_limit(preview_config or {})
    collect_all_for_worst = int((preview_config or {}).get("worst_false_positive_items", 0) or 0) > 0
    with torch.no_grad():
        for record in records:
            channels, label = _read_validation_window(
                record,
                bundle_dir=bundle_dir,
                channels=tuple(config["input_channels"]),
                window_size=int(config.get("training", {}).get("window_size", 256)),
                sampler_config=config.get("training", {}).get("sampler", {}),
                foreground_class_ids=foreground_class_ids,
                rasterio=rasterio,
                RasterWindow=RasterWindow,
                np=np,
                record_cache=record_cache,
            )
            channels = _apply_channel_perturbation(
                channels,
                input_channels=tuple(config["input_channels"]),
                perturbation=channel_perturbation,
                np=np,
            )
            inputs = torch.from_numpy(channels[None, ...].astype("float32")).to(device)
            if channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with _autocast(torch, enabled=use_amp):
                logits = model(inputs)
            foreground_probability = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
            foreground_probability = foreground_probability.astype("float32", copy=False)
            encoded_prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")
            prediction = _decode_prediction(encoded_prediction, source_class_ids, np=np)
            labels.append(label)
            predictions.append(prediction)
            sample_ids.append(record.sample_id)
            probabilities.append(
                foreground_probability[1] if foreground_probability.shape[0] > 1 else foreground_probability[0]
            )
            if _should_collect_preview(len(preview_items), preview_limit, collect_all_for_worst):
                preview_items.append((record.sample_id, channels, label, prediction))
    return _build_metrics_payload(
        labels,
        predictions,
        source_class_ids,
        preview_items,
        sample_ids=sample_ids,
        foreground_probabilities=probabilities,
        thresholds=list(config.get("metrics", {}).get("threshold_sweep", [0.5])),
        np=np,
    )


def _evaluate_trivial(
    records,
    *,
    bundle_dir,
    config,
    source_class_ids,
    rasterio,
    trivial_background_prediction,
    np,
):
    labels = []
    predictions = []
    sample_ids = []
    preview_paths = []
    for record in records:
        mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
        with rasterio.open(mask_path) as mask_ds:
            label = mask_ds.read(1)
        prediction = trivial_background_prediction(label)
        labels.append(label)
        predictions.append(prediction)
        sample_ids.append(record.sample_id)
        if len(preview_paths) < 12:
            preview_paths.append((record.sample_id, np.stack([label, label], axis=0), label, prediction))
    return _build_metrics_payload(labels, predictions, source_class_ids, preview_paths, sample_ids=sample_ids, np=np)


def _build_metrics_payload(
    labels,
    predictions,
    source_class_ids,
    preview_items,
    *,
    sample_ids=None,
    foreground_probabilities=None,
    thresholds=None,
    np,
):
    label_array = np.concatenate([label.ravel() for label in labels])
    prediction_array = np.concatenate([prediction.ravel() for prediction in predictions])
    matrix = compute_confusion_matrix(label_array, prediction_array, class_ids=source_class_ids)
    class_metrics = per_class_metrics(matrix, class_ids=source_class_ids)
    mean_iou = float(np.mean([metric.iou for metric in class_metrics[1:]])) if len(class_metrics) > 1 else 0.0
    foreground = [metric for metric in class_metrics if metric.class_id != 0]
    foreground_recall = float(np.mean([metric.recall for metric in foreground])) if foreground else 0.0
    sample_ids = sample_ids or [f"sample-{idx:04d}" for idx in range(len(labels))]
    foreground_class_ids = [class_id for class_id in source_class_ids if class_id != 0]
    prediction_summary = prediction_summary_rows(
        labels,
        predictions,
        sample_ids=sample_ids,
        foreground_class_ids=foreground_class_ids,
    )
    threshold_rows = []
    probability_rows = []
    if foreground_probabilities and len(foreground_class_ids) == 1:
        foreground_class_id = foreground_class_ids[0]
        for sample_id, label, probability in zip(sample_ids, labels, foreground_probabilities):
            probability_rows.append(
                {
                    "sample_id": sample_id,
                    **probability_summary(label, probability, foreground_class_id=foreground_class_id),
                }
            )
            for row in threshold_sweep(
                label,
                probability,
                foreground_class_id=foreground_class_id,
                thresholds=thresholds or [0.5],
            ):
                threshold_rows.append({"sample_id": sample_id, **row})
    return {
        "confusion_matrix": matrix,
        "per_class": class_metrics,
        "mean_iou": mean_iou,
        "foreground_recall": foreground_recall,
        "area": summarize_area(label_array, prediction_array, class_ids=source_class_ids),
        "preview_items": preview_items,
        "prediction_summary": prediction_summary,
        "threshold_sweep": threshold_rows,
        "foreground_probability_summary": probability_rows,
    }


def _write_run_outputs(
    run_dir: Path,
    payload: dict,
    source_class_ids: list[int],
    *,
    preview_config: dict | None = None,
) -> None:
    preview_config = preview_config or {}
    metrics = {
        "mean_iou": payload["mean_iou"],
        "foreground_recall": payload["foreground_recall"],
        "class_ids": source_class_ids,
    }
    if "train_loss_last" in payload:
        metrics["train_loss_last"] = payload["train_loss_last"]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_per_class_metrics(run_dir / "per_class_metrics.csv", payload["per_class"])
    _write_confusion_matrix(run_dir / "confusion_matrix.csv", payload["confusion_matrix"], source_class_ids)
    _write_area_summary(run_dir / "mask_area_summary.csv", payload["area"])
    _write_event_metrics_placeholder(run_dir / "event_metrics.csv", metrics)
    diagnostics_dir = run_dir / "diagnostics"
    _write_dict_rows(diagnostics_dir / "validation_prediction_summary.csv", payload.get("prediction_summary", []))
    _write_dict_rows(diagnostics_dir / "threshold_sweep.csv", payload.get("threshold_sweep", []))
    _write_dict_rows(
        diagnostics_dir / "foreground_probability_summary.csv", payload.get("foreground_probability_summary", [])
    )
    _write_dict_rows(diagnostics_dir / "foreground_window_coverage.csv", payload.get("foreground_window_coverage", []))
    _write_p14_gate_reports(run_dir, payload.get("prediction_summary", []))
    _write_binary_encode_decode_audit(diagnostics_dir / "binary_c5_encode_decode_audit.json", source_class_ids)
    preview_limit = _preview_limit(preview_config)
    preview_items = payload.get("preview_items", [])
    standard_preview_items = preview_items if preview_limit is None else preview_items[:preview_limit]
    preview_paths = []
    per_sample_dir = run_dir / "predictions" / "per_sample"
    for sample_id, channels, label, prediction in standard_preview_items:
        preview_paths.append(
            write_prediction_preview(
                channels=channels,
                label=label,
                prediction=prediction,
                output_path=per_sample_dir / f"{sample_id}.png",
                title=sample_id,
            )
        )
    if preview_paths:
        write_contact_sheet(preview_paths, run_dir / "predictions" / "validation_contact_sheet.png", columns=2)
        write_contact_sheet(preview_paths[: min(6, len(preview_paths))], run_dir / "predictions" / "failures_contact_sheet.png", columns=2)
    worst_rows = _select_worst_false_positive_rows(
        payload.get("prediction_summary", []),
        max_items=int(preview_config.get("worst_false_positive_items", 0) or 0),
    )
    _write_dict_rows(diagnostics_dir / "worst_false_positives.csv", worst_rows)
    worst_paths = _write_worst_false_positive_previews(
        run_dir,
        preview_items,
        worst_rows,
    )
    if worst_paths:
        write_contact_sheet(worst_paths, run_dir / "predictions" / "worst_false_positives_contact_sheet.png", columns=2)
    _write_artifact_inventory(diagnostics_dir / "artifact_inventory.json", run_dir)


def _resolve_resume_checkpoint(config: dict) -> dict:
    checkpoint_value = config.get("training", {}).get("resume_from_checkpoint")
    if checkpoint_value is None:
        return {}
    expanded = os.path.expandvars(str(checkpoint_value)).strip()
    if "$" in expanded:
        raise FileNotFoundError(f"Resume checkpoint contains unresolved environment variable: {checkpoint_value}")
    checkpoint_path = Path(expanded).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    sidecar_path = checkpoint_path.with_suffix(".json")
    source_epoch = None
    source_metric_value = None
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        source_epoch = sidecar.get("epoch")
        source_metric_value = sidecar.get("metric_value", sidecar.get("mean_iou"))
    return {
        "resume_from_checkpoint": str(checkpoint_path),
        "resume_source_epoch": source_epoch,
        "resume_source_metric_value": source_metric_value,
        "fine_tune": True,
    }


def _load_resume_checkpoint(model, resume_metadata: dict, *, device, torch) -> None:
    checkpoint_path = Path(resume_metadata["resume_from_checkpoint"])
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state)
    except Exception as exc:
        raise ValueError(f"Resume checkpoint is incompatible with model: {checkpoint_path}") from exc


def _write_p14_gate_reports(run_dir: Path, prediction_rows: list[dict]) -> None:
    by_sample_id = {str(row.get("sample_id", "")): row for row in prediction_rows}
    hard_rows = []
    for sample_id, baseline_iou in P14_HARD_WINDOW_BASELINE_IOU.items():
        row = by_sample_id.get(sample_id, {})
        current_iou = _optional_float(row.get("foreground_iou"))
        if current_iou is None:
            status = "missing"
        elif current_iou > baseline_iou:
            status = "improved"
        elif current_iou < baseline_iou:
            status = "regressed"
        else:
            status = "unchanged"
        hard_rows.append(
            {
                "sample_id": sample_id,
                "baseline_iou": baseline_iou,
                "current_iou": "" if current_iou is None else current_iou,
                "status": status,
                "zero_iou_after": current_iou == 0.0 if current_iou is not None else "",
                "label_foreground_pixels": row.get("label_foreground_pixels", ""),
                "predicted_foreground_pixels": row.get("predicted_foreground_pixels", ""),
                "predicted_label_area_ratio": row.get("predicted_label_area_ratio", ""),
                "foreground_precision": row.get("foreground_precision", ""),
                "foreground_recall": row.get("foreground_recall", ""),
            }
        )
    _write_dict_rows(run_dir / "diagnostics" / "p14_hard_window_gate.csv", hard_rows)
    _write_dict_rows(run_dir / "diagnostics" / "p14_tiny_bin_gate.csv", _p14_tiny_bin_rows(prediction_rows))


def _p14_tiny_bin_rows(prediction_rows: list[dict]) -> list[dict]:
    bins = [
        ("fg_lt_50", lambda value: value < 50),
        ("fg_lt_100", lambda value: value < 100),
        ("fg_lt_500", lambda value: value < 500),
    ]
    rows = []
    for name, predicate in bins:
        selected = []
        for row in prediction_rows:
            foreground_pixels = _optional_float(row.get("label_foreground_pixels"))
            if foreground_pixels is None or not predicate(foreground_pixels):
                continue
            iou = _optional_float(row.get("foreground_iou"))
            if iou is not None:
                selected.append(iou)
        rows.append(
            {
                "bin": name,
                "window_count": len(selected),
                "mean_iou": _mean(selected),
                "median_iou": _median(selected),
                "zero_iou_windows": sum(1 for value in selected if value == 0.0),
            }
        )
    return rows


def _optional_float(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def _median(values: list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _maybe_write_training_curves(run_dir: Path) -> dict:
    from .training_curves import write_training_curves

    log_path = run_dir / "logs" / "training_progress.log"
    if not log_path.exists():
        return {"status": "skipped", "reason": "missing_training_progress_log"}
    try:
        result = write_training_curves(run_dir)
    except ValueError as exc:
        return {"status": "skipped", "reason": str(exc)}
    inventory_path = run_dir / "diagnostics" / "artifact_inventory.json"
    if inventory_path.exists():
        _write_artifact_inventory(inventory_path, run_dir)
    return {"status": "written", **result}


def _preview_limit(preview_config: dict) -> int | None:
    value = preview_config.get("max_items", 12)
    if isinstance(value, str) and value.lower() == "all":
        return None
    return max(0, int(value))


def _should_collect_preview(current_count: int, preview_limit: int | None, collect_all_for_worst: bool) -> bool:
    if collect_all_for_worst:
        return True
    if preview_limit is None:
        return True
    return current_count < preview_limit


def _select_worst_false_positive_rows(rows: list[dict], *, max_items: int) -> list[dict]:
    if max_items <= 0:
        return []
    candidates = [row for row in rows if int(float(row.get("fp", 0) or 0)) > 0]
    ranked = sorted(
        candidates,
        key=lambda row: (
            int(float(row.get("fp", 0) or 0)),
            float(row.get("predicted_label_area_ratio", 0.0) or 0.0),
            -float(row.get("foreground_precision", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected = []
    for rank, row in enumerate(ranked[:max_items], start=1):
        item = dict(row)
        item["rank"] = rank
        selected.append(item)
    return selected


def _write_worst_false_positive_previews(run_dir: Path, preview_items: list, worst_rows: list[dict]) -> list[Path]:
    if not worst_rows:
        return []
    by_sample_id = {sample_id: (channels, label, prediction) for sample_id, channels, label, prediction in preview_items}
    output_dir = run_dir / "predictions" / "worst_false_positives"
    output_paths = []
    for row in worst_rows:
        sample_id = row.get("sample_id")
        if sample_id not in by_sample_id:
            continue
        channels, label, prediction = by_sample_id[sample_id]
        output_paths.append(
            write_prediction_preview(
                channels=channels,
                label=label,
                prediction=prediction,
                output_path=output_dir / f"{int(row['rank']):02d}__{sample_id}.png",
                title=f"worst-fp-{row['rank']} {sample_id}",
            )
        )
    return output_paths


def _write_per_class_metrics(path: Path, metrics) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class_id", "tp", "fp", "fn", "iou", "dice", "precision", "recall"])
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)


def _write_confusion_matrix(path: Path, matrix, class_ids: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label\\prediction", *class_ids])
        for class_id, row in zip(class_ids, matrix.tolist()):
            writer.writerow([class_id, *row])


def _write_area_summary(path: Path, area: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class_id", "label_pixels", "predicted_pixels"])
        writer.writeheader()
        for class_id, row in area.items():
            writer.writerow({"class_id": class_id, **row})


def _write_event_metrics_placeholder(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scope", "mean_iou", "foreground_recall"])
        writer.writeheader()
        writer.writerow(
            {
                "scope": "validation_windows",
                "mean_iou": metrics["mean_iou"],
                "foreground_recall": metrics["foreground_recall"],
            }
        )


def _write_dict_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_artifact_inventory(path: Path, run_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = [
        "run_manifest.json",
        "resolved_config.yaml",
        "metrics.json",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "mask_area_summary.csv",
        "event_metrics.csv",
        "diagnostics/validation_prediction_summary.csv",
        "diagnostics/threshold_sweep.csv",
        "diagnostics/foreground_probability_summary.csv",
        "diagnostics/foreground_window_coverage.csv",
        "diagnostics/worst_false_positives.csv",
        "diagnostics/p14_hard_window_gate.csv",
        "diagnostics/p14_tiny_bin_gate.csv",
        "diagnostics/binary_c5_encode_decode_audit.json",
        "training_curves.png",
        "training_curves.csv",
        "predictions/validation_contact_sheet.png",
        "predictions/worst_false_positives_contact_sheet.png",
        "checkpoints/best_mean_iou.pt",
        "checkpoints/best_mean_iou.json",
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_split_read": False,
        "artifacts": [{"path": item, "exists": (run_dir / item).exists()} for item in expected],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_binary_encode_decode_audit(path: Path, source_class_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_class_ids": source_class_ids,
        "ignore_index": 255,
        "status": "not_binary_c5",
    }
    if source_class_ids == [0, 2]:
        import numpy as np

        source = np.array([[0, 2, 255], [1, 2, 0]], dtype=np.uint8)
        encoded = encode_label(source, source_class_ids).tolist()
        decoded = decode_prediction(np.array(encoded, dtype=np.uint8), source_class_ids).tolist()
        payload.update(
            {
                "status": "passed",
                "source_example": source.tolist(),
                "encoded_example": encoded,
                "decoded_example": decoded,
                "out_of_scope_source_class_becomes_ignore": encoded[1][0] == 255,
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _encoded_label_counts(records, *, bundle_dir, source_class_ids: list[int], rasterio, np) -> list[int]:
    counts = [0 for _ in source_class_ids]
    for record in records:
        mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
        with rasterio.open(mask_path) as mask_ds:
            encoded = _encode_label(mask_ds.read(1), source_class_ids, np=np)
        valid = encoded != 255
        for index in range(len(source_class_ids)):
            counts[index] += int(np.logical_and(valid, encoded == index).sum())
    return counts


def _foreground_window_coverage_rows(records, *, bundle_dir, foreground_class_ids: list[int], window_size: int, rasterio, np):
    from .sampler import foreground_window_coverage

    samples = []
    split_by_sample_id = {}
    for record in records:
        if record.split == "test":
            continue
        mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
        with rasterio.open(mask_path) as mask_ds:
            samples.append((record.sample_id, mask_ds.read(1)))
        split_by_sample_id[record.sample_id] = record.split
    rows = foreground_window_coverage(samples, foreground_class_ids=foreground_class_ids, size=window_size)
    for row in rows:
        row["split"] = split_by_sample_id.get(row["sample_id"], "")
    return rows


def _resolve_cloud_or_bundle_path(bundle_dir: Path, value: str, *, subdir: str | None = None) -> Path:
    physical_value, band_index = split_band_reference(value)
    path = Path(physical_value)
    suffix = "" if band_index == 1 and "#band=" not in value else f"#band={band_index}"
    if path.exists():
        return Path(str(path) + suffix)
    if subdir is not None:
        mirror = bundle_dir / subdir / path.name
        if mirror.exists():
            return Path(str(mirror) + suffix)
    relative = bundle_dir / physical_value
    if relative.exists():
        return Path(str(relative) + suffix)
    return Path(str(path) + suffix)


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _split_counts(records) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    return counts


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"

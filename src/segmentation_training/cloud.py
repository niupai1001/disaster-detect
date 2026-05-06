from __future__ import annotations

import json
import platform
import subprocess
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import load_config
from .manifest import filter_records, load_model_input_manifest
from .metrics import compute_confusion_matrix, per_class_metrics, summarize_area
from .preview import write_contact_sheet, write_prediction_preview


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
        "split_counts": _split_counts(scoped_records),
        "command_line": command_line,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "cloud_hardware_summary": "record_on_cloud_before_training",
        "status": "prepared_cloud_training_harness",
    }
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
    from .losses import build_cross_entropy_loss
    from .models import build_resunet, build_unet

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
    train_records = [record for record in scoped_records if record.split == "train"]
    validation_records = [record for record in scoped_records if record.split == "validation"]
    if not validation_records:
        raise ValueError("No validation records available for cloud training")
    progress = _TrainingProgressLogger(run_dir)

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
        _write_run_outputs(run_dir, metrics_payload, source_class_ids)
        progress.log_epoch_metrics(
            epoch=1,
            max_epochs=1,
            train_loss=0.0,
            mean_iou=metrics_payload["mean_iou"],
            foreground_recall=metrics_payload["foreground_recall"],
        )
        manifest["status"] = "completed_trivial_baseline"
        _write_manifest(run_dir, manifest)
        return manifest

    if not train_records:
        raise ValueError("No training records available for cloud training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_family == "unet":
        model = build_unet(
            input_channels=len(config["input_channels"]),
            output_classes=len(source_class_ids),
            base_channels=int(config["model"].get("base_channels", 32)),
        )
    elif model_family == "resunet":
        model = build_resunet(
            input_channels=len(config["input_channels"]),
            output_classes=len(source_class_ids),
            base_channels=int(config["model"].get("base_channels", 32)),
        )
    else:
        raise ValueError(f"Unsupported cloud model family {model_family!r}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("training", {}).get("learning_rate", 0.001)))
    loss_fn = build_cross_entropy_loss(ignore_index=int(config["metrics"].get("ignore_index", 255)))
    max_epochs = int(config.get("training", {}).get("max_epochs", 1))
    batch_size = int(config.get("training", {}).get("batch_size", 4))
    window_size = int(config.get("training", {}).get("window_size", 256))
    log_interval = int(config.get("cloud", {}).get("log_interval_batches", 10))
    validation_interval = int(config.get("cloud", {}).get("validation_interval_epochs", 1))
    rng = np.random.default_rng(int(config.get("seed", 20260505)))
    total_batches = (len(train_records) + batch_size - 1) // batch_size

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
                rng=rng,
                rasterio=rasterio,
                RasterWindow=RasterWindow,
                np=np,
            )
            encoded_label = _encode_label(label, source_class_ids, np=np)
            batch_x.append(torch.from_numpy(channels.astype("float32")))
            batch_y.append(torch.from_numpy(encoded_label.astype("int64")))
            if len(batch_x) == batch_size:
                loss_value = _train_batch(model, optimizer, loss_fn, batch_x, batch_y, device, torch)
                losses.append(loss_value)
                epoch_losses.append(loss_value)
                batch_index += 1
                if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                    progress.log_batch(epoch, max_epochs, batch_index, total_batches, loss_value, epoch_losses)
                batch_x = []
                batch_y = []
        if batch_x:
            loss_value = _train_batch(model, optimizer, loss_fn, batch_x, batch_y, device, torch)
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
            )
            progress.log_epoch_metrics(
                epoch=epoch,
                max_epochs=max_epochs,
                train_loss=float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                mean_iou=epoch_metrics["mean_iou"],
                foreground_recall=epoch_metrics["foreground_recall"],
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
    )
    metrics_payload["train_loss_last"] = losses[-1] if losses else None
    _write_run_outputs(run_dir, metrics_payload, source_class_ids)
    torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")
    manifest["status"] = "completed_cloud_training"
    manifest["cloud_hardware_summary"] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_manifest(run_dir, manifest)
    return manifest


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


def _source_class_ids(config: dict) -> list[int]:
    scope = config["class_scope"]
    if scope == "binary_c2":
        return [0, 1]
    if scope == "binary_c5":
        return [0, 2]
    return [0, 1, 2]


def _train_batch(model, optimizer, loss_fn, batch_x, batch_y, device, torch) -> float:
    model.train()
    inputs = torch.stack(batch_x).to(device)
    labels = torch.stack(batch_y).to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(inputs), labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _read_training_window(
    record,
    *,
    bundle_dir: Path,
    channels: tuple[str, ...],
    window_size: int,
    rng,
    rasterio,
    RasterWindow,
    np,
):
    mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
    with rasterio.open(mask_path) as mask_ds:
        height = mask_ds.height
        width = mask_ds.width
        size = _safe_window_size(height, width, window_size)
        row = int(rng.integers(0, max(1, height - size + 1)))
        col = int(rng.integers(0, max(1, width - size + 1)))
        window = RasterWindow(col, row, size, size)
        label = mask_ds.read(1, window=window)
    band_arrays = []
    for channel in channels:
        band_path = _resolve_cloud_or_bundle_path(bundle_dir, record.input_band_paths[channel])
        with rasterio.open(band_path) as band_ds:
            band_arrays.append(_normalize_band(band_ds.read(1, window=window), np=np))
    channels_array = np.stack(band_arrays, axis=0)
    return _pad_window(channels_array, window_size, fill_value=0, np=np), _pad_window(
        label, window_size, fill_value=255, np=np
    )


def _read_validation_window(record, *, bundle_dir, channels, window_size, rasterio, RasterWindow, np):
    mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
    with rasterio.open(mask_path) as mask_ds:
        size = _safe_window_size(mask_ds.height, mask_ds.width, window_size)
        row = max(0, (mask_ds.height - size) // 2)
        col = max(0, (mask_ds.width - size) // 2)
        window = RasterWindow(col, row, size, size)
        label = mask_ds.read(1, window=window)
    band_arrays = []
    for channel in channels:
        band_path = _resolve_cloud_or_bundle_path(bundle_dir, record.input_band_paths[channel])
        with rasterio.open(band_path) as band_ds:
            band_arrays.append(_normalize_band(band_ds.read(1, window=window), np=np))
    channels_array = np.stack(band_arrays, axis=0)
    return _pad_window(channels_array, window_size, fill_value=0, np=np), _pad_window(
        label, window_size, fill_value=255, np=np
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
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype("float32")


def _encode_label(label, source_class_ids: list[int], *, np):
    encoded = np.full(label.shape, 255, dtype="uint8")
    encoded[label == 255] = 255
    for index, class_id in enumerate(source_class_ids):
        encoded[label == class_id] = index
    encoded[(encoded == 255) & (label == 0)] = 0
    return encoded


def _decode_prediction(encoded_prediction, source_class_ids: list[int], *, np):
    decoded = np.zeros(encoded_prediction.shape, dtype="uint8")
    for index, class_id in enumerate(source_class_ids):
        decoded[encoded_prediction == index] = class_id
    return decoded


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
):
    model.eval()
    labels = []
    predictions = []
    preview_paths = []
    with torch.no_grad():
        for record in records:
            channels, label = _read_validation_window(
                record,
                bundle_dir=bundle_dir,
                channels=tuple(config["input_channels"]),
                window_size=int(config.get("training", {}).get("window_size", 256)),
                rasterio=rasterio,
                RasterWindow=RasterWindow,
                np=np,
            )
            logits = model(torch.from_numpy(channels[None, ...].astype("float32")).to(device))
            encoded_prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")
            prediction = _decode_prediction(encoded_prediction, source_class_ids, np=np)
            labels.append(label)
            predictions.append(prediction)
            if len(preview_paths) < 12:
                preview_paths.append((record.sample_id, channels, label, prediction))
    return _build_metrics_payload(labels, predictions, source_class_ids, preview_paths, np=np)


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
    preview_paths = []
    for record in records:
        mask_path = _resolve_cloud_or_bundle_path(bundle_dir, record.mask_path, subdir="masks")
        with rasterio.open(mask_path) as mask_ds:
            label = mask_ds.read(1)
        prediction = trivial_background_prediction(label)
        labels.append(label)
        predictions.append(prediction)
        if len(preview_paths) < 12:
            preview_paths.append((record.sample_id, np.stack([label, label], axis=0), label, prediction))
    return _build_metrics_payload(labels, predictions, source_class_ids, preview_paths, np=np)


def _build_metrics_payload(labels, predictions, source_class_ids, preview_items, *, np):
    label_array = np.concatenate([label.ravel() for label in labels])
    prediction_array = np.concatenate([prediction.ravel() for prediction in predictions])
    matrix = compute_confusion_matrix(label_array, prediction_array, class_ids=source_class_ids)
    class_metrics = per_class_metrics(matrix, class_ids=source_class_ids)
    mean_iou = float(np.mean([metric.iou for metric in class_metrics[1:]])) if len(class_metrics) > 1 else 0.0
    foreground = [metric for metric in class_metrics if metric.class_id != 0]
    foreground_recall = float(np.mean([metric.recall for metric in foreground])) if foreground else 0.0
    return {
        "confusion_matrix": matrix,
        "per_class": class_metrics,
        "mean_iou": mean_iou,
        "foreground_recall": foreground_recall,
        "area": summarize_area(label_array, prediction_array, class_ids=source_class_ids),
        "preview_items": preview_items,
    }


def _write_run_outputs(run_dir: Path, payload: dict, source_class_ids: list[int]) -> None:
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
    preview_paths = []
    per_sample_dir = run_dir / "predictions" / "per_sample"
    for sample_id, channels, label, prediction in payload["preview_items"]:
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


def _resolve_cloud_or_bundle_path(bundle_dir: Path, value: str, *, subdir: str | None = None) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if subdir is not None:
        mirror = bundle_dir / subdir / path.name
        if mirror.exists():
            return mirror
    relative = bundle_dir / value
    if relative.exists():
        return relative
    return path


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

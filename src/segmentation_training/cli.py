from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
from statistics import mean, median
from pathlib import Path

from .bundle import build_common_channel_bundle, build_training_bundle
from .cloud import refuse_local_training, run_channel_contribution, run_cloud_training
from .config import load_config
from .diagnostic_audit import write_diagnostic_audit
from .manifest import class_split_counts, load_model_input_manifest, validate_record_paths, validate_record_raster_grids
from .models import build_model
from .paired_chips import LANDSLIDE_9CH_CHANNELS, build_paired_chip_contract
from .training_curves import write_training_curves


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="segmentation_training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-manifest")
    validate_parser.add_argument("--contract-dir", required=True)
    validate_parser.add_argument("--required-channel", action="append")
    validate_parser.add_argument("--skip-band-existence", action="store_true")

    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--contract-dir", required=True)
    bundle_parser.add_argument("--bundle-dir", required=True)
    bundle_parser.add_argument("--required-channel", action="append")
    bundle_parser.add_argument("--bundle-version", default="0.1")
    bundle_parser.add_argument("--mode", default="manifest-only")
    bundle_parser.add_argument("--class-scope", default="multiclass_c2_c5")
    bundle_parser.add_argument("--cloud-data-root", default="/data/training_bundle_v0_1")

    common_bundle_parser = subparsers.add_parser("common-channel-bundle")
    common_bundle_parser.add_argument("--c2-bundle-dir", required=True)
    common_bundle_parser.add_argument("--c5-bundle-dir", required=True)
    common_bundle_parser.add_argument("--bundle-dir", required=True)
    common_bundle_parser.add_argument("--bundle-version", default="p15-c2-c5-common4")

    paired_parser = subparsers.add_parser("paired-chip-contract")
    paired_parser.add_argument("--image-dir", required=True)
    paired_parser.add_argument("--mask-dir", required=True)
    paired_parser.add_argument("--output-dir", required=True)
    paired_parser.add_argument("--channel", action="append")
    paired_parser.add_argument("--class-id", type=int, default=1)
    paired_parser.add_argument("--class-name", default="C2_debris_flow")
    paired_parser.add_argument("--train-fraction", type=float, default=0.8)

    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument("--config", required=True)
    dry_parser.add_argument("--bundle-dir")
    dry_parser.add_argument("--contract-dir")
    dry_parser.add_argument("--max-samples", type=int, default=4)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--bundle-dir", default="/data/training_bundle_v0_1")
    train_parser.add_argument("--run-dir", default="/runs/segmentation_training/local_guarded")
    train_parser.add_argument("--cloud-confirm", action="store_true")

    summarize_parser = subparsers.add_parser("summarize-runs")
    summarize_parser.add_argument("--runs-dir", required=True)
    summarize_parser.add_argument("--output", required=True)
    summarize_parser.add_argument("--review-dir")

    plot_parser = subparsers.add_parser("plot-run")
    plot_parser.add_argument("--run-dir", required=True)
    plot_parser.add_argument("--output-png")
    plot_parser.add_argument("--output-csv")

    contribution_parser = subparsers.add_parser("channel-contribution")
    contribution_parser.add_argument("--config", required=True)
    contribution_parser.add_argument("--bundle-dir", required=True)
    contribution_parser.add_argument("--checkpoint", required=True)
    contribution_parser.add_argument("--output-dir", required=True)
    contribution_parser.add_argument("--cloud-confirm", action="store_true")

    diagnostic_parser = subparsers.add_parser("diagnostic-audit")
    diagnostic_parser.add_argument("--run-root", action="append", required=True)
    diagnostic_parser.add_argument("--output-dir", required=True)

    probe_parser = subparsers.add_parser("probe-preflight")
    probe_parser.add_argument("--bundle-dir", required=True)
    probe_parser.add_argument("--output-dir", required=True)
    probe_parser.add_argument("--run-root", required=True)
    probe_parser.add_argument("--config", action="append", required=True)
    probe_parser.add_argument("--max-samples", type=int, default=4)

    model_preflight_parser = subparsers.add_parser("model-preflight")
    model_preflight_parser.add_argument("--output-dir", required=True)
    model_preflight_parser.add_argument("--config", action="append", required=True)
    model_preflight_parser.add_argument("--batch-size", type=int, default=1)
    model_preflight_parser.add_argument("--height", type=int, default=256)
    model_preflight_parser.add_argument("--width", type=int, default=256)

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-manifest":
            return _validate_manifest(args)
        if args.command == "bundle":
            return _bundle(args)
        if args.command == "common-channel-bundle":
            return _common_channel_bundle(args)
        if args.command == "paired-chip-contract":
            return _paired_chip_contract(args)
        if args.command == "dry-run":
            return _dry_run(args)
        if args.command == "train":
            return _train(args, argv)
        if args.command == "summarize-runs":
            return _summarize_runs(args)
        if args.command == "plot-run":
            return _plot_run(args)
        if args.command == "channel-contribution":
            return _channel_contribution(args)
        if args.command == "diagnostic-audit":
            return _diagnostic_audit(args)
        if args.command == "probe-preflight":
            return _probe_preflight(args)
        if args.command == "model-preflight":
            return _model_preflight(args)
    except Exception as exc:
        parser.exit(2, f"segmentation_training: error: {exc}\n")
    parser.error("unsupported command")
    return 2


def _validate_manifest(args) -> int:
    contract_dir = Path(args.contract_dir)
    manifest_path = contract_dir / "model_input_manifest.csv"
    if not manifest_path.exists():
        manifest_path = contract_dir / "manifests" / "cloud_model_input_manifest.csv"
    records = load_model_input_manifest(
        manifest_path,
        required_channels=tuple(args.required_channel or ["F16", "F17"]),
    )
    errors = validate_record_paths(
        records,
        contract_dir=contract_dir,
        require_bands=not args.skip_band_existence,
    )
    if not args.skip_band_existence:
        errors.extend(
            validate_record_raster_grids(
                records,
                contract_dir=contract_dir,
                required_channels=tuple(args.required_channel or ["F16", "F17"]),
            )
        )
    if errors:
        raise ValueError("\n".join(errors[:20]))
    result = {
        "record_count": len(records),
        "split_class_counts": class_split_counts(records),
        "required_channels": args.required_channel or ["F16", "F17"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _paired_chip_contract(args) -> int:
    channels = tuple(args.channel) if args.channel else LANDSLIDE_9CH_CHANNELS
    result = build_paired_chip_contract(
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        output_dir=Path(args.output_dir),
        channels=channels,
        class_id=args.class_id,
        class_name=args.class_name,
        train_fraction=args.train_fraction,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "paired_rows": result.paired_rows,
                "unmatched_images": result.unmatched_images,
                "unmatched_masks": result.unmatched_masks,
                "train_rows": result.train_rows,
                "validation_rows": result.validation_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _bundle(args) -> int:
    manifest = build_training_bundle(
        contract_dir=Path(args.contract_dir),
        bundle_dir=Path(args.bundle_dir),
        required_channels=tuple(args.required_channel or ["F16", "F17"]),
        bundle_version=args.bundle_version,
        mode=args.mode,
        class_scope=args.class_scope,
        cloud_data_root=Path(args.cloud_data_root),
    )
    summary = {key: value for key, value in manifest.items() if key != "files"}
    summary["file_count"] = len(manifest["files"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _common_channel_bundle(args) -> int:
    manifest = build_common_channel_bundle(
        c2_bundle_dir=Path(args.c2_bundle_dir),
        c5_bundle_dir=Path(args.c5_bundle_dir),
        output_bundle_dir=Path(args.bundle_dir),
        bundle_version=args.bundle_version,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _dry_run(args) -> int:
    config = load_config(args.config)
    root = Path(args.bundle_dir or args.contract_dir or ".")
    if args.bundle_dir:
        manifest_path = root / "manifests" / "cloud_model_input_manifest.csv"
        contract_dir = root
    else:
        manifest_path = root / "model_input_manifest.csv"
        contract_dir = root
    records = load_model_input_manifest(manifest_path, required_channels=tuple(config["input_channels"]))
    selection_records = [record for record in records if record.split != "test"]
    checked_records = selection_records[: args.max_samples]
    errors = validate_record_paths(
        checked_records,
        contract_dir=contract_dir,
        require_bands=bool(args.bundle_dir),
    )
    errors.extend(
        validate_record_raster_grids(
            checked_records,
            contract_dir=contract_dir,
            required_channels=tuple(config["input_channels"]),
        )
    )
    if errors:
        raise ValueError("\n".join(errors[:20]))
    result = {
        "experiment_id": config["experiment_id"],
        "class_scope": config["class_scope"],
        "checked_samples": len(checked_records),
        "test_split_read": False,
        "status": "dry_run_passed",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _train(args, argv) -> int:
    if not args.cloud_confirm:
        print(refuse_local_training())
        return 2
    manifest = run_cloud_training(
        config_path=Path(args.config),
        bundle_dir=Path(args.bundle_dir),
        run_dir=Path(args.run_dir),
        command_line=["python", "-m", "segmentation_training", "train", *(argv or [])],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _summarize_runs(args) -> int:
    runs_dir = Path(args.runs_dir)
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_manifest_path = metrics_path.parent / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
        rows.append(_summarize_run(metrics_path.parent, metrics, run_manifest))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Segmentation Training Run Summary", ""]
    lines.append("| Run | Experiment | mean_iou | foreground_recall | Metrics |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['experiment_id']} | {row['mean_iou']} | {row['foreground_recall']} | `{row['path']}` |"
        )
    if not rows:
        lines.append("| no runs found |  |  |  |  |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review_dir = Path(args.review_dir) if args.review_dir else output.parent / "review"
    _write_review_pack(review_dir, rows)
    print(str(output))
    return 0


def _summarize_run(run_dir: Path, metrics: dict, run_manifest: dict) -> dict:
    foreground = _read_foreground_metrics(run_dir / "per_class_metrics.csv")
    area = _read_foreground_area(run_dir / "mask_area_summary.csv")
    best = _read_best_checkpoint(run_dir / "checkpoints" / "best_mean_iou.json")
    label_pixels = _to_float(area.get("label_pixels"))
    predicted_pixels = _to_float(area.get("predicted_pixels"))
    ratio = ""
    if label_pixels and predicted_pixels != "":
        ratio = predicted_pixels / label_pixels
    return {
        "run": run_dir.name,
        "experiment_id": run_manifest.get("experiment_id", ""),
        "model_family": (run_manifest.get("model") or {}).get("family", ""),
        "mean_iou": metrics.get("mean_iou", ""),
        "foreground_recall": metrics.get("foreground_recall", ""),
        "foreground_precision": foreground.get("precision", ""),
        "foreground_dice": foreground.get("dice", ""),
        "foreground_iou": foreground.get("iou", ""),
        "label_foreground_pixels": area.get("label_pixels", ""),
        "predicted_foreground_pixels": area.get("predicted_pixels", ""),
        "pred_label_area_ratio": ratio,
        "best_epoch": best.get("epoch", ""),
        "best_mean_iou": best.get("metric_value", best.get("mean_iou", "")),
        "test_split_read": run_manifest.get("test_split_read", ""),
        "path": str(run_dir / "metrics.json"),
        "run_dir": str(run_dir),
    }


def _write_review_pack(review_dir: Path, rows: list[dict]) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = review_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    ranked_rows = sorted(rows, key=lambda row: _sort_float(row.get("mean_iou")), reverse=True)
    best_rows = sorted(rows, key=lambda row: _sort_float(row.get("best_mean_iou")), reverse=True)
    threshold_rows = _threshold_operating_points(rows)
    validation_window_rows = _validation_window_diagnostics(rows)
    hard_case_rows = _hard_case_windows(rows)
    _write_compact_metrics(review_dir / "compact_metrics.csv", ranked_rows)
    _write_best_checkpoint_metrics(review_dir / "best_checkpoint_metrics.csv", best_rows)
    _write_threshold_operating_points(review_dir / "threshold_operating_points.csv", threshold_rows)
    _write_validation_window_diagnostics(review_dir / "validation_window_diagnostics.csv", validation_window_rows)
    _write_hard_case_windows(review_dir / "hard_case_windows.csv", hard_case_rows)
    _write_review_markdown(
        review_dir / "review.md",
        ranked_rows,
        best_rows,
        threshold_rows,
        validation_window_rows,
        hard_case_rows,
    )
    _write_raw_evidence_index(review_dir / "raw_evidence.md", ranked_rows)
    _copy_review_figures(figures_dir, ranked_rows)


def _write_compact_metrics(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "experiment_id",
        "model_family",
        "best_epoch",
        "best_mean_iou",
        "mean_iou",
        "foreground_dice",
        "foreground_precision",
        "foreground_recall",
        "pred_label_area_ratio",
        "area_ratio_band",
        "label_foreground_pixels",
        "predicted_foreground_pixels",
        "test_split_read",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {key: _format_cell(row.get(key, "")) for key in fieldnames}
            formatted["area_ratio_band"] = _area_ratio_band(row.get("pred_label_area_ratio"))
            writer.writerow(formatted)


def _write_best_checkpoint_metrics(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "experiment_id",
        "model_family",
        "best_epoch",
        "best_mean_iou",
        "final_mean_iou",
        "final_foreground_precision",
        "final_foreground_recall",
        "final_pred_label_area_ratio",
        "area_ratio_band",
        "test_split_read",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run": row.get("run", ""),
                    "experiment_id": row.get("experiment_id", ""),
                    "model_family": row.get("model_family", ""),
                    "best_epoch": _format_cell(row.get("best_epoch", "")),
                    "best_mean_iou": _format_cell(row.get("best_mean_iou", "")),
                    "final_mean_iou": _format_cell(row.get("mean_iou", "")),
                    "final_foreground_precision": _format_cell(row.get("foreground_precision", "")),
                    "final_foreground_recall": _format_cell(row.get("foreground_recall", "")),
                    "final_pred_label_area_ratio": _format_cell(row.get("pred_label_area_ratio", "")),
                    "area_ratio_band": _area_ratio_band(row.get("pred_label_area_ratio")),
                    "test_split_read": _format_cell(row.get("test_split_read", "")),
                    "run_dir": row.get("run_dir", ""),
                }
            )


def _write_threshold_operating_points(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "experiment_id",
        "model_family",
        "best_threshold",
        "foreground_iou",
        "foreground_dice",
        "foreground_precision",
        "foreground_recall",
        "pred_label_area_ratio",
        "area_ratio_band",
        "selection_scope",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key, "")) for key in fieldnames})


def _write_validation_window_diagnostics(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "experiment_id",
        "model_family",
        "window_count",
        "mean_iou",
        "median_iou",
        "mean_pred_label_area_ratio",
        "median_pred_label_area_ratio",
        "zero_iou_windows",
        "overexpanded_windows",
        "underpredicted_windows",
        "acceptable_area_windows",
        "worst_sample_id",
        "worst_foreground_iou",
        "selection_scope",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key, "")) for key in fieldnames})


def _write_hard_case_windows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "experiment_id",
        "model_family",
        "sample_id",
        "foreground_iou",
        "foreground_precision",
        "foreground_recall",
        "pred_label_area_ratio",
        "area_ratio_band",
        "label_foreground_pixels",
        "predicted_foreground_pixels",
        "all_background_prediction",
        "issue_flags",
        "selection_scope",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key, "")) for key in fieldnames})


def _write_review_markdown(
    path: Path,
    rows: list[dict],
    best_rows: list[dict],
    threshold_rows: list[dict],
    validation_window_rows: list[dict],
    hard_case_rows: list[dict],
) -> None:
    lines = [
        "# Review-First Training Summary",
        "",
        "> [!warning]",
        "> This summary is validation-only. The sealed test split must remain sealed.",
        "",
        "## Ranked By Final Mean IoU",
        "",
        "| Rank | Run | Family | Best epoch | Best mean IoU | Final mean IoU | Precision | Recall | Dice | Pred/label area |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("|  | no runs found |  |  |  |  |  |  |  |  |")
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['run']}`",
                    str(row.get("model_family", "")),
                    _format_cell(row.get("best_epoch", "")),
                    _format_float(row.get("best_mean_iou", "")),
                    _format_float(row.get("mean_iou", "")),
                    _format_float(row.get("foreground_precision", "")),
                    _format_float(row.get("foreground_recall", "")),
                    _format_float(row.get("foreground_dice", "")),
                    _format_float(row.get("pred_label_area_ratio", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Ranked By Best Checkpoint",
            "",
            "| Rank | Run | Family | Best epoch | Best mean IoU | Final mean IoU | Final area ratio | Area ratio band |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not best_rows:
        lines.append("|  | no runs found |  |  |  |  |  |  |")
    for index, row in enumerate(best_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['run']}`",
                    str(row.get("model_family", "")),
                    _format_cell(row.get("best_epoch", "")),
                    _format_float(row.get("best_mean_iou", "")),
                    _format_float(row.get("mean_iou", "")),
                    _format_float(row.get("pred_label_area_ratio", "")),
                    _area_ratio_band(row.get("pred_label_area_ratio")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Threshold-Calibrated Operating Points",
            "",
            "| Rank | Run | Family | Threshold | IoU | Precision | Recall | Pred/label area | Area ratio band |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not threshold_rows:
        lines.append("|  | no threshold sweep rows found |  |  |  |  |  |  |  |")
    for index, row in enumerate(threshold_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['run']}`",
                    str(row.get("model_family", "")),
                    _format_float(row.get("best_threshold", "")),
                    _format_float(row.get("foreground_iou", "")),
                    _format_float(row.get("foreground_precision", "")),
                    _format_float(row.get("foreground_recall", "")),
                    _format_float(row.get("pred_label_area_ratio", "")),
                    str(row.get("area_ratio_band", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Validation Window Diagnostics",
            "",
            "| Rank | Run | Windows | Median IoU | Mean IoU | Median area | Zero IoU | Overexpanded | Underpredicted | Worst sample |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not validation_window_rows:
        lines.append("|  | no validation window rows found |  |  |  |  |  |  |  |  |")
    for index, row in enumerate(validation_window_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['run']}`",
                    _format_cell(row.get("window_count", "")),
                    _format_float(row.get("median_iou", "")),
                    _format_float(row.get("mean_iou", "")),
                    _format_float(row.get("median_pred_label_area_ratio", "")),
                    _format_cell(row.get("zero_iou_windows", "")),
                    _format_cell(row.get("overexpanded_windows", "")),
                    _format_cell(row.get("underpredicted_windows", "")),
                    str(row.get("worst_sample_id", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Worst Validation Windows",
            "",
            "| Rank | Run | Sample | IoU | Precision | Recall | Area ratio | Issues |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not hard_case_rows:
        lines.append("|  | no hard-case rows found |  |  |  |  |  |  |")
    for index, row in enumerate(hard_case_rows[:20], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['run']}`",
                    str(row.get("sample_id", "")),
                    _format_float(row.get("foreground_iou", "")),
                    _format_float(row.get("foreground_precision", "")),
                    _format_float(row.get("foreground_recall", "")),
                    _format_float(row.get("pred_label_area_ratio", "")),
                    str(row.get("issue_flags", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Review Layout",
            "",
            "- `compact_metrics.csv` contains the comparison table for spreadsheet review.",
            "- `best_checkpoint_metrics.csv` ranks validation best checkpoints separately from final epoch metrics.",
            "- `threshold_operating_points.csv` chooses validation-only threshold operating points with area-ratio gating.",
            "- `validation_window_diagnostics.csv` aggregates validation-window failure modes per run.",
            "- `hard_case_windows.csv` lists the worst validation windows for geometry and hard-negative review.",
            "- `figures/` contains selected validation contact sheets and training curves.",
            "- `raw_evidence.md` links the raw run directories and metrics files without exposing JSON as the first review surface.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_raw_evidence_index(path: Path, rows: list[dict]) -> None:
    lines = ["# Raw Evidence Index", ""]
    if not rows:
        lines.append("- no runs found")
    for row in rows:
        lines.append(f"- `{row['run']}`: `{row['run_dir']}`")
        lines.append(f"  - metrics: `{row['path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_review_figures(figures_dir: Path, rows: list[dict]) -> None:
    candidates = [
        ("training_curves.png", "training_curves.png"),
        ("predictions/validation_contact_sheet.png", "validation_contact_sheet.png"),
        ("predictions/failures_contact_sheet.png", "failures_contact_sheet.png"),
    ]
    for row in rows:
        run_dir = Path(row["run_dir"])
        for source_name, output_name in candidates:
            source = run_dir / source_name
            if source.exists():
                shutil.copy2(source, figures_dir / f"{row['run']}__{output_name}")


def _threshold_operating_points(rows: list[dict]) -> list[dict]:
    selected = []
    for row in rows:
        threshold_rows = _aggregate_threshold_rows(Path(row["run_dir"]) / "diagnostics" / "threshold_sweep.csv")
        if not threshold_rows:
            continue
        best = max(
            threshold_rows,
            key=lambda item: (_area_band_rank(item["area_ratio_band"]), _sort_float(item.get("foreground_iou"))),
        )
        selected.append(
            {
                "run": row.get("run", ""),
                "experiment_id": row.get("experiment_id", ""),
                "model_family": row.get("model_family", ""),
                "best_threshold": best.get("threshold", ""),
                "foreground_iou": best.get("foreground_iou", ""),
                "foreground_dice": best.get("foreground_dice", ""),
                "foreground_precision": best.get("foreground_precision", ""),
                "foreground_recall": best.get("foreground_recall", ""),
                "pred_label_area_ratio": best.get("pred_label_area_ratio", ""),
                "area_ratio_band": best.get("area_ratio_band", ""),
                "selection_scope": "validation_only",
                "run_dir": row.get("run_dir", ""),
            }
        )
    return sorted(
        selected,
        key=lambda item: (_area_band_rank(item.get("area_ratio_band")), _sort_float(item.get("foreground_iou"))),
        reverse=True,
    )


def _validation_window_diagnostics(rows: list[dict]) -> list[dict]:
    diagnostics = []
    for row in rows:
        window_rows = _read_csv_dicts(Path(row["run_dir"]) / "diagnostics" / "validation_prediction_summary.csv")
        parsed_rows = [_parse_window_row(window_row) for window_row in window_rows]
        parsed_rows = [window_row for window_row in parsed_rows if isinstance(window_row["foreground_iou"], float)]
        if not parsed_rows:
            continue
        ious = [window_row["foreground_iou"] for window_row in parsed_rows]
        ratios = [window_row["pred_label_area_ratio"] for window_row in parsed_rows if isinstance(window_row["pred_label_area_ratio"], float)]
        worst = min(parsed_rows, key=lambda item: (_sort_float(item["foreground_iou"]), -_issue_score(item)))
        diagnostics.append(
            {
                "run": row.get("run", ""),
                "experiment_id": row.get("experiment_id", ""),
                "model_family": row.get("model_family", ""),
                "window_count": len(parsed_rows),
                "mean_iou": mean(ious),
                "median_iou": median(ious),
                "mean_pred_label_area_ratio": mean(ratios) if ratios else "",
                "median_pred_label_area_ratio": median(ratios) if ratios else "",
                "zero_iou_windows": sum(1 for window_row in parsed_rows if window_row["foreground_iou"] <= 0.0),
                "overexpanded_windows": sum(
                    1
                    for window_row in parsed_rows
                    if isinstance(window_row["pred_label_area_ratio"], float)
                    and window_row["pred_label_area_ratio"] > 4.0
                ),
                "underpredicted_windows": sum(
                    1
                    for window_row in parsed_rows
                    if isinstance(window_row["pred_label_area_ratio"], float)
                    and window_row["pred_label_area_ratio"] < 0.5
                ),
                "acceptable_area_windows": sum(
                    1
                    for window_row in parsed_rows
                    if isinstance(window_row["pred_label_area_ratio"], float)
                    and 0.5 <= window_row["pred_label_area_ratio"] <= 2.0
                ),
                "worst_sample_id": worst.get("sample_id", ""),
                "worst_foreground_iou": worst.get("foreground_iou", ""),
                "selection_scope": "validation_only",
                "run_dir": row.get("run_dir", ""),
            }
        )
    return sorted(
        diagnostics,
        key=lambda item: (_sort_float(item.get("median_iou")), _sort_float(item.get("mean_iou"))),
        reverse=True,
    )


def _hard_case_windows(rows: list[dict], limit: int = 100) -> list[dict]:
    hard_cases = []
    for row in rows:
        window_rows = _read_csv_dicts(Path(row["run_dir"]) / "diagnostics" / "validation_prediction_summary.csv")
        for window_row in window_rows:
            parsed = _parse_window_row(window_row)
            if not isinstance(parsed["foreground_iou"], float):
                continue
            flags = _issue_flags(parsed)
            hard_cases.append(
                {
                    "run": row.get("run", ""),
                    "experiment_id": row.get("experiment_id", ""),
                    "model_family": row.get("model_family", ""),
                    "sample_id": parsed.get("sample_id", ""),
                    "foreground_iou": parsed.get("foreground_iou", ""),
                    "foreground_precision": parsed.get("foreground_precision", ""),
                    "foreground_recall": parsed.get("foreground_recall", ""),
                    "pred_label_area_ratio": parsed.get("pred_label_area_ratio", ""),
                    "area_ratio_band": _area_ratio_band(parsed.get("pred_label_area_ratio", "")),
                    "label_foreground_pixels": parsed.get("label_foreground_pixels", ""),
                    "predicted_foreground_pixels": parsed.get("predicted_foreground_pixels", ""),
                    "all_background_prediction": str(parsed.get("all_background_prediction", False)).lower(),
                    "issue_flags": ";".join(flags),
                    "selection_scope": "validation_only",
                    "run_dir": row.get("run_dir", ""),
                }
            )
    return sorted(
        hard_cases,
        key=lambda item: (
            _sort_float(item.get("foreground_iou")),
            -len(str(item.get("issue_flags", "")).split(";")) if item.get("issue_flags") else 0,
            -_area_ratio_severity(item.get("pred_label_area_ratio")),
        ),
    )[:limit]


def _parse_window_row(row: dict) -> dict:
    return {
        "sample_id": row.get("sample_id", ""),
        "foreground_iou": _first_float(row, "foreground_iou", "iou", "mean_iou"),
        "foreground_precision": _first_float(row, "foreground_precision", "precision"),
        "foreground_recall": _first_float(row, "foreground_recall", "recall"),
        "pred_label_area_ratio": _first_float(row, "pred_label_area_ratio", "predicted_label_area_ratio"),
        "label_foreground_pixels": _first_float(row, "label_foreground_pixels", "foreground_pixels"),
        "predicted_foreground_pixels": _first_float(row, "predicted_foreground_pixels"),
        "all_background_prediction": _to_bool(row.get("all_background_prediction")),
    }


def _first_float(row: dict, *keys: str) -> float | str:
    for key in keys:
        parsed = _to_float(row.get(key))
        if isinstance(parsed, float):
            return parsed
    return ""


def _to_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _issue_flags(row: dict) -> list[str]:
    flags = []
    iou = row.get("foreground_iou")
    precision = row.get("foreground_precision")
    ratio = row.get("pred_label_area_ratio")
    if isinstance(iou, float) and iou <= 0.0:
        flags.append("zero_iou")
    if isinstance(ratio, float) and ratio > 4.0:
        flags.append("overexpanded")
    if isinstance(ratio, float) and ratio < 0.5:
        flags.append("underpredicted")
    if isinstance(precision, float) and isinstance(ratio, float) and precision < 0.1 and ratio > 2.0:
        flags.append("low_precision_spill")
    if row.get("all_background_prediction"):
        flags.append("all_background")
    return flags or ["low_iou_review"]


def _issue_score(row: dict) -> int:
    return len(_issue_flags(row))


def _area_ratio_severity(value) -> float:
    ratio = _to_float(value)
    if not isinstance(ratio, float):
        return 0.0
    if ratio < 0.5:
        return 0.5 - ratio
    if ratio > 4.0:
        return ratio - 4.0
    return 0.0


def _aggregate_threshold_rows(path: Path) -> list[dict]:
    rows = _read_csv_dicts(path)
    grouped: dict[str, dict] = {}
    for row in rows:
        threshold = row.get("threshold", "")
        if threshold == "":
            continue
        group = grouped.setdefault(
            threshold,
            {
                "threshold": threshold,
                "count": 0,
                "foreground_iou": 0.0,
                "foreground_dice": 0.0,
                "foreground_precision": 0.0,
                "foreground_recall": 0.0,
                "pred_label_area_ratio": 0.0,
            },
        )
        group["count"] += 1
        group["foreground_iou"] += _threshold_float(row, "foreground_iou", "iou", "mean_iou")
        group["foreground_dice"] += _threshold_float(row, "foreground_dice", "dice")
        group["foreground_precision"] += _threshold_float(row, "foreground_precision", "precision")
        group["foreground_recall"] += _threshold_float(row, "foreground_recall", "recall")
        group["pred_label_area_ratio"] += _threshold_float(row, "pred_label_area_ratio", "predicted_label_area_ratio")
    aggregated = []
    for group in grouped.values():
        count = group.pop("count")
        averaged = {
            "threshold": group["threshold"],
            "foreground_iou": group["foreground_iou"] / count,
            "foreground_dice": group["foreground_dice"] / count,
            "foreground_precision": group["foreground_precision"] / count,
            "foreground_recall": group["foreground_recall"] / count,
            "pred_label_area_ratio": group["pred_label_area_ratio"] / count,
        }
        averaged["area_ratio_band"] = _area_ratio_band(averaged["pred_label_area_ratio"])
        aggregated.append(averaged)
    return aggregated


def _threshold_float(row: dict, *keys: str) -> float:
    for key in keys:
        parsed = _to_float(row.get(key))
        if isinstance(parsed, float):
            return parsed
    return 0.0


def _area_ratio_band(value) -> str:
    ratio = _to_float(value)
    if not isinstance(ratio, float):
        return "missing"
    if 0.5 <= ratio <= 2.0:
        return "acceptable"
    if 2.0 < ratio <= 4.0:
        return "caution"
    if ratio > 4.0:
        return "blocked_overexpansion"
    return "blocked_underprediction"


def _area_band_rank(band: str) -> int:
    return {
        "acceptable": 3,
        "caution": 2,
        "blocked_underprediction": 1,
        "blocked_overexpansion": 0,
        "missing": -1,
    }.get(str(band), -1)


def _read_foreground_metrics(path: Path) -> dict:
    rows = _read_csv_dicts(path)
    for row in rows:
        if row.get("class_id") not in {"", "0", "background", "255"}:
            return row
    return {}


def _read_foreground_area(path: Path) -> dict:
    rows = _read_csv_dicts(path)
    for row in rows:
        if row.get("class_id") not in {"", "0", "background", "255"}:
            return row
    return {}


def _read_best_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(value) -> float | str:
    if value in {"", None}:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _sort_float(value) -> float:
    parsed = _to_float(value)
    return parsed if isinstance(parsed, float) else float("-inf")


def _format_float(value) -> str:
    parsed = _to_float(value)
    if not isinstance(parsed, float):
        return ""
    return f"{parsed:.6g}"


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _plot_run(args) -> int:
    result = write_training_curves(
        Path(args.run_dir),
        output_png=Path(args.output_png) if args.output_png else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _channel_contribution(args) -> int:
    if not args.cloud_confirm:
        print(refuse_local_training())
        return 2
    result = run_channel_contribution(
        config_path=Path(args.config),
        bundle_dir=Path(args.bundle_dir),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _diagnostic_audit(args) -> int:
    result = write_diagnostic_audit(args.run_root, Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _probe_preflight(args) -> int:
    bundle_dir = Path(args.bundle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    commands = []
    for config_path_text in args.config:
        config_path = Path(config_path_text)
        config = load_config(config_path)
        _ensure_probe_smoke_config(config, config_path)
        manifest_path = bundle_dir / "manifests" / "cloud_model_input_manifest.csv"
        records = load_model_input_manifest(manifest_path, required_channels=tuple(config["input_channels"]))
        selection_records = [record for record in records if record.split != "test"]
        checked_records = selection_records[: args.max_samples]
        errors = validate_record_paths(checked_records, contract_dir=bundle_dir, require_bands=True)
        errors.extend(
            validate_record_raster_grids(
                checked_records,
                contract_dir=bundle_dir,
                required_channels=tuple(config["input_channels"]),
            )
        )
        if errors:
            raise ValueError(f"{config_path}: " + "\n".join(errors[:20]))
        run_dir = Path(args.run_root) / config["experiment_id"]
        command = [
            "python",
            "-m",
            "segmentation_training",
            "train",
            "--config",
            str(config_path),
            "--bundle-dir",
            str(bundle_dir),
            "--run-dir",
            str(run_dir),
            "--cloud-confirm",
        ]
        command_text = " ".join(shlex.quote(item) for item in command)
        commands.append(command_text)
        rows.append(
            {
                "config": str(config_path),
                "experiment_id": config["experiment_id"],
                "model_family": config["model"]["family"],
                "input_channels": ";".join(config["input_channels"]),
                "max_epochs": config["training"]["max_epochs"],
                "max_train_samples": config["cloud"].get("max_train_samples", ""),
                "max_validation_samples": config["cloud"].get("max_validation_samples", ""),
                "checked_samples": len(checked_records),
                "test_split_read": "false",
                "run_dir": str(run_dir),
                "command": command_text,
                "status": "preflight_passed",
            }
        )
    _write_probe_preflight_matrix(output_dir / "probe_preflight_matrix.csv", rows)
    _write_probe_preflight_report(output_dir / "probe_preflight_report.md", rows)
    _write_probe_commands(output_dir / "cloud_smoke_commands.sh", commands)
    result = {
        "config_count": len(rows),
        "output_dir": str(output_dir),
        "status": "probe_preflight_passed",
        "test_split_read": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _ensure_probe_smoke_config(config: dict, config_path: Path) -> None:
    if int(config.get("training", {}).get("max_epochs", 0)) != 1:
        raise ValueError(f"{config_path}: probe preflight accepts smoke configs with max_epochs=1 only")
    if int(config.get("cloud", {}).get("max_train_samples", 0)) > 16:
        raise ValueError(f"{config_path}: probe preflight max_train_samples must be <= 16")
    if int(config.get("cloud", {}).get("max_validation_samples", 0)) > 8:
        raise ValueError(f"{config_path}: probe preflight max_validation_samples must be <= 8")


def _write_probe_preflight_matrix(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "config",
        "experiment_id",
        "model_family",
        "input_channels",
        "max_epochs",
        "max_train_samples",
        "max_validation_samples",
        "checked_samples",
        "test_split_read",
        "run_dir",
        "status",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_probe_preflight_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Probe Preflight Report",
        "",
        "This report verifies controlled smoke experiment readiness only. It does not launch training.",
        "",
        "## Scope Guard",
        "",
        "- `test_split_read=false` for every preflight row.",
        "- Full training remains blocked until QA accepts diagnostic artifacts and the controlled probe protocol.",
        "- Commands below are smoke commands and must be run only on the cloud machine with the intended bundle.",
        "",
        "## Preflight Matrix",
        "",
        "| Experiment | Family | Checked samples | Command |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['experiment_id']}` | {row['model_family']} | {row['checked_samples']} | `{row['command']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_probe_commands(path: Path, commands: list[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export PYTHONPATH=${PYTHONPATH:-src}",
        "",
        "# Controlled smoke commands only. Full training remains blocked until QA approval.",
        *commands,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _model_preflight(args) -> int:
    if args.batch_size < 1 or args.height < 1 or args.width < 1:
        raise ValueError("model-preflight dimensions and batch size must be positive")
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_path_text in args.config:
        config_path = Path(config_path_text)
        config = load_config(config_path)
        input_channels = len(config["input_channels"])
        output_classes = len(config["metrics"]["class_ids"])
        model = build_model(config["model"], input_channels=input_channels, output_classes=output_classes)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(args.batch_size, input_channels, args.height, args.width))
        actual_shape = tuple(int(value) for value in logits.shape)
        expected_shape = (args.batch_size, output_classes, args.height, args.width)
        if actual_shape != expected_shape:
            raise ValueError(f"{config_path}: expected output shape {expected_shape}, got {actual_shape}")
        rows.append(
            {
                "config": str(config_path),
                "experiment_id": config["experiment_id"],
                "model_family": config["model"]["family"],
                "input_channel_count": input_channels,
                "output_class_count": output_classes,
                "synthetic_input_shape": _shape_text((args.batch_size, input_channels, args.height, args.width)),
                "output_shape": _shape_text(actual_shape),
                "test_split_read": "false",
                "status": "model_preflight_passed",
            }
        )
    _write_model_preflight_matrix(output_dir / "model_preflight_matrix.csv", rows)
    _write_model_preflight_report(output_dir / "model_preflight_report.md", rows)
    result = {
        "config_count": len(rows),
        "output_dir": str(output_dir),
        "status": "model_preflight_passed",
        "test_split_read": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in shape)


def _write_model_preflight_matrix(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "config",
        "experiment_id",
        "model_family",
        "input_channel_count",
        "output_class_count",
        "synthetic_input_shape",
        "output_shape",
        "test_split_read",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_model_preflight_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Model Preflight Report",
        "",
        "This report verifies model construction and one synthetic forward pass only; it does not read data or launch training.",
        "",
        "## Scope Guard",
        "",
        "- `test_split_read=false` for every model preflight row.",
        "- The sealed test split remains sealed.",
        "- Full training remains blocked until QA accepts diagnostic artifacts and the controlled probe protocol.",
        "",
        "## Model Matrix",
        "",
        "| Experiment | Family | Synthetic input | Output | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['experiment_id']}` | {row['model_family']} | `{row['synthetic_input_shape']}` | "
            f"`{row['output_shape']}` | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

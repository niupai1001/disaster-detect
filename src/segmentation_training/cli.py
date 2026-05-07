from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from .bundle import build_training_bundle
from .cloud import refuse_local_training, run_channel_contribution, run_cloud_training
from .config import load_config
from .manifest import class_split_counts, load_model_input_manifest, validate_record_paths, validate_record_raster_grids
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

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-manifest":
            return _validate_manifest(args)
        if args.command == "bundle":
            return _bundle(args)
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
    _write_compact_metrics(review_dir / "compact_metrics.csv", ranked_rows)
    _write_review_markdown(review_dir / "review.md", ranked_rows)
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
        "label_foreground_pixels",
        "predicted_foreground_pixels",
        "test_split_read",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key, "")) for key in fieldnames})


def _write_review_markdown(path: Path, rows: list[dict]) -> None:
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
            "## Review Layout",
            "",
            "- `compact_metrics.csv` contains the comparison table for spreadsheet review.",
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

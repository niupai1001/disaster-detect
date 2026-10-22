from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import build_training_bundle
from .cloud import refuse_local_training, run_cloud_training
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

    plot_parser = subparsers.add_parser("plot-run")
    plot_parser.add_argument("--run-dir", required=True)
    plot_parser.add_argument("--output-png")
    plot_parser.add_argument("--output-csv")

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
        rows.append(
            {
                "run": metrics_path.parent.name,
                "experiment_id": run_manifest.get("experiment_id", ""),
                "mean_iou": metrics.get("mean_iou", ""),
                "foreground_recall": metrics.get("foreground_recall", ""),
                "path": str(metrics_path),
            }
        )
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
    print(str(output))
    return 0


def _plot_run(args) -> int:
    result = write_training_curves(
        Path(args.run_dir),
        output_png=Path(args.output_png) if args.output_png else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

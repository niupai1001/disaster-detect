from __future__ import annotations

import argparse
from pathlib import Path

from .phase_a import build_phase_a_contract
from .phase_b import build_channel_alignment_audit, build_phase_b_masks, build_phase_b_preflight, build_visual_qa


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SegmentationDatasetContract v0.1 Phase A artifacts.")
    parser.add_argument("--project-root", default=".", help="Project root containing database/csv文本.")
    parser.add_argument(
        "--output-dir",
        default=".agent-team/artifacts/task-ef3a9c94ddb7/implementation/segmentation_contract_v0_1",
        help="Directory where Phase A contract artifacts will be written.",
    )
    parser.add_argument(
        "--label-csv",
        action="append",
        dest="label_csvs",
        help="Label CSV path. May be repeated. Defaults to database/csv文本/*.csv.",
    )
    parser.add_argument(
        "--channel",
        action="append",
        dest="channels",
        help="Declared channel name in order. Defaults to F01..F17.",
    )
    parser.add_argument(
        "--phase-b-preflight",
        action="store_true",
        help="Create Phase B preflight directories and reports after Phase A.",
    )
    parser.add_argument(
        "--phase-b-build",
        action="store_true",
        help="Build Phase B geospatial masks and reports after Phase A.",
    )
    parser.add_argument(
        "--phase-b-limit",
        type=int,
        help="Optional maximum number of manifest rows to process during Phase B build.",
    )
    parser.add_argument(
        "--visual-qa",
        action="store_true",
        help="Generate overlay preview PNGs, contact sheet, and training_manifest.csv.",
    )
    parser.add_argument(
        "--visual-qa-max-items",
        type=int,
        default=36,
        help="Maximum number of overlay tiles in the contact sheet.",
    )
    parser.add_argument(
        "--channel-audit",
        action="store_true",
        help="Generate channel_alignment_report.csv and model_input_manifest.csv.",
    )
    parser.add_argument(
        "--required-channel",
        action="append",
        dest="required_channels",
        help="Required model input channel. May be repeated. Defaults to F16,F17.",
    )
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    if args.label_csvs:
        label_csvs = [Path(path) if Path(path).is_absolute() else root / path for path in args.label_csvs]
    else:
        label_csvs = sorted((root / "database" / "csv文本").glob("*.csv"))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    raster_paths = sorted((root / "database").glob("**/*.tif"))
    result = build_phase_a_contract(label_csvs, output_dir, channel_names=args.channels, raster_paths=raster_paths)
    if args.phase_b_build:
        build_phase_b_masks(result.output_dir, limit=args.phase_b_limit)
    elif args.phase_b_preflight:
        build_phase_b_preflight(result.output_dir)
    if args.visual_qa:
        build_visual_qa(result.output_dir, max_items=args.visual_qa_max_items)
    if args.channel_audit:
        build_channel_alignment_audit(result.output_dir, required_channels=args.required_channels or ("F16", "F17"))
    print(result.output_dir)
    return 0

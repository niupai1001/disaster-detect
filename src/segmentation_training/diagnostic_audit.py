from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRoot:
    label: str
    path: Path


AREA_BANDS = (
    ("acceptable", 0.5, 2.0),
    ("caution", 2.0, 4.0),
)


def write_diagnostic_audit(run_roots: list[str], output_dir: Path) -> dict:
    roots = [_parse_run_root(item) for item in run_roots]
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _collect_runs(roots)
    threshold_rows = _collect_threshold_rows(runs)
    hard_negative_rows = _collect_hard_negative_rows(runs)
    probability_rows = _collect_probability_rows(runs)
    coverage_rows = _collect_coverage_rows(runs)

    _write_csv(output_dir / "validation_operating_point_audit.csv", [_audit_row(run) for run in runs])
    _write_csv(output_dir / "candidate_ranking_by_protocol.csv", [_ranking_row(run) for run in _rank_runs(runs)])
    _write_csv(output_dir / "threshold_aggregate_by_run.csv", threshold_rows)
    _write_csv(output_dir / "hard_negative_inventory.csv", hard_negative_rows)
    _write_markdown(output_dir / "validation_operating_point_audit.md", _operating_point_markdown(runs, threshold_rows))
    _write_markdown(output_dir / "sealed_split_guard_report.md", _sealed_split_guard_markdown(runs))
    _write_markdown(output_dir / "hard_negative_inventory.md", _hard_negative_markdown(hard_negative_rows))
    _write_markdown(output_dir / "hard_negative_taxonomy.yaml", _hard_negative_taxonomy_yaml())
    _write_markdown(output_dir / "channel_normalization_audit.md", _channel_audit_markdown(probability_rows, coverage_rows))
    _write_markdown(output_dir / "controlled_probe_protocol.md", _probe_protocol_markdown())
    _write_csv(output_dir / "controlled_probe_matrix.csv", _probe_matrix_rows())
    _write_markdown(output_dir / "probe_acceptance_criteria.md", _probe_acceptance_markdown())
    _write_markdown(output_dir / "implementation_scope_guard.md", _implementation_scope_guard_markdown())
    return {
        "output_dir": str(output_dir),
        "run_count": len(runs),
        "test_split_read_count": sum(1 for run in runs if _to_bool(run.get("test_split_read"))),
        "status": "diagnostic_audit_written",
    }


def _parse_run_root(value: str) -> RunRoot:
    if "=" in value:
        label, path = value.split("=", 1)
        return RunRoot(label=label.strip() or Path(path).name, path=Path(path))
    path = Path(value)
    return RunRoot(label=path.name, path=path)


def _collect_runs(roots: list[RunRoot]) -> list[dict]:
    runs: list[dict] = []
    for root in roots:
        compact_path = root.path / "review" / "compact_metrics.csv"
        compact_rows = _read_csv(compact_path)
        if compact_rows:
            for row in compact_rows:
                run_dir = _resolve_run_dir(root, row)
                runs.append(_normalize_run_row(root, row, run_dir))
            continue
        for metrics_path in sorted(root.path.glob("*/metrics.json")):
            run_dir = metrics_path.parent
            runs.append(_run_from_raw_files(root, run_dir))
    return runs


def _resolve_run_dir(root: RunRoot, row: dict) -> Path:
    configured = Path(row.get("run_dir") or "")
    if configured.exists():
        return configured
    local = root.path / row.get("run", "")
    if local.exists():
        return local
    return configured if str(configured) else local


def _normalize_run_row(root: RunRoot, row: dict, run_dir: Path) -> dict:
    manifest = _read_json(run_dir / "run_manifest.json")
    normalized = dict(row)
    normalized["source"] = root.label
    normalized["run"] = row.get("run") or run_dir.name
    normalized["run_dir"] = str(run_dir)
    normalized["experiment_id"] = row.get("experiment_id") or manifest.get("experiment_id", "")
    normalized["model_family"] = row.get("model_family") or (manifest.get("model") or {}).get("family", "")
    normalized["test_split_read"] = row.get("test_split_read")
    if normalized["test_split_read"] in {"", None}:
        normalized["test_split_read"] = manifest.get("test_split_read", "")
    return normalized


def _run_from_raw_files(root: RunRoot, run_dir: Path) -> dict:
    metrics = _read_json(run_dir / "metrics.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    foreground = _read_foreground_row(run_dir / "per_class_metrics.csv")
    area = _read_foreground_row(run_dir / "mask_area_summary.csv")
    label_pixels = _to_float(area.get("label_pixels"))
    predicted_pixels = _to_float(area.get("predicted_pixels"))
    area_ratio = ""
    if label_pixels and isinstance(predicted_pixels, float):
        area_ratio = predicted_pixels / label_pixels
    return {
        "source": root.label,
        "run": run_dir.name,
        "experiment_id": manifest.get("experiment_id", ""),
        "model_family": (manifest.get("model") or {}).get("family", ""),
        "mean_iou": metrics.get("mean_iou", ""),
        "foreground_precision": foreground.get("precision", ""),
        "foreground_recall": metrics.get("foreground_recall", foreground.get("recall", "")),
        "foreground_dice": foreground.get("dice", ""),
        "foreground_iou": foreground.get("iou", metrics.get("mean_iou", "")),
        "pred_label_area_ratio": area_ratio,
        "test_split_read": manifest.get("test_split_read", ""),
        "run_dir": str(run_dir),
    }


def _collect_threshold_rows(runs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        source_rows = _read_csv(Path(run["run_dir"]) / "diagnostics" / "threshold_sweep.csv")
        for row in source_rows:
            rows.append(
                {
                    "source": run["source"],
                    "run": run["run"],
                    "experiment_id": run.get("experiment_id", ""),
                    "threshold": row.get("threshold", ""),
                    "foreground_iou": row.get("foreground_iou", row.get("iou", row.get("mean_iou", ""))),
                    "foreground_precision": row.get("foreground_precision", row.get("precision", "")),
                    "foreground_recall": row.get("foreground_recall", row.get("recall", "")),
                    "pred_label_area_ratio": row.get("pred_label_area_ratio", run.get("pred_label_area_ratio", "")),
                    "selection_scope": "validation_only",
                }
            )
    return rows


def _collect_hard_negative_rows(runs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        fp_rows = _read_csv(Path(run["run_dir"]) / "diagnostics" / "worst_false_positives.csv")
        for index, row in enumerate(fp_rows, start=1):
            rows.append(
                {
                    "source": run["source"],
                    "run": run["run"],
                    "rank": index,
                    "sample_id": row.get("sample_id", row.get("record_id", "")),
                    "false_positive_pixels": row.get("false_positive_pixels", row.get("predicted_foreground_pixels", "")),
                    "foreground_pixels": row.get("foreground_pixels", row.get("label_foreground_pixels", "")),
                    "negative_category": "needs_manual_review",
                    "split": "validation",
                    "review_status": "taxonomy_seed_only",
                    "sampler_candidate": "no_validation_to_training_leakage",
                }
            )
    return rows


def _collect_probability_rows(runs: list[dict]) -> list[dict]:
    return _collect_diagnostic_csv(runs, "foreground_probability_summary.csv")


def _collect_coverage_rows(runs: list[dict]) -> list[dict]:
    return _collect_diagnostic_csv(runs, "foreground_window_coverage.csv")


def _collect_diagnostic_csv(runs: list[dict], filename: str) -> list[dict]:
    rows: list[dict] = []
    for run in runs:
        for row in _read_csv(Path(run["run_dir"]) / "diagnostics" / filename):
            with_run = {"source": run["source"], "run": run["run"], **row}
            rows.append(with_run)
    return rows


def _audit_row(run: dict) -> dict:
    return {
        "source": run.get("source", ""),
        "run": run.get("run", ""),
        "experiment_id": run.get("experiment_id", ""),
        "model_family": run.get("model_family", ""),
        "best_epoch": run.get("best_epoch", ""),
        "best_mean_iou": run.get("best_mean_iou", ""),
        "final_mean_iou": run.get("mean_iou", ""),
        "foreground_precision": run.get("foreground_precision", ""),
        "foreground_recall": run.get("foreground_recall", ""),
        "foreground_dice": run.get("foreground_dice", ""),
        "pred_label_area_ratio": run.get("pred_label_area_ratio", ""),
        "area_ratio_band": _area_ratio_band(run.get("pred_label_area_ratio")),
        "test_split_read": run.get("test_split_read", ""),
        "run_dir": run.get("run_dir", ""),
    }


def _ranking_row(run: dict) -> dict:
    row = _audit_row(run)
    row["ranking_protocol"] = "final_mean_iou_with_area_ratio_gate"
    row["primary_candidate_allowed"] = "yes" if row["area_ratio_band"] == "acceptable" else "no"
    return row


def _rank_runs(runs: list[dict]) -> list[dict]:
    return sorted(runs, key=lambda run: (_area_rank(run), _to_float(run.get("mean_iou")) or float("-inf")), reverse=True)


def _area_rank(run: dict) -> int:
    band = _area_ratio_band(run.get("pred_label_area_ratio"))
    if band == "acceptable":
        return 2
    if band == "caution":
        return 1
    return 0


def _area_ratio_band(value) -> str:
    ratio = _to_float(value)
    if not isinstance(ratio, float):
        return "missing"
    for label, low, high in AREA_BANDS:
        if low <= ratio <= high:
            return label
    if ratio > 4.0:
        return "blocked_overexpansion"
    return "blocked_underprediction"


def _operating_point_markdown(runs: list[dict], threshold_rows: list[dict]) -> str:
    lines = [
        "# Validation Operating-Point Audit",
        "",
        "This report is validation-only. It freezes a review surface before model claims or full training decisions.",
        "",
        "## Candidate Ranking",
        "",
        "| Run | Family | Final mean IoU | Precision | Recall | Area ratio | Band | Primary candidate |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in _rank_runs(runs):
        row = _ranking_row(run)
        lines.append(
            f"| `{row['run']}` | {row['model_family']} | {_fmt(row['final_mean_iou'])} | "
            f"{_fmt(row['foreground_precision'])} | {_fmt(row['foreground_recall'])} | "
            f"{_fmt(row['pred_label_area_ratio'])} | {row['area_ratio_band']} | {row['primary_candidate_allowed']} |"
        )
    lines.extend(
        [
            "",
            "## Threshold Policy",
            "",
            "- Thresholds are selected from validation diagnostics only.",
            "- Area ratio is a first-class gate: acceptable `0.5x-2.0x`, caution `2.0x-4.0x`, blocked above `4.0x` unless used only as high-recall diagnostic behavior.",
            "- This audit does not choose a final model.",
            f"- Threshold rows inspected: `{len(threshold_rows)}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sealed_split_guard_markdown(runs: list[dict]) -> str:
    flagged = [run for run in runs if _to_bool(run.get("test_split_read"))]
    status = "FAIL" if flagged else "PASS"
    lines = [
        "# Sealed Split Guard Report",
        "",
        f"Status: **{status}**",
        "",
    ]
    if flagged:
        lines.append("Runs with sealed-test access evidence:")
        for run in flagged:
            lines.append(f"- `{run['run']}` from `{run['source']}` reported `test_split_read={run.get('test_split_read')}`.")
    else:
        lines.append("No sealed test reads were found in the inspected run manifests or compact metrics.")
    lines.extend(
        [
            "",
            "This guard is evidence for the diagnostic audit only. Implementation should keep emitting `run_manifest.json` with `test_split_read=false` and retain source paths in downstream reports.",
        ]
    )
    return "\n".join(lines) + "\n"


def _hard_negative_markdown(rows: list[dict]) -> str:
    lines = [
        "# Hard-Negative Inventory",
        "",
        "This inventory uses validation evidence only to seed a false-positive taxonomy. It must not convert validation examples into training candidates.",
        "",
        "## Taxonomy Seed",
        "",
        "- mountain_texture",
        "- bright_bare_soil",
        "- cloud_shadow",
        "- roads_drainage_settlement_edges",
        "- water_margins",
        "- burn_like_terrain",
        "- needs_manual_review",
        "",
        f"Validation false-positive rows indexed: `{len(rows)}`.",
    ]
    return "\n".join(lines) + "\n"


def _hard_negative_taxonomy_yaml() -> str:
    return """categories:
  mountain_texture: high-contrast slopes/ridges that can match broad foreground expansion
  bright_bare_soil: high brightness and sparse vegetation resembling exposed scars
  cloud_shadow: spectral extremes and edges that can dominate derived indices
  roads_drainage_settlement_edges: linear high-contrast features
  water_margins: sharp spectral transitions and index artifacts
  burn_like_terrain: low vegetation and dark/bright contrast
  needs_manual_review: insufficient evidence for automatic labeling
rules:
  validation_rows_define_taxonomy_only: true
  validation_rows_must_not_be_training_candidates: true
  train_candidates_require_train_split: true
"""


def _channel_audit_markdown(probability_rows: list[dict], coverage_rows: list[dict]) -> str:
    return "\n".join(
        [
            "# Channel Contribution And Normalization Audit",
            "",
            "This diagnostic entrypoint summarizes available probability and foreground-coverage evidence. It does not run training.",
            "",
            f"- Foreground probability summary rows found: `{len(probability_rows)}`.",
            f"- Foreground window coverage rows found: `{len(coverage_rows)}`.",
            "- Required follow-up: scan train/validation channel ranges for finite counts, min/max, percentiles, and derived-index scaling.",
            "- Required follow-up: run channel contribution only as isolated evaluation/inference with QA-visible logs.",
            "- 11-channel remains an audit/repair route until this evidence explains instability.",
        ]
    ) + "\n"


def _probe_protocol_markdown() -> str:
    return "\n".join(
        [
            "# Controlled Probe Protocol",
            "",
            "Probe design is deferred until diagnostic artifacts pass QA. This file is a protocol, not a training launch plan.",
            "",
            "## Candidate Roles",
            "",
            "| Candidate | Role |",
            "| --- | --- |",
            "| DeepLabV3+ 11-channel | leading reference candidate only |",
            "| RGB U-Net | stable baseline |",
            "| Transformer-UNet 11-channel | conservative-area comparator |",
            "| Repaired 11-channel U-Net | audit/repair route only if diagnostics support repair |",
            "",
            "## Explicitly deferred",
            "",
            "- `python -m segmentation_training train ...`",
            "",
            "Full cloud training remains blocked until diagnostic artifacts pass QA and the controlled probe protocol is accepted.",
        ]
    ) + "\n"


def _probe_matrix_rows() -> list[dict]:
    return [
        {
            "candidate": "DeepLabV3+ 11-channel",
            "role": "leading_reference_candidate",
            "changed_factor": "none_until_diagnostics_accepted",
            "training_authorized": "no",
        },
        {
            "candidate": "RGB U-Net",
            "role": "stable_baseline",
            "changed_factor": "input_channels_only_if_probe_approved",
            "training_authorized": "no",
        },
        {
            "candidate": "Transformer-UNet 11-channel",
            "role": "conservative_area_comparator",
            "changed_factor": "architecture_only_if_probe_approved",
            "training_authorized": "no",
        },
        {
            "candidate": "Repaired 11-channel U-Net",
            "role": "audit_repair_route",
            "changed_factor": "normalization_or_channel_repair_only_if_wp4_supports",
            "training_authorized": "no",
        },
    ]


def _probe_acceptance_markdown() -> str:
    return "\n".join(
        [
            "# Probe Acceptance Criteria",
            "",
            "- One changed factor per probe.",
            "- Same validation split, threshold grid, checkpoint policy, and area-ratio reporting.",
            "- `run_manifest.json` must report `test_split_read=false`.",
            "- No broad architecture sweep.",
            "- Full cloud training remains blocked until QA review.",
        ]
    ) + "\n"


def _implementation_scope_guard_markdown() -> str:
    return "\n".join(
        [
            "# Implementation Scope Guard",
            "",
            "Allowed now:",
            "",
            "- diagnostic tooling and reports",
            "- validation-only aggregation",
            "- sealed split guard reporting",
            "- dry-run/config validation",
            "- isolated evaluation artifacts",
            "",
            "Not allowed now:",
            "",
            "- sealed-test reads",
            "- full cloud training",
            "- using validation hard negatives as train candidates",
            "- final model winner claims",
        ]
    ) + "\n"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_foreground_row(path: Path) -> dict:
    for row in _read_csv(path):
        if row.get("class_id") not in {"", "0", "background", "255"}:
            return row
    return {}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _to_float(value) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _fmt(value) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.6g}"

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .manifest import DISASTER_ID_BY_CLASS_ID, ModelInputRecord, load_model_input_manifest, validate_research_contract


REQUIRED_RESEARCH_FIELDS = ("disaster_id", "label_confidence", "ignore_mask_path")


@dataclass(frozen=True)
class AuditSource:
    name: str
    manifest_path: Path
    contract_dir: Path
    required_channels: tuple[str, ...]


def parse_audit_source(value: str) -> AuditSource:
    parts = value.split("|")
    if len(parts) != 4:
        raise ValueError("Audit source must be NAME|MANIFEST_PATH|CONTRACT_DIR|CHANNEL1,CHANNEL2")
    name, manifest_path, contract_dir, channels = parts
    required_channels = tuple(channel.strip() for channel in channels.split(",") if channel.strip())
    if not name.strip() or not required_channels:
        raise ValueError("Audit source name and channels are required")
    return AuditSource(
        name=name.strip(),
        manifest_path=Path(manifest_path),
        contract_dir=Path(contract_dir),
        required_channels=required_channels,
    )


def parse_bundle_source(value: str) -> AuditSource:
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError("Bundle source must be NAME|BUNDLE_DIR|CHANNEL1,CHANNEL2")
    name, bundle_dir, channels = parts
    bundle_path = Path(bundle_dir)
    return parse_audit_source(
        f"{name}|{bundle_path / 'manifests' / 'cloud_model_input_manifest.csv'}|{bundle_path}|{channels}"
    )


def build_research_contract_audit(
    sources: Iterable[AuditSource],
    *,
    output_dir: Path,
    required_disasters: Iterable[str] = ("C2", "C5"),
) -> dict:
    source_list = list(sources)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summaries = []
    load_errors: list[str] = []
    records_by_source: list[tuple[AuditSource, ModelInputRecord]] = []
    for source in source_list:
        try:
            records = load_model_input_manifest(source.manifest_path, required_channels=source.required_channels)
        except Exception as exc:
            records = []
            load_errors.append(f"{source.name}: {exc}")
        source_summaries.append(
            {
                "name": source.name,
                "manifest_path": str(source.manifest_path),
                "contract_dir": str(source.contract_dir),
                "required_channels": list(source.required_channels),
                "record_count": len(records),
            }
        )
        records_by_source.extend((source, record) for record in records)

    records = [record for _source, record in records_by_source]
    contract_errors = validate_research_contract(records, required_disasters=required_disasters)
    contract_errors.extend(load_errors)
    missing_required_fields = _missing_required_fields(records)
    split_disaster_counts = _split_disaster_counts(records)
    label_confidence_counts = _label_confidence_counts(records)
    ignore_mask_declared_count = sum(1 for record in records if record.row.get("ignore_mask_path", "").strip())
    missing_ignore_mask_count = _missing_ignore_mask_count(records_by_source)
    status = "ready_for_e1" if not contract_errors and not any(missing_required_fields.values()) else "blocked"

    audit = {
        "status": status,
        "source_count": len(source_summaries),
        "sources": source_summaries,
        "total_records": len(records),
        "required_disasters": list(required_disasters),
        "split_disaster_counts": split_disaster_counts,
        "missing_required_fields": missing_required_fields,
        "label_confidence_counts": label_confidence_counts,
        "ignore_mask_declared_count": ignore_mask_declared_count,
        "missing_ignore_mask_count": missing_ignore_mask_count,
        "contract_error_count": len(contract_errors),
        "contract_errors": contract_errors,
    }
    _write_json(output_dir / "audit.json", audit)
    _write_markdown(output_dir / "audit.md", audit)
    _write_errors_csv(output_dir / "contract_errors.csv", contract_errors)
    return audit


def _missing_required_fields(records: Iterable[ModelInputRecord]) -> dict[str, int]:
    counts = {field: 0 for field in REQUIRED_RESEARCH_FIELDS}
    for record in records:
        for field in REQUIRED_RESEARCH_FIELDS:
            if field not in record.row:
                counts[field] += 1
            elif field != "ignore_mask_path" and not record.row[field].strip():
                counts[field] += 1
    return counts


def _split_disaster_counts(records: Iterable[ModelInputRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        disaster = record.row.get("disaster_id", "").strip() or DISASTER_ID_BY_CLASS_ID.get(record.class_id, "unknown")
        counts.setdefault(record.split, {})
        counts[record.split][disaster] = counts[record.split].get(disaster, 0) + 1
    return counts


def _label_confidence_counts(records: Iterable[ModelInputRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.row.get("label_confidence", "").strip().lower()
        if not value:
            counts["missing"] += 1
        elif _is_number(value):
            counts["numeric"] += 1
        else:
            counts[value] += 1
    return dict(sorted(counts.items()))


def _missing_ignore_mask_count(records_by_source: Iterable[tuple[AuditSource, ModelInputRecord]]) -> int:
    missing = 0
    for source, record in records_by_source:
        value = record.row.get("ignore_mask_path", "").strip()
        if value and not (source.contract_dir / value).exists():
            missing += 1
    return missing


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _write_json(path: Path, audit: dict) -> None:
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_errors_csv(path: Path, errors: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "error"])
        writer.writeheader()
        for index, error in enumerate(errors, start=1):
            writer.writerow({"index": index, "error": error})


def _write_markdown(path: Path, audit: dict) -> None:
    lines = [
        "# E0 C2/C5 Data Label Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "## Summary",
        "",
        f"- Sources: {audit['source_count']}",
        f"- Records: {audit['total_records']}",
        f"- Contract errors: {audit['contract_error_count']}",
        f"- Ignore masks declared: {audit['ignore_mask_declared_count']}",
        f"- Missing ignore masks: {audit['missing_ignore_mask_count']}",
        "",
        "## Missing Required Fields",
        "",
        "| Field | Missing rows |",
        "| --- | ---: |",
    ]
    for field, count in audit["missing_required_fields"].items():
        lines.append(f"| {field} | {count} |")

    lines.extend(["", "## Split Disaster Counts", "", "| Split | Disaster | Records |", "| --- | --- | ---: |"])
    for split, disaster_counts in sorted(audit["split_disaster_counts"].items()):
        for disaster, count in sorted(disaster_counts.items()):
            lines.append(f"| {split} | {disaster} | {count} |")

    lines.extend(["", "## First Contract Errors", ""])
    if audit["contract_errors"]:
        for error in audit["contract_errors"][:20]:
            lines.append(f"- {error}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

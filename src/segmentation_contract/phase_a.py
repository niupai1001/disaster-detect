from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


MANIFEST_COLUMNS = [
    "sample_id",
    "event_id",
    "event_date",
    "scene_id",
    "source_raster_id",
    "source_band_paths",
    "candidate_scenes",
    "polygon_id",
    "source_csv",
    "source_row_number",
    "category_raw",
    "class_id",
    "class_name",
    "split",
    "mask_path",
    "crs",
    "transform",
    "height",
    "width",
    "nodata_policy",
    "qa_flags",
]

SKIPPED_COLUMNS = [
    "source_csv",
    "row_number",
    "event_key",
    "category_raw",
    "polygon_id",
    "wkt_snippet_or_hash",
    "parse_error",
    "action",
]

DEFAULT_CHANNELS = [f"F{idx:02d}" for idx in range(1, 18)]


@dataclass(frozen=True)
class PhaseAResult:
    output_dir: Path
    parsed_rows: int
    skipped_rows: int
    malformed_rows: int
    taxonomy_pending_rows: int
    leakage_check: str


def build_phase_a_contract(
    label_csvs: list[Path] | tuple[Path, ...],
    output_dir: Path,
    *,
    channel_names: list[str] | tuple[str, ...] | None = None,
    raster_paths: list[Path] | tuple[Path, ...] | None = None,
) -> PhaseAResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    category_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    events: set[str] = set()
    taxonomy_pending = 0
    malformed = 0

    raster_index = _build_raster_index(raster_paths or [])

    for csv_path in label_csvs:
        rows, skipped, pending, malformed_count = _read_label_csv(Path(csv_path), len(manifest_rows), raster_index)
        manifest_rows.extend(rows)
        skipped_rows.extend(skipped)
        taxonomy_pending += pending
        malformed += malformed_count

    split_by_event = _assign_splits(sorted({row["event_id"] for row in manifest_rows}))
    for row in manifest_rows:
        row["split"] = split_by_event[row["event_id"]]
        category_counts[row["category_raw"]] += 1
        class_counts[row["class_id"]] += 1
        split_counts[row["split"]] += 1
        events.add(row["event_id"])

    leakage_check = _leakage_check(manifest_rows)
    _write_json(output_dir / "class_map.json", _class_map())
    _write_json(output_dir / "channels.json", _channels(channel_names or DEFAULT_CHANNELS))
    _write_csv(output_dir / "samples_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
    _write_csv(output_dir / "skipped_label_rows.csv", SKIPPED_COLUMNS, skipped_rows)
    (output_dir / "split_policy.md").write_text(_split_policy(), encoding="utf-8")
    (output_dir / "dataset_contract.md").write_text(_dataset_contract(), encoding="utf-8")
    (output_dir / "phase_b_plan.md").write_text(_phase_b_plan(), encoding="utf-8")
    (output_dir / "README.md").write_text(_readme(), encoding="utf-8")
    (output_dir / "qa_summary.md").write_text(
        _qa_summary(
            parsed_rows=len(manifest_rows),
            skipped_rows=len(skipped_rows),
            malformed_rows=malformed,
            taxonomy_pending_rows=taxonomy_pending,
            category_counts=category_counts,
            class_counts=class_counts,
            split_counts=split_counts,
            event_count=len(events),
            leakage_check=leakage_check,
        ),
        encoding="utf-8",
    )

    return PhaseAResult(
        output_dir=output_dir,
        parsed_rows=len(manifest_rows),
        skipped_rows=len(skipped_rows),
        malformed_rows=malformed,
        taxonomy_pending_rows=taxonomy_pending,
        leakage_check=leakage_check,
    )


def _read_label_csv(
    csv_path: Path,
    sample_offset: int,
    raster_index: dict[tuple[str, str, str, str], dict[str, object]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], int, int]:
    manifest_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    taxonomy_pending = 0
    malformed = 0
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_number, row in enumerate(reader, start=2):
            normalized = {str(key or "").strip(): (value or "").strip() for key, value in row.items()}
            category_raw = normalized.get("category", "")
            polygon_id = normalized.get("poly_id", "")
            event_key = _event_id(normalized)
            wkt = normalized.get("geometry_wkt", "")

            skip_error = _skip_error(normalized, event_key)
            if skip_error:
                if skip_error == "invalid_wkt":
                    malformed += 1
                skipped_rows.append(_skipped_row(csv_path, row_number, event_key, category_raw, polygon_id, wkt, skip_error))
                continue

            class_id, class_name, qa_flags = _classify(category_raw)
            if class_id is None:
                skipped_rows.append(
                    _skipped_row(csv_path, row_number, event_key, category_raw, polygon_id, wkt, "unrecognized_category")
                )
                continue
            if "taxonomy_pending" in qa_flags:
                taxonomy_pending += 1

            sample_id = f"sample-{sample_offset + len(manifest_rows) + 1:06d}"
            raster_key = _raster_key(normalized, event_key)
            raster_match = raster_index.get(raster_key, {})
            candidate_scenes = raster_match.get("candidate_scenes", [])
            scene_band_paths = raster_match.get("band_paths", {})
            source_band_paths = _format_band_paths(scene_band_paths) if isinstance(scene_band_paths, dict) else ""
            source_raster_id = str(raster_match.get("source_raster_id", "")) if raster_match else ""
            scene_id = str(raster_match.get("scene_id", "")) if raster_match else ""
            event_date = raster_key[-1]
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "event_id": event_key,
                    "event_date": event_date,
                    "scene_id": scene_id or "pending_phase_b",
                    "source_raster_id": source_raster_id or _planned_raster_id(normalized, event_key),
                    "source_band_paths": source_band_paths,
                    "candidate_scenes": json.dumps(candidate_scenes, ensure_ascii=False, sort_keys=True),
                    "polygon_id": polygon_id,
                    "source_csv": str(csv_path),
                    "source_row_number": str(row_number),
                    "category_raw": category_raw,
                    "class_id": str(class_id),
                    "class_name": class_name,
                    "split": "unassigned",
                    "mask_path": f"masks/{sample_id}.tif",
                    "crs": "pending_geospatial_validation",
                    "transform": "pending_geospatial_validation",
                    "height": "pending_geospatial_validation",
                    "width": "pending_geospatial_validation",
                    "nodata_policy": "pending_geospatial_validation",
                    "qa_flags": ";".join(qa_flags) if qa_flags else "phase_a_only",
                }
            )
    return manifest_rows, skipped_rows, taxonomy_pending, malformed


def _classify(category_raw: str) -> tuple[int | None, str, list[str]]:
    category = category_raw.strip().upper()
    if category.startswith("C"):
        category = category[1:]
    if category == "2":
        return 1, "C2_debris_flow", []
    if category == "5":
        return 2, "C5_fire", []
    if category == "3":
        return 255, "ignore", ["taxonomy_pending"]
    return None, "", []


def _event_id(row: dict[str, str]) -> str:
    grid = row.get("grid_id", "").strip() or "unknown_grid"
    date = _normalize_date(row.get("evt_date", ""))
    category = row.get("category", "").strip() or "unknown_category"
    return f"C{category}_{grid}_EV{date}" if date else ""


def _normalize_date(value: str) -> str:
    parts = re.findall(r"\d+", value)
    if len(parts) >= 3:
        year, month, day = parts[:3]
        return f"{int(year):04d}{int(month):02d}{int(day):02d}"
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) == 8:
        return digits
    return digits


def _raster_key(row: dict[str, str], event_id: str) -> tuple[str, str, str, str]:
    category = row.get("category", "").strip().upper()
    if category.startswith("C"):
        category = category[1:]
    grid = row.get("grid_id", "").strip()
    polygon = row.get("poly_id", "").strip()
    event_date = event_id.rsplit("EV", 1)[-1] if "EV" in event_id else ""
    return category, grid, polygon, event_date


def _planned_raster_id(row: dict[str, str], event_id: str) -> str:
    category, grid, polygon, event_date = _raster_key(row, event_id)
    if category and grid and polygon and event_date:
        return f"C{category}_{grid}_{polygon}_EV{event_date}"
    return "pending_phase_b"


def _skip_error(row: dict[str, str], event_key: str) -> str | None:
    if not event_key:
        return "missing_event_id"
    if not row.get("poly_id", "").strip():
        return "missing_polygon_id"
    if not row.get("category", "").strip():
        return "missing_category"
    wkt = row.get("geometry_wkt", "")
    if not _looks_like_wkt(wkt):
        return "invalid_wkt"
    return None


def _looks_like_wkt(wkt: str) -> bool:
    value = wkt.strip().upper()
    return (value.startswith("POLYGON") or value.startswith("MULTIPOLYGON")) and "((" in value and "))" in value


def _build_raster_index(raster_paths: list[Path] | tuple[Path, ...]) -> dict[tuple[str, str, str, str], dict[str, object]]:
    scenes_by_key: dict[tuple[str, str, str, str], dict[str, dict[str, object]]] = {}
    for path in raster_paths:
        parsed = _parse_raster_name(Path(path))
        if not parsed:
            continue
        key = (parsed["category"], parsed["grid"], parsed["polygon"], parsed["event_date"])
        scene = scenes_by_key.setdefault(key, {}).setdefault(
            parsed["image_date"],
            {
                "scene_id": f"IM{parsed['image_date']}",
                "image_date": parsed["image_date"],
                "event_date": parsed["event_date"],
                "days_after_event": _days_between(parsed["event_date"], parsed["image_date"]),
                "temporal_role": _temporal_role(parsed["event_date"], parsed["image_date"]),
                "source_raster_id": f"C{parsed['category']}_{parsed['grid']}_{parsed['polygon']}_EV{parsed['event_date']}",
                "band_paths": {},
            },
        )
        band_paths = scene["band_paths"]
        if isinstance(band_paths, dict):
            band_paths.setdefault(parsed["band"], str(path))
    index: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for key, scenes in scenes_by_key.items():
        candidate_scenes = [scenes[image_date] for image_date in sorted(scenes)]
        primary_scene = _primary_scene(candidate_scenes)
        index[key] = {
            "source_raster_id": str(primary_scene.get("source_raster_id", "")) if primary_scene else "",
            "scene_id": str(primary_scene.get("scene_id", "")) if primary_scene else "",
            "band_paths": dict(primary_scene.get("band_paths", {})) if primary_scene else {},
            "candidate_scenes": candidate_scenes,
        }
    return index


def _primary_scene(candidate_scenes: list[dict[str, object]]) -> dict[str, object]:
    post_scenes = [
        scene
        for scene in candidate_scenes
        if str(scene.get("temporal_role", "")) in {"same_day", "post"}
        and str(scene.get("days_after_event", "")).lstrip("-").isdigit()
    ]
    return (post_scenes or candidate_scenes)[0] if candidate_scenes else {}


def _days_between(event_date: str, image_date: str) -> str:
    from datetime import datetime

    try:
        event = datetime.strptime(event_date, "%Y%m%d")
        image = datetime.strptime(image_date, "%Y%m%d")
    except ValueError:
        return ""
    return str((image - event).days)


def _temporal_role(event_date: str, image_date: str) -> str:
    days = _days_between(event_date, image_date)
    if days == "":
        return "unknown"
    value = int(days)
    if value < 0:
        return "pre"
    if value == 0:
        return "same_day"
    return "post"


def _parse_raster_name(path: Path) -> dict[str, str] | None:
    match = re.search(
        r"_C(?P<category>\d+)_(?P<grid>[0-9A-Z]+)_(?P<polygon>\d+)_EV(?P<event_date>\d{8})_IM(?P<image_date>\d{8})_(?P<band>F\d{2})",
        path.name,
    )
    if not match:
        return None
    return match.groupdict()


def _format_band_paths(band_paths: dict[str, str]) -> str:
    return ";".join(f"{band}:{band_paths[band]}" for band in sorted(band_paths))


def _skipped_row(
    csv_path: Path,
    row_number: int,
    event_key: str,
    category_raw: str,
    polygon_id: str,
    wkt: str,
    parse_error: str,
) -> dict[str, str]:
    return {
        "source_csv": str(csv_path),
        "row_number": str(row_number),
        "event_key": event_key,
        "category_raw": category_raw,
        "polygon_id": polygon_id,
        "wkt_snippet_or_hash": _wkt_snippet_or_hash(wkt),
        "parse_error": parse_error,
        "action": "skip",
    }


def _wkt_snippet_or_hash(wkt: str) -> str:
    if len(wkt) <= 96:
        return wkt
    digest = hashlib.sha1(wkt.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{wkt[:80]}... sha1:{digest}"


def _assign_splits(event_ids: list[str]) -> dict[str, str]:
    split_by_event: dict[str, str] = {}
    for index, event_id in enumerate(event_ids):
        bucket = index % 10
        if bucket < 7:
            split = "train"
        elif bucket < 9:
            split = "validation"
        else:
            split = "test"
        split_by_event[event_id] = split
    return split_by_event


def _leakage_check(rows: list[dict[str, str]]) -> str:
    splits_by_event: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_event[row["event_id"]].add(row["split"])
    return "pass" if all(len(splits) == 1 for splits in splits_by_event.values()) else "fail"


def _class_map() -> dict[str, dict[str, object]]:
    return {
        "0": {"name": "background", "name_zh": "\u80cc\u666f", "trainable": True, "ignore": False},
        "1": {"name": "C2_debris_flow", "name_zh": "C2_\u6ce5\u77f3\u6d41", "trainable": True, "ignore": False},
        "2": {"name": "C5_fire", "name_zh": "C5_\u706b\u707e", "trainable": True, "ignore": False},
        "255": {"name": "ignore", "name_zh": "\u5ffd\u7565", "trainable": False, "ignore": True},
    }


def _channels(channel_names: list[str] | tuple[str, ...]) -> dict[str, object]:
    channels = []
    for index, name in enumerate(channel_names):
        channels.append(
            {
                "index": index,
                "name": name,
                "source": "raw_band",
                "required": True,
                "derived": False,
                "formula": None,
                "source_bands": [name],
                "nodata_policy": "pending_phase_b",
                "normalization_policy": "pending_phase_b",
                "status": "planned",
            }
        )
    return {"version": "0.1", "phase": "A", "channels": channels}


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _qa_summary(
    *,
    parsed_rows: int,
    skipped_rows: int,
    malformed_rows: int,
    taxonomy_pending_rows: int,
    category_counts: Counter[str],
    class_counts: Counter[str],
    split_counts: Counter[str],
    event_count: int,
    leakage_check: str,
) -> str:
    lines = [
        "# QA Summary",
        "",
        "phase: A",
        "phase_a_readiness: schema_manifest_review_only",
        "phase_b_readiness: not_started",
        f"parsed_rows: {parsed_rows}",
        f"skipped_rows: {skipped_rows}",
        f"malformed_rows: {malformed_rows}",
        f"taxonomy_pending_rows: {taxonomy_pending_rows}",
        f"event_count: {event_count}",
        f"leakage_check: {leakage_check}",
        "",
        "## Rows By Raw Category",
        *_counter_lines(category_counts),
        "",
        "## Rows By Class ID",
        *_counter_lines(class_counts),
        "",
        "## Split Counts",
        *_counter_lines(split_counts),
        "",
        "## Training Blockers",
        "- Phase B geospatial mask alignment has not run.",
        "- No model training is authorized from Phase A artifacts alone.",
    ]
    return "\n".join(lines) + "\n"


def _counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- none: 0"]
    return [f"- {key}: {counter[key]}" for key in sorted(counter)]


def _split_policy() -> str:
    return """# Split Policy

The primary split grouping key is `event_id`.

- No event may cross train, validation, and test splits.
- No polygon, raster chip, temporal pair, duplicate tile, or derived augmentation from the same event may cross splits.
- Validation and test splits must not be tuned against during prototype iteration.
- Split summaries must report event, polygon, raster, and class counts.
"""


def _dataset_contract() -> str:
    return """# SegmentationDatasetContract v0.1

This Phase A contract defines schema-level artifacts for multispectral semantic segmentation.

It supports C2 debris flow and C5 fire as trainable foreground classes. C3 is taxonomy pending and maps to `255=ignore` until confirmed.

Phase A does not produce geospatially validated masks and does not authorize model training.
"""


def _phase_b_plan() -> str:
    return """# Phase B Plan

Phase B will introduce `rasterio` and `shapely` to rasterize WKT polygons into single-band integer masks aligned to reference rasters.

Required outputs are `masks/`, `previews/`, `reports/wkt_mask_alignment_report.csv`, and `reports/geometry_validity_report.csv`.
"""


def _readme() -> str:
    return """# SegmentationDatasetContract v0.1 Phase A

Generated by `segmentation_contract` without geospatial dependencies.

These artifacts are for schema and manifest review only. They do not authorize model training.
"""

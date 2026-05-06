from __future__ import annotations

import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .phase_a import MANIFEST_COLUMNS


GEOMETRY_REPORT_COLUMNS = [
    "sample_id",
    "source_csv",
    "source_row_number",
    "is_valid",
    "is_empty",
    "area",
    "bounds",
    "geometry_status",
    "blocker",
]

ALIGNMENT_REPORT_COLUMNS = [
    "sample_id",
    "source_raster_id",
    "reference_raster_path",
    "mask_path",
    "crs",
    "transform",
    "height",
    "width",
    "foreground_pixels",
    "alignment_status",
    "blocker",
]


@dataclass(frozen=True)
class PhaseBPreflightResult:
    contract_dir: Path
    readiness: str
    shapely_available: bool
    rasterio_available: bool
    manifest_rows: int


@dataclass(frozen=True)
class PhaseBMaskResult:
    contract_dir: Path
    masks_written: int
    rows_processed: int
    rows_blocked: int


@dataclass(frozen=True)
class VisualQAResult:
    contract_dir: Path
    training_rows: int
    preview_items: int
    contact_sheet_path: Path


@dataclass(frozen=True)
class ChannelAlignmentResult:
    contract_dir: Path
    audited_rows: int
    model_input_rows: int
    required_channels: tuple[str, ...]


CHANNEL_ALIGNMENT_COLUMNS = [
    "sample_id",
    "class_id",
    "class_name",
    "split",
    "required_channels",
    "available_channels",
    "channel_alignment_status",
    "reference_shape",
    "mismatch_detail",
    "blocker",
]

MODEL_INPUT_COLUMNS = MANIFEST_COLUMNS + ["input_channels", "input_band_paths"]


def build_phase_b_preflight(contract_dir: Path) -> PhaseBPreflightResult:
    contract_dir = Path(contract_dir)
    masks_dir = contract_dir / "masks"
    previews_dir = contract_dir / "previews"
    reports_dir = contract_dir / "reports"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest_rows(contract_dir / "samples_manifest.csv")
    shapely_available = _module_available("shapely")
    rasterio_available = _module_available("rasterio")
    readiness = "ready_for_geospatial_validation" if shapely_available and rasterio_available else "blocked_missing_geospatial_dependencies"

    _write_geometry_report(reports_dir / "geometry_validity_report.csv", manifest_rows, readiness)
    _write_alignment_report(reports_dir / "wkt_mask_alignment_report.csv", manifest_rows, readiness)
    (reports_dir / "phase_b_dependency_report.md").write_text(
        _dependency_report(
            readiness=readiness,
            shapely_available=shapely_available,
            rasterio_available=rasterio_available,
            manifest_count=len(manifest_rows),
        ),
        encoding="utf-8",
    )

    return PhaseBPreflightResult(
        contract_dir=contract_dir,
        readiness=readiness,
        shapely_available=shapely_available,
        rasterio_available=rasterio_available,
        manifest_rows=len(manifest_rows),
    )


def build_phase_b_masks(
    contract_dir: Path,
    *,
    limit: int | None = None,
    reference_channels: list[str] | tuple[str, ...] = ("F16", "F17"),
) -> PhaseBMaskResult:
    _require_module("shapely")
    _require_module("rasterio")
    import rasterio
    from rasterio.features import rasterize
    from shapely import wkt
    from shapely.geometry import box

    contract_dir = Path(contract_dir)
    masks_dir = contract_dir / "masks"
    previews_dir = contract_dir / "previews"
    reports_dir = contract_dir / "reports"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest_rows(contract_dir / "samples_manifest.csv")
    if limit is not None:
        manifest_rows = manifest_rows[:limit]

    geometry_rows: list[dict[str, str]] = []
    alignment_rows: list[dict[str, str]] = []
    masks_written = 0
    manifest_updates: dict[str, dict[str, str]] = {}

    for row in manifest_rows:
        geometry = None
        geometry_status = "not_checked"
        geometry_blocker = ""
        try:
            source = _read_source_row(Path(row.get("source_csv", "")), int(row.get("source_row_number", "0")))
            geometry = wkt.loads(source.get("geometry_wkt", ""))
            geometry_status = _geometry_status(geometry)
            if geometry_status != "valid":
                geometry_blocker = geometry_status
        except Exception as exc:
            geometry_status = "invalid_wkt"
            geometry_blocker = str(exc)

        geometry_rows.append(_geometry_report_row(row, geometry, geometry_status, geometry_blocker))

        reference_raster = _reference_raster_path(row, reference_channels=reference_channels)
        if not reference_raster:
            alignment_rows.append(
                _alignment_report_row(row, "", "", "", "", "", "", "0", "blocked_missing_reference_raster")
            )
            continue
        if geometry is None or geometry_status != "valid":
            alignment_rows.append(
                _alignment_report_row(row, str(reference_raster), "", "", "", "", "", "0", "blocked_invalid_geometry")
            )
            continue

        mask_path = contract_dir / row.get("mask_path", "")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(reference_raster) as src:
            raster_bounds = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            if not geometry.intersects(raster_bounds):
                alignment_rows.append(
                    _alignment_report_row(
                        row,
                        str(reference_raster),
                        str(mask_path),
                        str(src.crs),
                        str(src.transform),
                        str(src.height),
                        str(src.width),
                        "0",
                        "blocked_geometry_outside_raster",
                    )
                )
                continue
            mask = rasterize(
                [(geometry, int(row.get("class_id") or 0))],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )
            profile = src.profile.copy()
            profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(mask, 1)
            foreground_pixels = str(int((mask == int(row.get("class_id") or 0)).sum()))
            masks_written += 1
            manifest_updates[row.get("sample_id", "")] = {
                "crs": str(src.crs),
                "transform": str(src.transform),
                "height": str(src.height),
                "width": str(src.width),
                "nodata_policy": "mask_nodata_255_background_0",
            }
            alignment_rows.append(
                _alignment_report_row(
                    row,
                    str(reference_raster),
                    str(mask_path),
                    str(src.crs),
                    str(src.transform),
                    str(src.height),
                    str(src.width),
                    foreground_pixels,
                    "mask_written",
                )
            )

    if manifest_updates:
        _update_manifest(contract_dir / "samples_manifest.csv", manifest_updates)
    _write_csv(reports_dir / "geometry_validity_report.csv", GEOMETRY_REPORT_COLUMNS, geometry_rows)
    _write_csv(reports_dir / "wkt_mask_alignment_report.csv", ALIGNMENT_REPORT_COLUMNS, alignment_rows)
    (reports_dir / "phase_b_dependency_report.md").write_text(
        _dependency_report(
            readiness="masks_generated" if masks_written else "blocked_no_masks_written",
            shapely_available=True,
            rasterio_available=True,
            manifest_count=len(manifest_rows),
        ),
        encoding="utf-8",
    )

    return PhaseBMaskResult(
        contract_dir=contract_dir,
        masks_written=masks_written,
        rows_processed=len(manifest_rows),
        rows_blocked=len(manifest_rows) - masks_written,
    )


def build_visual_qa(contract_dir: Path, *, max_items: int = 36) -> VisualQAResult:
    _require_module("rasterio")
    _require_module("PIL")
    import numpy
    import rasterio
    from PIL import Image, ImageDraw

    contract_dir = Path(contract_dir)
    previews_dir = contract_dir / "previews"
    reports_dir = contract_dir / "reports"
    previews_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_manifest_rows(contract_dir / "samples_manifest.csv")
    alignment_rows = _read_alignment_rows(contract_dir / "reports" / "wkt_mask_alignment_report.csv")
    alignment_by_sample = {row["sample_id"]: row for row in alignment_rows}
    training_rows = [
        row
        for row in manifest_rows
        if alignment_by_sample.get(row.get("sample_id", ""), {}).get("alignment_status") == "mask_written"
        and row.get("class_id") != "255"
    ]
    _write_csv(contract_dir / "training_manifest.csv", MANIFEST_COLUMNS, training_rows)

    preview_rows = _sample_preview_rows(training_rows, max_items)
    tiles = []
    for row in preview_rows:
        alignment = alignment_by_sample.get(row["sample_id"], {})
        reference_path = Path(alignment.get("reference_raster_path", ""))
        mask_path = Path(alignment.get("mask_path", ""))
        if not reference_path.exists() or not mask_path.exists():
            continue
        with rasterio.open(reference_path) as src:
            image = src.read(1)
        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1)
        tile = _overlay_tile(Image, ImageDraw, numpy, image, mask, row)
        tile_path = previews_dir / f"{row['sample_id']}_overlay.png"
        tile.save(tile_path)
        tiles.append(tile)

    contact_sheet_path = previews_dir / "contact_sheet.png"
    if tiles:
        _contact_sheet(Image, tiles).save(contact_sheet_path)
    else:
        Image.new("RGB", (512, 256), "white").save(contact_sheet_path)

    (reports_dir / "visual_qa_summary.md").write_text(
        _visual_qa_summary(
            manifest_rows=len(manifest_rows),
            training_rows=len(training_rows),
            preview_items=len(tiles),
            ignored_rows=sum(1 for row in manifest_rows if row.get("class_id") == "255"),
            contact_sheet_path=contact_sheet_path,
        ),
        encoding="utf-8",
    )

    return VisualQAResult(
        contract_dir=contract_dir,
        training_rows=len(training_rows),
        preview_items=len(tiles),
        contact_sheet_path=contact_sheet_path,
    )


def build_channel_alignment_audit(
    contract_dir: Path,
    *,
    required_channels: list[str] | tuple[str, ...] = ("F16", "F17"),
) -> ChannelAlignmentResult:
    _require_module("rasterio")
    import rasterio

    contract_dir = Path(contract_dir)
    reports_dir = contract_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    required = tuple(required_channels)
    training_rows = _read_manifest_rows(contract_dir / "training_manifest.csv")
    audit_rows: list[dict[str, str]] = []
    model_rows: list[dict[str, str]] = []

    for row in training_rows:
        band_paths = _parse_band_paths(row.get("source_band_paths", ""))
        available = sorted(band_paths)
        missing = [band for band in required if band not in band_paths or not band_paths[band].exists()]
        if missing:
            audit_rows.append(
                _channel_audit_row(
                    row,
                    required,
                    available,
                    "blocked_missing_required_channels",
                    "",
                    "",
                    f"missing: {';'.join(missing)}",
                )
            )
            continue

        metas = []
        for band in required:
            with rasterio.open(band_paths[band]) as ds:
                metas.append(
                    (
                        band,
                        str(ds.crs),
                        tuple(round(value, 12) for value in tuple(ds.transform)[:6]),
                        ds.height,
                        ds.width,
                    )
                )
        shape_key = {(crs, transform, height, width) for _band, crs, transform, height, width in metas}
        if len(shape_key) != 1:
            detail = "; ".join(f"{band}:{height}x{width}" for band, _crs, _transform, height, width in metas)
            audit_rows.append(
                _channel_audit_row(row, required, available, "blocked_grid_mismatch", "", detail, "grid_mismatch")
            )
            continue

        band_grid = next(iter(shape_key))
        mask_path = Path(row.get("mask_path", ""))
        if not mask_path.is_absolute():
            mask_path = contract_dir / mask_path
        if not mask_path.exists():
            audit_rows.append(
                _channel_audit_row(row, required, available, "blocked_missing_mask", "", str(mask_path), "missing_mask")
            )
            continue
        with rasterio.open(mask_path) as mask_ds:
            mask_grid = (
                str(mask_ds.crs),
                tuple(round(value, 12) for value in tuple(mask_ds.transform)[:6]),
                mask_ds.height,
                mask_ds.width,
            )
        if mask_grid != band_grid:
            detail = f"mask:{mask_grid[2]}x{mask_grid[3]}; bands:{band_grid[2]}x{band_grid[3]}"
            audit_rows.append(
                _channel_audit_row(
                    row,
                    required,
                    available,
                    "blocked_mask_grid_mismatch",
                    "",
                    detail,
                    "mask_grid_mismatch",
                )
            )
            continue

        _crs, _transform, height, width = band_grid
        input_band_paths = ";".join(f"{band}:{band_paths[band]}" for band in required)
        audit_rows.append(
            _channel_audit_row(row, required, available, "aligned", f"{height}x{width}", "", "")
        )
        model_row = dict(row)
        model_row["input_channels"] = ";".join(required)
        model_row["input_band_paths"] = input_band_paths
        model_rows.append(model_row)

    _write_csv(reports_dir / "channel_alignment_report.csv", CHANNEL_ALIGNMENT_COLUMNS, audit_rows)
    _write_csv(contract_dir / "model_input_manifest.csv", MODEL_INPUT_COLUMNS, model_rows)
    (reports_dir / "model_input_summary.md").write_text(
        _model_input_summary(required_channels=required, audited_rows=len(training_rows), model_input_rows=len(model_rows)),
        encoding="utf-8",
    )
    return ChannelAlignmentResult(
        contract_dir=contract_dir,
        audited_rows=len(training_rows),
        model_input_rows=len(model_rows),
        required_channels=required,
    )


def _parse_band_paths(text: str) -> dict[str, Path]:
    parsed = {}
    for item in text.split(";"):
        if ":" not in item:
            continue
        band, path = item.split(":", 1)
        parsed[band] = Path(path)
    return parsed


def _channel_audit_row(
    row: dict[str, str],
    required_channels: tuple[str, ...],
    available_channels: list[str],
    status: str,
    reference_shape: str,
    mismatch_detail: str,
    blocker: str,
) -> dict[str, str]:
    return {
        "sample_id": row.get("sample_id", ""),
        "class_id": row.get("class_id", ""),
        "class_name": row.get("class_name", ""),
        "split": row.get("split", ""),
        "required_channels": ";".join(required_channels),
        "available_channels": ";".join(available_channels),
        "channel_alignment_status": status,
        "reference_shape": reference_shape,
        "mismatch_detail": mismatch_detail,
        "blocker": blocker,
    }


def _model_input_summary(*, required_channels: tuple[str, ...], audited_rows: int, model_input_rows: int) -> str:
    return f"""# Model Input Summary

required_channels: {';'.join(required_channels)}
audited_training_rows: {audited_rows}
model_input_rows: {model_input_rows}
excluded_rows: {audited_rows - model_input_rows}

## Scope

`model_input_manifest.csv` is the conservative input contract for later training code.
It includes only rows from `training_manifest.csv` whose required channels share CRS, transform, height, and width.
This step does not train a model or select a model architecture.
"""


def _read_alignment_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _sample_preview_rows(rows: list[dict[str, str]], max_items: int) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("class_name", ""), []).append(row)
    sampled: list[dict[str, str]] = []
    while len(sampled) < max_items:
        added = False
        for class_name in sorted(grouped):
            class_rows = grouped[class_name]
            if class_rows:
                sampled.append(class_rows.pop(0))
                added = True
                if len(sampled) >= max_items:
                    break
        if not added:
            break
    return sampled


def _overlay_tile(Image, ImageDraw, numpy, image, mask, row: dict[str, str]):
    base = _normalize_to_uint8(numpy, image)
    rgb = numpy.stack([base, base, base], axis=-1)
    foreground = mask > 0
    rgb[foreground, 0] = 255
    rgb[foreground, 1] = (rgb[foreground, 1] * 0.35).astype("uint8")
    rgb[foreground, 2] = (rgb[foreground, 2] * 0.35).astype("uint8")
    tile = Image.fromarray(rgb, mode="RGB")
    tile.thumbnail((256, 256))
    canvas = Image.new("RGB", (280, 320), "white")
    canvas.paste(tile, ((280 - tile.width) // 2, 8))
    draw = ImageDraw.Draw(canvas)
    label = f"{row.get('sample_id')} | {row.get('class_name')} | {row.get('split')}"
    draw.text((8, 272), label, fill=(0, 0, 0))
    draw.text((8, 292), f"fg px: {int((mask > 0).sum())}", fill=(80, 80, 80))
    return canvas


def _normalize_to_uint8(numpy, image):
    array = image.astype("float64", copy=False)
    valid = array[numpy.isfinite(array)]
    if valid.size == 0:
        return numpy.zeros(array.shape, dtype="uint8")
    low, high = numpy.percentile(valid, [2, 98])
    if high <= low:
        high = low + 1
    scaled = (array - low) / (high - low)
    scaled = numpy.clip(scaled, 0, 1)
    scaled = numpy.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return (scaled * 255).astype("uint8")


def _contact_sheet(Image, tiles):
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 280, rows * 320), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * 280
        y = (index // columns) * 320
        sheet.paste(tile, (x, y))
    return sheet


def _visual_qa_summary(
    *,
    manifest_rows: int,
    training_rows: int,
    preview_items: int,
    ignored_rows: int,
    contact_sheet_path: Path,
) -> str:
    return f"""# Visual QA Summary

manifest_rows: {manifest_rows}
training_manifest_rows: {training_rows}
ignored_class_255_rows_excluded: {ignored_rows}
preview_items: {preview_items}
contact_sheet: {contact_sheet_path}

## Training Rule

`training_manifest.csv` includes only rows with `alignment_status == mask_written` and `class_id != 255`.
C3 / `255=ignore` rows are kept in the audit artifacts but excluded from foreground training.
"""


def _update_manifest(path: Path, updates_by_sample_id: dict[str, dict[str, str]]) -> None:
    rows = _read_manifest_rows(path)
    for row in rows:
        update = updates_by_sample_id.get(row.get("sample_id", ""))
        if update:
            row.update(update)
    _write_csv(path, MANIFEST_COLUMNS, rows)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_module(name: str) -> None:
    if not _module_available(name):
        raise RuntimeError(f"Phase B requires `{name}`. Install it before building masks.")


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_source_row(path: Path, source_row_number: int) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_number, row in enumerate(reader, start=2):
            if row_number == source_row_number:
                return {str(key or "").strip(): (value or "").strip() for key, value in row.items()}
    raise ValueError(f"source row {source_row_number} not found in {path}")


def _reference_raster_path(
    row: dict[str, str],
    *,
    reference_channels: list[str] | tuple[str, ...] = ("F16", "F17"),
) -> Path | None:
    band_paths = row.get("source_band_paths", "")
    parsed = []
    for item in band_paths.split(";"):
        if ":" not in item:
            continue
        band, path = item.split(":", 1)
        parsed.append((band, Path(path)))
    by_band = {band: path for band, path in parsed}
    for band in reference_channels:
        path = by_band.get(band)
        if path is not None and path.exists():
            return path
    for _band, path in sorted(parsed):
        if path.exists():
            return path
    return None


def _geometry_status(geometry) -> str:
    if geometry.is_empty:
        return "empty_geometry"
    if not geometry.is_valid:
        return "invalid_geometry"
    if geometry.area <= 0:
        return "zero_area_geometry"
    return "valid"


def _geometry_report_row(row: dict[str, str], geometry, status: str, blocker: str) -> dict[str, str]:
    if geometry is None:
        is_valid = "false"
        is_empty = "unknown"
        area = ""
        bounds = ""
    else:
        is_valid = str(bool(geometry.is_valid)).lower()
        is_empty = str(bool(geometry.is_empty)).lower()
        area = f"{geometry.area:.12g}"
        bounds = ",".join(f"{value:.12g}" for value in geometry.bounds) if not geometry.is_empty else ""
    return {
        "sample_id": row.get("sample_id", ""),
        "source_csv": row.get("source_csv", ""),
        "source_row_number": row.get("source_row_number", ""),
        "is_valid": is_valid,
        "is_empty": is_empty,
        "area": area,
        "bounds": bounds,
        "geometry_status": status,
        "blocker": blocker,
    }


def _alignment_report_row(
    row: dict[str, str],
    reference_raster_path: str,
    mask_path: str,
    crs: str,
    transform: str,
    height: str,
    width: str,
    foreground_pixels: str,
    status: str,
) -> dict[str, str]:
    blocker = "" if status == "mask_written" else status
    return {
        "sample_id": row.get("sample_id", ""),
        "source_raster_id": row.get("source_raster_id", ""),
        "reference_raster_path": reference_raster_path,
        "mask_path": mask_path,
        "crs": crs,
        "transform": transform,
        "height": height,
        "width": width,
        "foreground_pixels": foreground_pixels,
        "alignment_status": status,
        "blocker": blocker,
    }


def _write_geometry_report(path: Path, rows: list[dict[str, str]], readiness: str) -> None:
    blocker = "" if readiness == "ready_for_geospatial_validation" else "install shapely before geometry validation"
    report_rows = [
        {
            "sample_id": row.get("sample_id", ""),
            "source_csv": row.get("source_csv", ""),
            "source_row_number": row.get("source_row_number", ""),
            "is_valid": "",
            "is_empty": "",
            "area": "",
            "bounds": "",
            "geometry_status": "pending_shapely_validation",
            "blocker": blocker,
        }
        for row in rows
    ]
    _write_csv(path, GEOMETRY_REPORT_COLUMNS, report_rows)


def _write_alignment_report(path: Path, rows: list[dict[str, str]], readiness: str) -> None:
    blocker = "" if readiness == "ready_for_geospatial_validation" else "install rasterio before mask alignment validation"
    report_rows = [
        {
            "sample_id": row.get("sample_id", ""),
            "source_raster_id": row.get("source_raster_id", ""),
            "reference_raster_path": "",
            "mask_path": row.get("mask_path", ""),
            "crs": "",
            "transform": "",
            "height": "",
            "width": "",
            "foreground_pixels": "0",
            "alignment_status": "pending_rasterio_validation",
            "blocker": blocker,
        }
        for row in rows
    ]
    _write_csv(path, ALIGNMENT_REPORT_COLUMNS, report_rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dependency_report(
    *,
    readiness: str,
    shapely_available: bool,
    rasterio_available: bool,
    manifest_count: int,
) -> str:
    shapely_status = "available" if shapely_available else "missing"
    rasterio_status = "available" if rasterio_available else "missing"
    if readiness == "masks_generated":
        scope = (
            "Phase B generated geospatial mask files and validation reports for rows with linked reference rasters. "
            "Rows without reference rasters remain blocked and must not be used for training."
        )
    else:
        scope = (
            "This preflight creates the Phase B artifact structure and reports pending geospatial validation state. "
            "It does not rasterize WKT polygons, write masks, create previews, or authorize model training."
        )
    return f"""# Phase B Dependency Report

phase: B
phase_b_readiness: {readiness}
manifest_rows: {manifest_count}
shapely: {shapely_status}
rasterio: {rasterio_status}

## Scope

{scope}
"""

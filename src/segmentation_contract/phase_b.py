from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from .phase_a import MANIFEST_COLUMNS
from .post_scene import select_post_disaster_scene


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
class ModelInputPreviewResult:
    contract_dir: Path
    preview_items: int
    contact_sheet_path: Path
    rgb_channels: tuple[str, ...]
    false_color_channels: tuple[str, ...]


@dataclass(frozen=True)
class ChannelAlignmentResult:
    contract_dir: Path
    audited_rows: int
    model_input_rows: int
    required_channels: tuple[str, ...]


@dataclass(frozen=True)
class PostDisasterModelInputResult:
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

POST_DISASTER_SELECTION_COLUMNS = [
    "event_date",
    "selected_image_date",
    "days_after_event",
    "temporal_role",
    "selected_scene_id",
    "channel_group_id",
    "selection_strategy",
    "quality_score",
    "quality_score_status",
    "quality_score_detail",
    "raw_channel_paths",
    "resampled_channel_paths",
    "derived_index_paths",
    "reference_channel",
    "reference_grid",
    "resampling_policy",
    "normalization_policy",
    "temporal_qa_flags",
    "band_mapping_version",
    "selection_status",
    "blocker",
]

MODEL_INPUT_V2_COLUMNS = [
    *MANIFEST_COLUMNS,
    *[column for column in POST_DISASTER_SELECTION_COLUMNS if column not in MANIFEST_COLUMNS],
    "input_channels",
    "input_band_paths",
]

POST_DISASTER_AUDIT_COLUMNS = [
    "sample_id",
    "event_id",
    "event_date",
    "class_id",
    "class_name",
    "split",
    "required_channels",
    "selection_status",
    "selected_image_date",
    "days_after_event",
    "temporal_role",
    "selected_scene_id",
    "channel_group_id",
    "selection_strategy",
    "quality_score",
    "quality_score_status",
    "quality_score_detail",
    "raw_channel_paths",
    "input_channels",
    "input_band_paths",
    "mask_path",
    "temporal_qa_flags",
    "blocker",
]

POST_DISASTER_SCENE_QUALITY_COLUMNS = [
    "sample_id",
    "event_id",
    "class_id",
    "class_name",
    "split",
    "scene_id",
    "image_date",
    "days_after_event",
    "required_channels",
    "has_required_channels",
    "post_window_status",
    "quality_score",
    "quality_score_status",
    "quality_score_detail",
    "selected",
]


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
        and row.get("split") != "test"
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


def build_model_input_previews(
    contract_dir: Path,
    *,
    max_items: int = 36,
    rgb_channels: tuple[str, str, str] = ("F03", "F02", "F01"),
    false_color_channels: tuple[str, str, str] = ("F11", "F07", "F03"),
) -> ModelInputPreviewResult:
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

    model_rows = [
        row
        for row in _read_manifest_rows(contract_dir / "model_input_manifest.csv")
        if row.get("split") != "test" and row.get("class_id") != "255"
    ]
    preview_rows = _sample_preview_rows(model_rows, max_items)
    tiles = []
    for row in preview_rows:
        band_paths = _parse_band_paths(row.get("input_band_paths", ""))
        mask_path = _resolve_preview_path(contract_dir, row.get("mask_path", ""))
        if not mask_path.exists():
            continue
        arrays = {}
        try:
            for channel, path in band_paths.items():
                resolved = _resolve_preview_path(contract_dir, str(path))
                if not resolved.exists():
                    arrays = {}
                    break
                with rasterio.open(resolved) as src:
                    arrays[channel] = src.read(1)
            if not arrays:
                continue
            with rasterio.open(mask_path) as mask_src:
                mask = mask_src.read(1)
        except Exception:
            continue
        tile = _model_input_tile(
            Image,
            ImageDraw,
            numpy,
            arrays,
            mask,
            row,
            rgb_channels=rgb_channels,
            false_color_channels=false_color_channels,
        )
        tile_path = previews_dir / f"model_input_{row['sample_id']}_composite.png"
        tile.save(tile_path)
        tiles.append(tile)

    contact_sheet_path = previews_dir / "model_input_contact_sheet.png"
    if tiles:
        _wide_contact_sheet(Image, tiles).save(contact_sheet_path)
    else:
        Image.new("RGB", (640, 256), "white").save(contact_sheet_path)

    (reports_dir / "model_input_visual_qa_summary.md").write_text(
        _model_input_visual_qa_summary(
            manifest_rows=len(model_rows),
            preview_items=len(tiles),
            contact_sheet_path=contact_sheet_path,
            rgb_channels=rgb_channels,
            false_color_channels=false_color_channels,
        ),
        encoding="utf-8",
    )
    return ModelInputPreviewResult(
        contract_dir=contract_dir,
        preview_items=len(tiles),
        contact_sheet_path=contact_sheet_path,
        rgb_channels=rgb_channels,
        false_color_channels=false_color_channels,
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


def build_post_disaster_model_input_manifest(
    contract_dir: Path,
    *,
    required_channels: list[str] | tuple[str, ...],
    derived_indices: list[str] | tuple[str, ...] = (),
    max_days_after_event: int | None = None,
    band_mapping_version: str = "unconfirmed",
    selection_strategy: str = "quality_then_earliest",
) -> PostDisasterModelInputResult:
    _require_module("rasterio")
    _require_module("numpy")
    import numpy
    import rasterio
    from rasterio.warp import Resampling, reproject

    contract_dir = Path(contract_dir)
    reports_dir = contract_dir / "reports"
    model_inputs_dir = contract_dir / "model_inputs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_inputs_dir.mkdir(parents=True, exist_ok=True)
    required = tuple(required_channels)
    derived = tuple(index.upper() for index in derived_indices)
    manifest_path = contract_dir / "training_manifest.csv"
    if not manifest_path.exists():
        manifest_path = contract_dir / "samples_manifest.csv"
    candidate_rows = [
        row
        for row in _read_manifest_rows(manifest_path)
        if row.get("split") in {"train", "validation"} and row.get("class_id") != "255"
    ]
    audit_rows: list[dict[str, str]] = []
    model_rows: list[dict[str, str]] = []
    scene_quality_rows: list[dict[str, str]] = []

    for row in candidate_rows:
        scene_quality_scores = _score_candidate_scenes(
            contract_dir=contract_dir,
            row=row,
            required_channels=required,
            rasterio=rasterio,
            resampling=Resampling.average,
            numpy=numpy,
        )
        selection = select_post_disaster_scene(
            row,
            required_channels=required,
            max_days_after_event=max_days_after_event,
            selection_strategy=selection_strategy,
            scene_quality_scores=scene_quality_scores,
        )
        scene_quality_rows.extend(
            _scene_quality_report_rows(
                row=row,
                required_channels=required,
                max_days_after_event=max_days_after_event,
                selection=selection,
                scene_quality_scores=scene_quality_scores,
            )
        )
        if selection["selection_status"] != "selected":
            audit_rows.append({column: selection.get(column, "") for column in POST_DISASTER_AUDIT_COLUMNS})
            continue
        try:
            aligned_paths, reference_grid = _align_selected_channels_to_mask_grid(
                contract_dir=contract_dir,
                model_inputs_dir=model_inputs_dir,
                row=row,
                selection=selection,
                required_channels=required,
                rasterio=rasterio,
                reproject=reproject,
                resampling=Resampling.bilinear,
            )
            derived_paths = _derive_index_channels(
                aligned_paths=aligned_paths,
                model_inputs_dir=model_inputs_dir,
                row=row,
                derived_indices=derived,
                rasterio=rasterio,
                numpy=numpy,
            )
        except Exception as exc:
            blocked_selection = dict(selection)
            blocked_selection["selection_status"] = "blocked_resampling_failed"
            blocked_selection["blocker"] = str(exc)
            blocked_selection["temporal_qa_flags"] = "blocked_resampling_failed"
            audit_rows.append({column: blocked_selection.get(column, "") for column in POST_DISASTER_AUDIT_COLUMNS})
            continue
        audit_rows.append({column: selection.get(column, "") for column in POST_DISASTER_AUDIT_COLUMNS})
        model_row = dict(row)
        model_row.update(
            {
                "event_date": selection["event_date"],
                "selected_image_date": selection["selected_image_date"],
                "days_after_event": selection["days_after_event"],
                "temporal_role": selection["temporal_role"],
                "selected_scene_id": selection["selected_scene_id"],
                "channel_group_id": selection["channel_group_id"],
                "selection_strategy": selection["selection_strategy"],
                "quality_score": selection["quality_score"],
                "quality_score_status": selection["quality_score_status"],
                "quality_score_detail": selection["quality_score_detail"],
                "raw_channel_paths": selection["raw_channel_paths"],
                "resampled_channel_paths": _format_band_paths(aligned_paths),
                "derived_index_paths": _format_band_paths(derived_paths),
                "reference_channel": required[0] if required else "",
                "reference_grid": reference_grid,
                "resampling_policy": "bilinear_to_mask_grid",
                "normalization_policy": "train_only_stats_required_before_training",
                "temporal_qa_flags": selection["temporal_qa_flags"],
                "band_mapping_version": band_mapping_version,
                "selection_status": selection["selection_status"],
                "blocker": "",
                "input_channels": ";".join((*required, *derived)),
                "input_band_paths": _format_band_paths({**aligned_paths, **derived_paths}),
            }
        )
        model_rows.append(model_row)

    _write_csv(reports_dir / "post_disaster_channel_audit.csv", POST_DISASTER_AUDIT_COLUMNS, audit_rows)
    _write_csv(reports_dir / "post_disaster_scene_quality_report.csv", POST_DISASTER_SCENE_QUALITY_COLUMNS, scene_quality_rows)
    _write_csv(contract_dir / "model_input_manifest.csv", MODEL_INPUT_V2_COLUMNS, model_rows)
    _write_csv(
        reports_dir / "split_leakage_report.csv",
        ["scope", "test_split_read", "selection_splits", "audited_rows", "model_input_rows"],
        [
            {
                "scope": "post_disaster_model_input_manifest",
                "test_split_read": "false",
                "selection_splits": "train;validation",
                "audited_rows": str(len(candidate_rows)),
                "model_input_rows": str(len(model_rows)),
            }
        ],
    )
    (reports_dir / "post_disaster_model_input_summary.md").write_text(
        _post_disaster_model_input_summary(
            required_channels=required,
            derived_indices=derived,
            audited_rows=len(candidate_rows),
            model_input_rows=len(model_rows),
            max_days_after_event=max_days_after_event,
            selection_strategy=selection_strategy,
        ),
        encoding="utf-8",
    )
    return PostDisasterModelInputResult(
        contract_dir=contract_dir,
        audited_rows=len(candidate_rows),
        model_input_rows=len(model_rows),
        required_channels=required,
    )


def _score_candidate_scenes(
    *,
    contract_dir: Path,
    row: dict[str, str],
    required_channels: tuple[str, ...],
    rasterio,
    resampling,
    numpy,
) -> dict[str, dict[str, str]]:
    try:
        scenes = json.loads(row.get("candidate_scenes", ""))
    except json.JSONDecodeError:
        return {}
    if not isinstance(scenes, list):
        return {}
    scores: dict[str, dict[str, str]] = {}
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        score = _score_scene_cloud_proxy(
            contract_dir=contract_dir,
            scene=scene,
            required_channels=required_channels,
            rasterio=rasterio,
            resampling=resampling,
            numpy=numpy,
        )
        for key in (str(scene.get("scene_id", "")), str(scene.get("image_date", ""))):
            if key:
                scores[key] = score
    return scores


def _scene_quality_report_rows(
    *,
    row: dict[str, str],
    required_channels: tuple[str, ...],
    max_days_after_event: int | None,
    selection: dict[str, str],
    scene_quality_scores: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    try:
        scenes = json.loads(row.get("candidate_scenes", ""))
    except json.JSONDecodeError:
        return []
    if not isinstance(scenes, list):
        return []
    rows: list[dict[str, str]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id", ""))
        image_date = str(scene.get("image_date", ""))
        days = str(scene.get("days_after_event", ""))
        band_paths = scene.get("band_paths", {})
        has_required = isinstance(band_paths, dict) and all(
            channel in band_paths and str(band_paths[channel]) for channel in required_channels
        )
        post_window_status = _scene_post_window_status(days, max_days_after_event)
        quality = scene_quality_scores.get(scene_id) or scene_quality_scores.get(image_date) or _quality_score(
            "1",
            "unscored",
            "no_quality_score_available",
        )
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "event_id": row.get("event_id", ""),
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "split": row.get("split", ""),
                "scene_id": scene_id,
                "image_date": image_date,
                "days_after_event": days,
                "required_channels": ";".join(required_channels),
                "has_required_channels": str(bool(has_required)).lower(),
                "post_window_status": post_window_status,
                "quality_score": quality["quality_score"],
                "quality_score_status": quality["quality_score_status"],
                "quality_score_detail": quality["quality_score_detail"],
                "selected": str(scene_id == selection.get("selected_scene_id", "")).lower(),
            }
        )
    return rows


def _scene_post_window_status(days_after_event: str, max_days_after_event: int | None) -> str:
    try:
        days = int(days_after_event)
    except (TypeError, ValueError):
        return "unknown"
    if days < 0:
        return "pre_event"
    if max_days_after_event is not None and days > max_days_after_event:
        return "outside_post_window"
    return "within_post_window"


def _score_scene_cloud_proxy(
    *,
    contract_dir: Path,
    scene: dict,
    required_channels: tuple[str, ...],
    rasterio,
    resampling,
    numpy,
) -> dict[str, str]:
    band_paths = scene.get("band_paths", {})
    if not isinstance(band_paths, dict):
        return _quality_score("1", "blocked_missing_band_paths", "missing_band_paths")
    channels = _quality_score_channels(required_channels, band_paths)
    if not channels:
        return _quality_score("1", "blocked_missing_quality_channels", "missing_quality_channels")
    arrays = []
    used_channels = []
    try:
        for channel in channels:
            path = _resolve_candidate_path(contract_dir, str(band_paths[channel]))
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                array = ds.read(1, out_shape=(64, 64), resampling=resampling).astype("float64")
                if ds.nodata is not None:
                    array[array == ds.nodata] = numpy.nan
            arrays.append(_to_reflectance(numpy, array))
            used_channels.append(channel)
    except Exception as exc:
        return _quality_score("1", "blocked_quality_read_failed", f"read_failed={exc}")
    if not arrays:
        return _quality_score("1", "blocked_no_readable_quality_channels", "no_readable_quality_channels")
    stack = numpy.stack(arrays, axis=0)
    valid = numpy.isfinite(stack).all(axis=0)
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if valid_fraction == 0.0:
        return _quality_score("1", "blocked_no_valid_quality_pixels", f"channels={','.join(used_channels)}")
    brightness = numpy.nanmean(stack, axis=0)
    whiteness = 1.0 - (numpy.nanstd(stack, axis=0) / (brightness + 1e-6))
    bright = (brightness > 0.28) & valid
    cloud_like = bright & (whiteness > 0.75)
    bright_fraction = float(bright.mean())
    cloud_proxy_fraction = float(cloud_like.mean())
    invalid_fraction = 1.0 - valid_fraction
    score = min(1.0, max(0.0, 0.65 * cloud_proxy_fraction + 0.25 * bright_fraction + 0.10 * invalid_fraction))
    detail = (
        f"channels={','.join(used_channels)},"
        f"cloud_proxy_fraction={cloud_proxy_fraction:.4f},"
        f"bright_fraction={bright_fraction:.4f},"
        f"valid_fraction={valid_fraction:.4f}"
    )
    return _quality_score(f"{score:.6f}", "scored", detail)


def _quality_score_channels(required_channels: tuple[str, ...], band_paths: dict) -> list[str]:
    preferred = ["F03", "F02", "F01", "F04"]
    channels = [channel for channel in preferred if channel in required_channels and channel in band_paths]
    for channel in required_channels:
        if channel in band_paths and channel not in channels:
            channels.append(channel)
    return channels[:3]


def _to_reflectance(numpy, array):
    valid = array[numpy.isfinite(array)]
    if valid.size == 0:
        return array
    scale = 10000.0 if float(numpy.nanpercentile(valid, 99)) > 2.0 else 1.0
    return numpy.clip(array / scale, 0.0, 1.0)


def _quality_score(score: str, status: str, detail: str) -> dict[str, str]:
    return {
        "quality_score": score,
        "quality_score_status": status,
        "quality_score_detail": detail,
    }


def _align_selected_channels_to_mask_grid(
    *,
    contract_dir: Path,
    model_inputs_dir: Path,
    row: dict[str, str],
    selection: dict[str, str],
    required_channels: tuple[str, ...],
    rasterio,
    reproject,
    resampling,
) -> tuple[dict[str, str], str]:
    mask_path = _resolve_contract_path(contract_dir, row.get("mask_path", ""))
    if not mask_path.exists():
        raise FileNotFoundError(f"missing mask {mask_path}")
    raw_paths = _parse_band_paths(selection.get("input_band_paths", ""))
    aligned_paths: dict[str, str] = {}
    with rasterio.open(mask_path) as reference_ds:
        reference_grid = _dataset_grid_summary(reference_ds)
        for channel in required_channels:
            source_path = raw_paths.get(channel)
            if source_path is None:
                raise FileNotFoundError(f"missing selected source for {channel}")
            if not source_path.exists():
                raise FileNotFoundError(f"missing selected source for {channel}: {source_path}")
            target_path = model_inputs_dir / f"{row.get('sample_id', 'sample')}_{channel}.tif"
            _write_band_on_reference_grid(
                source_path=source_path,
                target_path=target_path,
                reference_ds=reference_ds,
                rasterio=rasterio,
                reproject=reproject,
                resampling=resampling,
            )
            aligned_paths[channel] = str(target_path)
    return aligned_paths, reference_grid


def _write_band_on_reference_grid(
    *,
    source_path: Path,
    target_path: Path,
    reference_ds,
    rasterio,
    reproject,
    resampling,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as source_ds:
        profile = source_ds.profile.copy()
        profile.update(
            driver="GTiff",
            height=reference_ds.height,
            width=reference_ds.width,
            count=1,
            crs=reference_ds.crs,
            transform=reference_ds.transform,
            compress="deflate",
            tiled=False,
        )
        with rasterio.open(target_path, "w", **profile) as target_ds:
            reproject(
                source=rasterio.band(source_ds, 1),
                destination=rasterio.band(target_ds, 1),
                src_transform=source_ds.transform,
                src_crs=source_ds.crs,
                dst_transform=reference_ds.transform,
                dst_crs=reference_ds.crs,
                src_nodata=source_ds.nodata,
                dst_nodata=source_ds.nodata,
                resampling=resampling,
            )


def _derive_index_channels(
    *,
    aligned_paths: dict[str, str],
    model_inputs_dir: Path,
    row: dict[str, str],
    derived_indices: tuple[str, ...],
    rasterio,
    numpy,
) -> dict[str, str]:
    if not derived_indices:
        return {}
    arrays = {
        channel: _read_reflectance_channel(Path(path), rasterio=rasterio, numpy=numpy)
        for channel, path in aligned_paths.items()
    }
    profile_path = Path(next(iter(aligned_paths.values())))
    derived_paths: dict[str, str] = {}
    with rasterio.open(profile_path) as profile_ds:
        profile = profile_ds.profile.copy()
    profile.update(dtype="float32", count=1, nodata=None, compress="deflate", predictor=3, zlevel=9, tiled=False)
    for index_name in derived_indices:
        derived_array = _compute_derived_index(index_name, arrays, numpy=numpy).astype("float32")
        target_path = model_inputs_dir / f"{row.get('sample_id', 'sample')}_{index_name}.tif"
        with rasterio.open(target_path, "w", **profile) as dst:
            dst.write(derived_array, 1)
        derived_paths[index_name] = str(target_path)
    return derived_paths


def _read_reflectance_channel(path: Path, *, rasterio, numpy):
    with rasterio.open(path) as ds:
        array = ds.read(1).astype("float64")
        if ds.nodata is not None:
            array[array == ds.nodata] = numpy.nan
    return _to_reflectance(numpy, array)


def _compute_derived_index(index_name: str, arrays: dict[str, object], *, numpy):
    if index_name == "NDVI":
        return _safe_ratio(arrays["F07"] - arrays["F03"], arrays["F07"] + arrays["F03"], numpy=numpy)
    if index_name == "NBR":
        return _safe_ratio(arrays["F07"] - arrays["F12"], arrays["F07"] + arrays["F12"], numpy=numpy)
    if index_name == "NDMI":
        return _safe_ratio(arrays["F07"] - arrays["F11"], arrays["F07"] + arrays["F11"], numpy=numpy)
    if index_name == "NDWI":
        return _safe_ratio(arrays["F02"] - arrays["F07"], arrays["F02"] + arrays["F07"], numpy=numpy)
    if index_name == "MNDWI":
        return _safe_ratio(arrays["F02"] - arrays["F11"], arrays["F02"] + arrays["F11"], numpy=numpy)
    if index_name == "NBR2":
        return _safe_ratio(arrays["F11"] - arrays["F12"], arrays["F11"] + arrays["F12"], numpy=numpy)
    if index_name == "MIRBI":
        return (10.0 * arrays["F12"]) - (9.8 * arrays["F11"]) + 2.0
    if index_name == "BAIS2":
        red = arrays["F03"]
        red_edge_2 = arrays["F05"]
        red_edge_3 = arrays["F06"]
        narrow_nir = arrays["F08"]
        swir2 = arrays["F12"]
        spectral_product = _safe_divide(red_edge_2 * red_edge_3 * narrow_nir, red, numpy=numpy)
        spectral_product = numpy.where(spectral_product < 0, numpy.nan, spectral_product)
        first_term = 1.0 - numpy.sqrt(spectral_product)
        denominator = numpy.sqrt(swir2 + narrow_nir)
        second_term = _safe_divide(swir2 - narrow_nir, denominator, numpy=numpy) + 1.0
        return first_term * second_term
    if index_name == "BRIGHTNESS":
        return numpy.nanmean(numpy.stack([arrays["F03"], arrays["F02"], arrays["F01"]], axis=0), axis=0)
    raise ValueError(f"Unsupported derived index {index_name!r}")


def _safe_ratio(numerator, denominator, *, numpy):
    output = numpy.zeros_like(numerator, dtype="float64")
    valid = numpy.isfinite(numerator) & numpy.isfinite(denominator) & (numpy.abs(denominator) > 1e-8)
    output[valid] = numerator[valid] / denominator[valid]
    output[~valid] = numpy.nan
    return numpy.clip(output, -1.0, 1.0)


def _safe_divide(numerator, denominator, *, numpy):
    output = numpy.zeros_like(numerator, dtype="float64")
    valid = numpy.isfinite(numerator) & numpy.isfinite(denominator) & (numpy.abs(denominator) > 1e-8)
    output[valid] = numerator[valid] / denominator[valid]
    output[~valid] = numpy.nan
    return output


def _resolve_contract_path(contract_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else contract_dir / path


def _resolve_preview_path(contract_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return contract_dir / path


def _resolve_candidate_path(contract_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = contract_dir / path
    if candidate.exists():
        return candidate
    return path


def _dataset_grid_summary(dataset) -> str:
    transform = tuple(round(float(value), 12) for value in tuple(dataset.transform)[:6])
    return f"crs={dataset.crs} transform={transform} shape={dataset.height}x{dataset.width}"


def _parse_band_paths(text: str) -> dict[str, Path]:
    parsed = {}
    for item in text.split(";"):
        if ":" not in item:
            continue
        band, path = item.split(":", 1)
        parsed[band] = Path(path)
    return parsed


def _format_band_paths(band_paths: dict[str, str]) -> str:
    return ";".join(f"{band}:{band_paths[band]}" for band in band_paths)


def _post_disaster_model_input_summary(
    *,
    required_channels: tuple[str, ...],
    derived_indices: tuple[str, ...],
    audited_rows: int,
    model_input_rows: int,
    max_days_after_event: int | None,
    selection_strategy: str,
) -> str:
    window = "unbounded" if max_days_after_event is None else str(max_days_after_event)
    return f"""# Post-Disaster Model Input Summary

required_channels: {';'.join(required_channels)}
derived_indices: {';'.join(derived_indices) if derived_indices else 'none'}
selection_splits: train;validation
test_split_read: false
max_days_after_event: {window}
selection_strategy: {selection_strategy}
audited_rows: {audited_rows}
model_input_rows: {model_input_rows}
excluded_rows: {audited_rows - model_input_rows}

## Scope

`model_input_manifest.csv` contains post-disaster same-date channel stacks for train/validation model development.
This step does not read sealed test rows for selection and does not train a model.
"""


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


def _model_input_tile(
    Image,
    ImageDraw,
    numpy,
    arrays: dict[str, object],
    mask,
    row: dict[str, str],
    *,
    rgb_channels: tuple[str, str, str],
    false_color_channels: tuple[str, str, str],
):
    rgb = _blend_preview_mask(numpy, _compose_preview_rgb(numpy, arrays, rgb_channels), mask)
    false_color = _blend_preview_mask(numpy, _compose_preview_rgb(numpy, arrays, false_color_channels), mask)
    left = Image.fromarray(rgb, mode="RGB")
    right = Image.fromarray(false_color, mode="RGB")
    left.thumbnail((260, 260))
    right.thumbnail((260, 260))

    canvas = Image.new("RGB", (560, 348), "white")
    canvas.paste(left, ((260 - left.width) // 2 + 8, 26))
    canvas.paste(right, ((260 - right.width) // 2 + 292, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"RGB {'/'.join(rgb_channels)}", fill=(0, 0, 0))
    draw.text((292, 6), f"False {'/'.join(false_color_channels)}", fill=(0, 0, 0))
    draw.text((8, 292), f"{row.get('sample_id')} | {row.get('class_name')} | {row.get('split')}", fill=(0, 0, 0))
    draw.text(
        (8, 314),
        f"image: {row.get('selected_image_date', '')} | channels: {row.get('input_channels', '')}",
        fill=(80, 80, 80),
    )
    return canvas


def _compose_preview_rgb(numpy, arrays: dict[str, object], preferred_channels: tuple[str, str, str]):
    channels = [channel for channel in preferred_channels if channel in arrays]
    if len(channels) < 3:
        channels = [*channels, *[channel for channel in sorted(arrays) if channel not in channels]][:3]
    while len(channels) < 3:
        channels.append(channels[-1])
    planes = [_normalize_to_uint8(numpy, arrays[channel]) for channel in channels[:3]]
    return numpy.stack(planes, axis=-1)


def _blend_preview_mask(numpy, rgb, mask):
    blended = rgb.copy()
    foreground = mask > 0
    if foreground.any():
        overlay = numpy.array([255, 32, 32], dtype="float64")
        blended[foreground] = (0.62 * blended[foreground].astype("float64") + 0.38 * overlay).astype("uint8")
    return blended


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


def _wide_contact_sheet(Image, tiles):
    columns = 2
    tile_width, tile_height = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
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


def _model_input_visual_qa_summary(
    *,
    manifest_rows: int,
    preview_items: int,
    contact_sheet_path: Path,
    rgb_channels: tuple[str, ...],
    false_color_channels: tuple[str, ...],
) -> str:
    return f"""# Model Input Visual QA Summary

manifest_rows: {manifest_rows}
preview_items: {preview_items}
contact_sheet: {contact_sheet_path}
rgb_channels: {';'.join(rgb_channels)}
false_color_channels: {';'.join(false_color_channels)}

## Scope

These previews are rendered from `model_input_manifest.csv` / `input_band_paths`, not from the single reference raster used by the original WKT mask preview.
The current scene selection policy uses a transparent optical cloud-proxy score, then falls back to earlier post-disaster dates for ties.
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

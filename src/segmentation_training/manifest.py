from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "sample_id",
    "event_id",
    "class_id",
    "class_name",
    "split",
    "mask_path",
    "input_channels",
    "input_band_paths",
}

CLASS_SCOPES = {
    "binary_c2": {1},
    "binary_c5": {2},
    "multiclass_c2_c5": {1, 2},
    "c2_c5": {1, 2},
}


@dataclass(frozen=True)
class ModelInputRecord:
    sample_id: str
    event_id: str
    split: str
    class_id: int
    class_name: str
    mask_path: str
    input_channels: tuple[str, ...]
    input_band_paths: dict[str, str]
    row: dict[str, str]

    def with_paths(
        self,
        *,
        mask_path: str | None = None,
        input_band_paths: dict[str, str] | None = None,
        input_channels: tuple[str, ...] | None = None,
    ) -> "ModelInputRecord":
        row = dict(self.row)
        if mask_path is not None:
            row["mask_path"] = mask_path
        if input_band_paths is not None:
            row["input_band_paths"] = format_band_paths(input_band_paths)
        if input_channels is not None:
            row["input_channels"] = ";".join(input_channels)
        return ModelInputRecord(
            sample_id=row["sample_id"],
            event_id=row.get("event_id", ""),
            split=row["split"],
            class_id=int(row["class_id"]),
            class_name=row["class_name"],
            mask_path=row["mask_path"],
            input_channels=tuple(parse_channels(row["input_channels"])),
            input_band_paths=parse_band_paths(row["input_band_paths"]),
            row=row,
        )


def parse_channels(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", ";").split(";") if part.strip()]


def parse_band_paths(value: str) -> dict[str, str]:
    band_paths: dict[str, str] = {}
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid band path entry {part!r}; expected CHANNEL:path")
        channel, path = part.split(":", 1)
        channel = channel.strip()
        path = path.strip()
        if not channel or not path:
            raise ValueError(f"Invalid band path entry {part!r}; channel and path are required")
        band_paths[channel] = path
    return band_paths


def split_band_reference(value: str) -> tuple[str, int]:
    path, marker, suffix = value.rpartition("#band=")
    if not marker:
        return value, 1
    if not path.strip():
        raise ValueError(f"Invalid band reference {value!r}; path is required")
    try:
        band_index = int(suffix)
    except ValueError as exc:
        raise ValueError(f"Invalid band reference {value!r}; band must be an integer") from exc
    if band_index < 1:
        raise ValueError(f"Invalid band reference {value!r}; band must be >= 1")
    return path, band_index


def format_band_paths(band_paths: dict[str, str]) -> str:
    return ";".join(f"{channel}:{path}" for channel, path in band_paths.items())


def load_model_input_manifest(
    path: Path | str,
    *,
    required_channels: Iterable[str] = ("F16", "F17"),
) -> list[ModelInputRecord]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
        records = [_record_from_row(row, required_channels=tuple(required_channels)) for row in reader]
    return records


def _record_from_row(row: dict[str, str], *, required_channels: tuple[str, ...]) -> ModelInputRecord:
    channels = tuple(parse_channels(row["input_channels"]))
    band_paths = parse_band_paths(row["input_band_paths"])
    missing_channels = [channel for channel in required_channels if channel not in channels or channel not in band_paths]
    if missing_channels:
        sample_id = row.get("sample_id", "<unknown>")
        raise ValueError(f"{sample_id} missing required channels: {', '.join(missing_channels)}")
    return ModelInputRecord(
        sample_id=row["sample_id"],
        event_id=row.get("event_id", ""),
        split=row["split"],
        class_id=int(row["class_id"]),
        class_name=row["class_name"],
        mask_path=row["mask_path"],
        input_channels=channels,
        input_band_paths=band_paths,
        row=dict(row),
    )


def filter_records(
    records: Iterable[ModelInputRecord],
    *,
    split: str | None = None,
    class_scope: str = "multiclass_c2_c5",
) -> list[ModelInputRecord]:
    if class_scope not in CLASS_SCOPES:
        raise ValueError(f"Unsupported class_scope {class_scope!r}")
    allowed_classes = CLASS_SCOPES[class_scope]
    filtered = []
    for record in records:
        if split is not None and record.split != split:
            continue
        if record.class_id not in allowed_classes:
            continue
        filtered.append(record)
    return filtered


def class_split_counts(records: Iterable[ModelInputRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        class_key = str(record.class_id)
        counts.setdefault(record.split, {})
        counts[record.split][class_key] = counts[record.split].get(class_key, 0) + 1
    return counts


def resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def rebase_path(value: str, *, local_root: Path, target_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            relative = path.relative_to(local_root.resolve())
        except ValueError:
            relative = Path(*path.parts[1:])
    else:
        relative = path
    return str(target_root / relative)


def rebase_records(
    records: Iterable[ModelInputRecord],
    *,
    local_root: Path,
    target_root: Path,
    required_channels: Iterable[str] = ("F16", "F17"),
) -> list[ModelInputRecord]:
    rebased = []
    required = tuple(required_channels)
    for record in records:
        band_paths = {
            channel: rebase_path(record.input_band_paths[channel], local_root=local_root, target_root=target_root)
            for channel in required
        }
        rebased.append(
            record.with_paths(
                mask_path=rebase_path(record.mask_path, local_root=local_root, target_root=target_root),
                input_channels=required,
                input_band_paths=band_paths,
            )
        )
    return rebased


def validate_record_paths(
    records: Iterable[ModelInputRecord],
    *,
    contract_dir: Path,
    require_bands: bool = True,
) -> list[str]:
    errors: list[str] = []
    for record in records:
        mask_path = resolve_existing_data_path(contract_dir, record.mask_path, subdir="masks")
        if not mask_path.exists():
            errors.append(f"{record.sample_id}: missing mask {mask_path}")
        if require_bands:
            for channel, path in record.input_band_paths.items():
                band_path = resolve_existing_data_path(contract_dir, split_band_reference(path)[0])
                if not band_path.exists():
                    errors.append(f"{record.sample_id}: missing {channel} raster {band_path}")
    return errors


def validate_record_raster_grids(
    records: Iterable[ModelInputRecord],
    *,
    contract_dir: Path,
    required_channels: Iterable[str] = ("F16", "F17"),
) -> list[str]:
    import rasterio

    errors: list[str] = []
    required = tuple(required_channels)
    for record in records:
        mask_path = resolve_existing_data_path(contract_dir, record.mask_path, subdir="masks")
        if not mask_path.exists():
            errors.append(f"{record.sample_id}: missing mask {mask_path}")
            continue
        band_paths = []
        missing = []
        for channel in required:
            value = record.input_band_paths.get(channel)
            if value is None:
                missing.append(channel)
                continue
            band_value, band_index = split_band_reference(value)
            band_path = resolve_existing_data_path(contract_dir, band_value)
            if not band_path.exists():
                missing.append(channel)
                continue
            band_paths.append((channel, band_path, band_index))
        if missing:
            errors.append(f"{record.sample_id}: missing required channels for grid validation: {', '.join(missing)}")
            continue
        try:
            with rasterio.open(mask_path) as mask_ds:
                mask_grid = _raster_grid(mask_ds)
            band_grids = []
            for channel, band_path, band_index in band_paths:
                with rasterio.open(band_path) as band_ds:
                    if band_index > band_ds.count:
                        raise ValueError(f"{channel} requests band {band_index}, but raster has {band_ds.count} band(s)")
                    band_grids.append((channel, band_path, _raster_grid(band_ds)))
        except Exception as exc:
            errors.append(f"{record.sample_id}: raster grid validation failed: {exc}")
            continue
        reference_channel, reference_path, reference_grid = band_grids[0]
        mismatched_channels = [
            f"{channel}={_format_grid(grid)}"
            for channel, _path, grid in band_grids[1:]
            if grid != reference_grid
        ]
        if mismatched_channels:
            errors.append(
                f"{record.sample_id}: input channel grid mismatch against {reference_channel} "
                f"{reference_path}: {'; '.join(mismatched_channels)}"
            )
            continue
        if mask_grid != reference_grid:
            errors.append(
                f"{record.sample_id}: mask grid mismatch against {reference_channel} {reference_path}: "
                f"mask={_format_grid(mask_grid)} {mask_path}; {reference_channel}={_format_grid(reference_grid)}"
            )
    return errors


def resolve_existing_data_path(root: Path, value: str, *, subdir: str | None = None) -> Path:
    direct = resolve_under_root(root, value)
    if direct.exists():
        return direct
    cwd_path = resolve_under_root(Path.cwd(), value)
    if cwd_path.exists():
        return cwd_path
    if subdir is not None:
        local_mirror = root / subdir / Path(value).name
        if local_mirror.exists():
            return local_mirror
    return direct


def _raster_grid(dataset) -> tuple[str, tuple[float, ...], int, int]:
    return (
        str(dataset.crs),
        tuple(round(float(value), 12) for value in tuple(dataset.transform)[:6]),
        int(dataset.height),
        int(dataset.width),
    )


def _format_grid(grid: tuple[str, tuple[float, ...], int, int]) -> str:
    crs, transform, height, width = grid
    return f"crs={crs} transform={transform} shape={height}x{width}"


def write_manifest(path: Path, records: list[ModelInputRecord]) -> None:
    if not records:
        raise ValueError("Cannot write an empty manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].row.keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.row)

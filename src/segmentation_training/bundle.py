from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .manifest import (
    ModelInputRecord,
    class_split_counts,
    filter_records,
    load_model_input_manifest,
    rebase_path,
    resolve_under_root,
    split_band_reference,
    write_manifest,
)


COMMON_C2_CHANNEL_MAP = {
    "BLUE": "B",
    "GREEN": "G",
    "RED": "R",
    "NIR": "NIR",
}

COMMON_C5_CHANNEL_MAP = {
    "BLUE": "F01",
    "GREEN": "F02",
    "RED": "F03",
    "NIR": "F07",
}

COMMON_OPTICAL_CHANNELS = ("BLUE", "GREEN", "RED", "NIR")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_exists(source: Path, target: Path) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def build_training_bundle(
    *,
    contract_dir: Path,
    bundle_dir: Path,
    required_channels: tuple[str, ...] = ("F16", "F17"),
    bundle_version: str = "0.1",
    mode: str = "manifest-only",
    class_scope: str = "multiclass_c2_c5",
    cloud_data_root: Path = Path("/data/training_bundle_v0_1"),
) -> dict:
    if mode != "manifest-only":
        raise ValueError("v0.1 only supports mode='manifest-only'")

    contract_dir = contract_dir.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = bundle_dir / "metadata"
    manifests_dir = bundle_dir / "manifests"
    masks_dir = bundle_dir / "masks"
    model_inputs_dir = bundle_dir / "model_inputs"
    configs_dir = bundle_dir / "configs"
    reports_dir = metadata_dir / "source_reports"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    model_inputs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for name in ("class_map.json", "channels.json"):
        copy_if_exists(contract_dir / name, metadata_dir / name)

    source_reports = contract_dir / "reports"
    if source_reports.exists():
        for report_path in sorted(source_reports.iterdir()):
            if report_path.is_file():
                copy_if_exists(report_path, reports_dir / report_path.name)
    source_configs = Path.cwd() / "configs" / "segmentation_training"
    if source_configs.exists():
        for config_path in sorted(source_configs.glob("*.yaml")):
            copy_if_exists(config_path, configs_dir / config_path.name)

    records = load_model_input_manifest(contract_dir / "model_input_manifest.csv", required_channels=required_channels)
    scoped_records = filter_records(records, class_scope=class_scope)
    project_root = _infer_project_root(contract_dir)

    copied_masks = 0
    for record in scoped_records:
        source_mask = resolve_under_root(contract_dir, record.mask_path)
        if not source_mask.exists():
            raise FileNotFoundError(f"Missing mask for {record.sample_id}: {source_mask}")
        target_mask = masks_dir / Path(record.mask_path).name
        copy_if_exists(source_mask, target_mask)
        copied_masks += 1

    rebased_records = []
    for record in scoped_records:
        band_paths = {}
        for channel in required_channels:
            band_paths[channel] = _bundle_band_path(
                record.input_band_paths[channel],
                contract_dir=contract_dir,
                bundle_dir=bundle_dir,
                project_root=project_root,
                cloud_data_root=cloud_data_root,
            )
        rebased_records.append(
            record.with_paths(
                mask_path=str(cloud_data_root / "masks" / Path(record.mask_path).name),
                input_channels=required_channels,
                input_band_paths=band_paths,
            )
        )
    cloud_manifest_path = manifests_dir / "cloud_model_input_manifest.csv"
    write_manifest(cloud_manifest_path, rebased_records)

    included_files = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "bundle_manifest.json":
            included_files.append(
                {
                    "path": str(path.relative_to(bundle_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    bundle_manifest = {
        "bundle_version": bundle_version,
        "mode": mode,
        "contract_dir": str(contract_dir),
        "cloud_data_root": str(cloud_data_root),
        "required_channels": list(required_channels),
        "class_scope": class_scope,
        "record_count": len(scoped_records),
        "copied_mask_count": copied_masks,
        "split_class_counts": class_split_counts(scoped_records),
        "files": included_files,
    }
    (metadata_dir / "bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (bundle_dir / "README.md").write_text(_bundle_readme(bundle_manifest), encoding="utf-8")
    return bundle_manifest


def build_common_channel_bundle(
    *,
    c2_bundle_dir: Path,
    c5_bundle_dir: Path,
    output_bundle_dir: Path,
    bundle_version: str = "p15-c2-c5-common4",
) -> dict:
    c2_records = load_model_input_manifest(
        c2_bundle_dir / "manifests" / "cloud_model_input_manifest.csv",
        required_channels=tuple(COMMON_C2_CHANNEL_MAP.values()),
    )
    c5_records = load_model_input_manifest(
        c5_bundle_dir / "manifests" / "cloud_model_input_manifest.csv",
        required_channels=tuple(COMMON_C5_CHANNEL_MAP.values()),
    )
    selected_records = [
        *_remap_common_records(filter_records(c2_records, class_scope="binary_c2"), COMMON_C2_CHANNEL_MAP),
        *_remap_common_records(filter_records(c5_records, class_scope="binary_c5"), COMMON_C5_CHANNEL_MAP),
    ]

    output_bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_bundle_dir / "metadata"
    manifests_dir = output_bundle_dir / "manifests"
    metadata_dir.mkdir(exist_ok=True)
    manifests_dir.mkdir(exist_ok=True)

    write_manifest(manifests_dir / "cloud_model_input_manifest.csv", selected_records)
    class_map = {
        "0": {"name": "background", "trainable": False},
        "1": {"name": "C2_debris_flow", "trainable": True},
        "2": {"name": "C5_fire", "trainable": True},
    }
    (metadata_dir / "class_map.json").write_text(
        json.dumps(class_map, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    bundle_manifest = {
        "bundle_version": bundle_version,
        "mode": "common-channel-merge",
        "source_bundles": {
            "c2": str(c2_bundle_dir),
            "c5": str(c5_bundle_dir),
        },
        "required_channels": list(COMMON_OPTICAL_CHANNELS),
        "class_scope": "multiclass_c2_c5",
        "record_count": len(selected_records),
        "split_class_counts": class_split_counts(selected_records),
    }
    (metadata_dir / "bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_bundle_dir / "README.md").write_text(_common_bundle_readme(bundle_manifest), encoding="utf-8")
    return bundle_manifest


def _remap_common_records(records, channel_map: dict[str, str]):
    remapped = []
    required = tuple(COMMON_OPTICAL_CHANNELS)
    for record in records:
        band_paths = {logical: record.input_band_paths[source] for logical, source in channel_map.items()}
        remapped.append(_standard_manifest_record(record.with_paths(input_channels=required, input_band_paths=band_paths)))
    return remapped


def _standard_manifest_record(record: ModelInputRecord) -> ModelInputRecord:
    row = {
        "sample_id": record.sample_id,
        "event_id": record.event_id,
        "class_id": str(record.class_id),
        "class_name": record.class_name,
        "split": record.split,
        "mask_path": record.mask_path,
        "input_channels": ";".join(record.input_channels),
        "input_band_paths": ";".join(f"{channel}:{record.input_band_paths[channel]}" for channel in record.input_channels),
    }
    return ModelInputRecord(
        sample_id=row["sample_id"],
        event_id=row["event_id"],
        split=row["split"],
        class_id=int(row["class_id"]),
        class_name=row["class_name"],
        mask_path=row["mask_path"],
        input_channels=record.input_channels,
        input_band_paths={channel: record.input_band_paths[channel] for channel in record.input_channels},
        row=row,
    )


def _infer_project_root(contract_dir: Path) -> Path:
    for parent in [contract_dir, *contract_dir.parents]:
        if parent.name == ".agent-team":
            return parent.parent
    return Path.cwd()


def _bundle_band_path(
    value: str,
    *,
    contract_dir: Path,
    bundle_dir: Path,
    project_root: Path,
    cloud_data_root: Path,
) -> str:
    physical_value, band_index = split_band_reference(value)
    suffix = "" if band_index == 1 and "#band=" not in value else f"#band={band_index}"
    value_path = Path(physical_value)
    if value_path.is_absolute():
        source_path = value_path
    elif (contract_dir / value_path).exists():
        source_path = contract_dir / value_path
    elif value_path.exists():
        source_path = value_path.resolve()
    else:
        source_path = contract_dir / value_path
    try:
        relative_to_contract = source_path.resolve().relative_to(contract_dir.resolve())
    except ValueError:
        return rebase_path(value, local_root=project_root, target_root=cloud_data_root)
    if relative_to_contract.parts and relative_to_contract.parts[0] == "model_inputs":
        target_path = bundle_dir / relative_to_contract
        copy_if_exists(source_path, target_path)
        return str(relative_to_contract) + suffix
    return rebase_path(value, local_root=project_root, target_root=cloud_data_root)


def _bundle_readme(bundle_manifest: dict) -> str:
    return "\n".join(
        [
            f"# Segmentation Training Bundle v{bundle_manifest['bundle_version']}",
            "",
            "This is a cloud-first manifest-only bundle. Masks and metadata are packaged here;",
            "post-disaster aligned model inputs are packaged when present; large raw raster bands",
            "are expected to be mounted on the cloud server under the",
            f"configured cloud data root: `{bundle_manifest['cloud_data_root']}`.",
            "",
            "Run `python -m segmentation_training validate-manifest` and `dry-run` on the",
            "cloud server before launching training.",
            "",
        ]
    )


def _common_bundle_readme(bundle_manifest: dict) -> str:
    return "\n".join(
        [
            f"# Common C2/C5 Optical Bundle v{bundle_manifest['bundle_version']}",
            "",
            "This bundle merges existing C2 landslide and C5 fire cloud manifests into",
            "`BLUE/GREEN/RED/NIR` logical channels for dataset-scale multiclass probing.",
            "It references the source bundle cloud paths directly and does not copy raw rasters.",
            "",
        ]
    )

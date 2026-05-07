from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .manifest import (
    class_split_counts,
    filter_records,
    load_model_input_manifest,
    rebase_path,
    resolve_under_root,
    write_manifest,
)


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
    value_path = Path(value)
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
        return str(relative_to_contract)
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

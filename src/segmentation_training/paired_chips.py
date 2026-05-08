from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


LANDSLIDE_9CH_CHANNELS = (
    "B",
    "G",
    "R",
    "NIR",
    "ELEVATION",
    "SLOPE",
    "ASPECT",
    "CURVATURE",
    "TPI",
)

MODEL_INPUT_COLUMNS = [
    "sample_id",
    "event_id",
    "class_id",
    "class_name",
    "split",
    "mask_path",
    "input_channels",
    "input_band_paths",
    "source_image_path",
    "source_mask_path",
    "qa_flags",
]


@dataclass(frozen=True)
class PairedChipContractResult:
    output_dir: Path
    paired_rows: int
    unmatched_images: int
    unmatched_masks: int
    train_rows: int
    validation_rows: int


def build_paired_chip_contract(
    *,
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    channels: tuple[str, ...] = LANDSLIDE_9CH_CHANNELS,
    class_id: int = 1,
    class_name: str = "C2_debris_flow",
    train_fraction: float = 0.8,
) -> PairedChipContractResult:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = {_sample_key(path, suffix=".tif"): path for path in image_dir.glob("*.tif")}
    masks = {_sample_key(path, suffix="_mask.tif"): path for path in mask_dir.glob("*_mask.tif")}
    paired_keys = sorted(set(images) & set(masks), key=_sample_sort_key)

    train_cutoff = max(1, int(len(paired_keys) * train_fraction)) if paired_keys else 0
    rows = []
    for index, key in enumerate(paired_keys):
        image_path = images[key].resolve()
        mask_path = masks[key].resolve()
        split = "train" if index < train_cutoff else "validation"
        rows.append(
            {
                "sample_id": f"landslide-{key}",
                "event_id": f"landslide_chip_{key}",
                "class_id": str(class_id),
                "class_name": class_name,
                "split": split,
                "mask_path": str(mask_path),
                "input_channels": ";".join(channels),
                "input_band_paths": ";".join(
                    f"{channel}:{image_path}#band={band_index}"
                    for band_index, channel in enumerate(channels, start=1)
                ),
                "source_image_path": str(image_path),
                "source_mask_path": str(mask_path),
                "qa_flags": "paired_multiband_chip",
            }
        )

    _write_csv(output_dir / "model_input_manifest.csv", MODEL_INPUT_COLUMNS, rows)
    _write_json(output_dir / "class_map.json", _class_map(class_id=class_id, class_name=class_name))
    _write_json(output_dir / "channels.json", _channels(channels))
    (output_dir / "qa_summary.md").write_text(
        _qa_summary(
            paired_rows=len(rows),
            unmatched_images=len(set(images) - set(masks)),
            unmatched_masks=len(set(masks) - set(images)),
            train_rows=sum(1 for row in rows if row["split"] == "train"),
            validation_rows=sum(1 for row in rows if row["split"] == "validation"),
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(), encoding="utf-8")

    return PairedChipContractResult(
        output_dir=output_dir,
        paired_rows=len(rows),
        unmatched_images=len(set(images) - set(masks)),
        unmatched_masks=len(set(masks) - set(images)),
        train_rows=sum(1 for row in rows if row["split"] == "train"),
        validation_rows=sum(1 for row in rows if row["split"] == "validation"),
    )


def _sample_key(path: Path, *, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        return path.stem
    return name[: -len(suffix)]


def _sample_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    if match:
        return int(match.group(1)), value
    return 10**9, value


def _class_map(*, class_id: int, class_name: str) -> dict[str, dict[str, object]]:
    return {
        "0": {"name": "background", "trainable": True, "ignore": False},
        str(class_id): {"name": class_name, "trainable": True, "ignore": False},
        "255": {"name": "ignore", "trainable": False, "ignore": True},
    }


def _channels(channels: tuple[str, ...]) -> dict[str, object]:
    return {
        "version": "paired-chip-v0.1",
        "channels": [
            {
                "index": index,
                "name": channel,
                "source": "paired_multiband_chip",
                "required": True,
                "derived": False,
                "source_bands": [channel],
                "normalization_policy": "runtime_percentile_2_98",
            }
            for index, channel in enumerate(channels)
        ],
    }


def _qa_summary(
    *,
    paired_rows: int,
    unmatched_images: int,
    unmatched_masks: int,
    train_rows: int,
    validation_rows: int,
) -> str:
    return "\n".join(
        [
            "# Paired Chip QA Summary",
            "",
            f"paired_rows: {paired_rows}",
            f"unmatched_images: {unmatched_images}",
            f"unmatched_masks: {unmatched_masks}",
            f"train_rows: {train_rows}",
            f"validation_rows: {validation_rows}",
            "",
            "## Channel Mapping",
            "- B, G, R, NIR, ELEVATION, SLOPE, ASPECT, CURVATURE, TPI",
            "",
        ]
    )


def _readme() -> str:
    return "\n".join(
        [
            "# Paired Landslide Chip Contract",
            "",
            "This contract pairs multiband landslide chips with raster masks for binary C2 training.",
            "",
        ]
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

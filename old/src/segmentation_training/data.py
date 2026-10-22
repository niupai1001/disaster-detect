from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import ModelInputRecord


def read_raster_array(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.read(1)


def read_record_arrays(record: ModelInputRecord, *, root: Path, channels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    band_arrays = []
    for channel in channels:
        path = Path(record.input_band_paths[channel])
        if not path.is_absolute():
            path = root / path
        band_arrays.append(read_raster_array(path))
    mask_path = Path(record.mask_path)
    if not mask_path.is_absolute():
        mask_path = root / mask_path
    mask = read_raster_array(mask_path)
    return np.stack(band_arrays, axis=0), mask


def to_torch_tensors(channels: np.ndarray, mask: np.ndarray):
    import torch

    return torch.from_numpy(channels.astype("float32")), torch.from_numpy(mask.astype("int64"))


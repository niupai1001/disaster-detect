from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 220, 220),
    2: (220, 32, 32),
    255: (128, 128, 128),
}


def write_prediction_preview(
    *,
    channels: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    title: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_image = _render_channels(channels)
    label_image = _render_mask(label)
    prediction_image = _render_mask(prediction)
    overlay_image = _render_error_overlay(label, prediction)
    panels = [channel_image, label_image, prediction_image, overlay_image]
    width = sum(panel.width for panel in panels)
    title_height = 24 if title else 0
    canvas = Image.new("RGB", (width, panels[0].height + title_height), (255, 255, 255))
    if title:
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), title, fill=(0, 0, 0))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_height))
        x += panel.width
    canvas.save(output_path)
    return output_path


def write_contact_sheet(image_paths: list[Path], output_path: Path, *, columns: int = 2) -> Path:
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (255, 255, 255))
    for idx, image in enumerate(images):
        x = (idx % columns) * tile_w
        y = (idx // columns) * tile_h
        sheet.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _render_channels(channels: np.ndarray) -> Image.Image:
    array = np.asarray(channels)
    if array.ndim == 3:
        if array.shape[0] <= 4:
            array = array[0]
        else:
            array = array[:, :, 0]
    normalized = _normalize(array)
    return Image.fromarray(normalized, mode="L").convert("RGB")


def _render_mask(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color
    return Image.fromarray(rgb, mode="RGB")


def _render_error_overlay(label: np.ndarray, prediction: np.ndarray) -> Image.Image:
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    valid = label != 255
    false_positive = np.logical_and(valid, prediction != 0) & (label != prediction)
    false_negative = np.logical_and(valid, label != 0) & (prediction != label)
    correct_foreground = np.logical_and(valid, label == prediction) & (label != 0)
    rgb[correct_foreground] = (80, 180, 80)
    rgb[false_positive] = (255, 220, 0)
    rgb[false_negative] = (255, 0, 255)
    rgb[label == 255] = CLASS_COLORS[255]
    return Image.fromarray(rgb, mode="RGB")


def _normalize(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(arr[finite], 2))
    hi = float(np.percentile(arr[finite], 98))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    row: int
    col: int
    height: int
    width: int


def deterministic_windows(height: int, width: int, *, size: int = 256, max_windows: int = 8) -> list[Window]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    win_h = min(size, height)
    win_w = min(size, width)
    rows = sorted({0, max(0, (height - win_h) // 2), max(0, height - win_h)})
    cols = sorted({0, max(0, (width - win_w) // 2), max(0, width - win_w)})
    windows = [Window(row, col, win_h, win_w) for row in rows for col in cols]
    return windows[:max_windows]


def choose_training_window(
    mask: np.ndarray,
    *,
    size: int,
    rng,
    foreground_class_ids: list[int],
    foreground_probability: float = 0.0,
    min_foreground_pixels: int = 1,
    max_attempts: int = 16,
) -> Window:
    height, width = mask.shape
    win_h, win_w = _window_shape(height, width, size)
    if bool(rng.random() < foreground_probability):
        foreground = np.argwhere(np.isin(mask, foreground_class_ids))
        if foreground.size:
            row_idx, col_idx = foreground[int(rng.integers(0, len(foreground)))]
            row = _clamp(int(row_idx) - win_h // 2, 0, max(0, height - win_h))
            col = _clamp(int(col_idx) - win_w // 2, 0, max(0, width - win_w))
            return Window(row, col, win_h, win_w)
    best = _random_window(height, width, win_h, win_w, rng)
    best_pixels = window_foreground_stats(mask, best, foreground_class_ids=foreground_class_ids)["foreground_pixels"]
    for _ in range(max_attempts):
        candidate = _random_window(height, width, win_h, win_w, rng)
        pixels = window_foreground_stats(mask, candidate, foreground_class_ids=foreground_class_ids)["foreground_pixels"]
        if pixels >= min_foreground_pixels:
            return candidate
        if pixels > best_pixels:
            best = candidate
            best_pixels = pixels
    return best


def choose_validation_window(
    mask: np.ndarray,
    *,
    size: int,
    foreground_class_ids: list[int],
    policy: str = "center",
) -> Window:
    height, width = mask.shape
    win_h, win_w = _window_shape(height, width, size)
    if policy == "foreground_center":
        foreground = np.argwhere(np.isin(mask, foreground_class_ids))
        if foreground.size:
            row_idx, col_idx = foreground[len(foreground) // 2]
            return Window(
                _clamp(int(row_idx) - win_h // 2, 0, max(0, height - win_h)),
                _clamp(int(col_idx) - win_w // 2, 0, max(0, width - win_w)),
                win_h,
                win_w,
            )
    if policy not in {"center", "foreground_center"}:
        raise ValueError(f"Unsupported validation window policy {policy!r}")
    return Window(max(0, (height - win_h) // 2), max(0, (width - win_w) // 2), win_h, win_w)


def window_foreground_stats(
    mask: np.ndarray,
    window: Window,
    *,
    foreground_class_ids: list[int],
    ignore_index: int = 255,
) -> dict:
    crop = mask[window.row : window.row + window.height, window.col : window.col + window.width]
    valid = crop != ignore_index
    foreground = np.logical_and(valid, np.isin(crop, foreground_class_ids))
    valid_pixels = int(valid.sum())
    foreground_pixels = int(foreground.sum())
    return {
        "window_row": window.row,
        "window_col": window.col,
        "window_height": window.height,
        "window_width": window.width,
        "valid_pixels": valid_pixels,
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": float(foreground_pixels / valid_pixels) if valid_pixels else 0.0,
    }


def foreground_window_coverage(
    samples: list[tuple[str, np.ndarray]],
    *,
    foreground_class_ids: list[int],
    size: int,
    ignore_index: int = 255,
) -> list[dict]:
    rows = []
    for sample_id, mask in samples:
        full_valid = mask != ignore_index
        full_foreground = np.logical_and(full_valid, np.isin(mask, foreground_class_ids))
        center = choose_validation_window(mask, size=size, foreground_class_ids=foreground_class_ids, policy="center")
        foreground_center = choose_validation_window(
            mask,
            size=size,
            foreground_class_ids=foreground_class_ids,
            policy="foreground_center",
        )
        center_stats = window_foreground_stats(
            mask,
            center,
            foreground_class_ids=foreground_class_ids,
            ignore_index=ignore_index,
        )
        fg_center_stats = window_foreground_stats(
            mask,
            foreground_center,
            foreground_class_ids=foreground_class_ids,
            ignore_index=ignore_index,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "full_mask_foreground_pixels": int(full_foreground.sum()),
                "full_mask_valid_pixels": int(full_valid.sum()),
                "full_mask_foreground_fraction": float(full_foreground.sum() / full_valid.sum())
                if full_valid.sum()
                else 0.0,
                "center_window_pixels": center_stats["foreground_pixels"],
                "foreground_center_window_pixels": fg_center_stats["foreground_pixels"],
            }
        )
    return rows


def _window_shape(height: int, width: int, size: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return min(size, height), min(size, width)


def _random_window(height: int, width: int, win_h: int, win_w: int, rng) -> Window:
    return Window(
        int(rng.integers(0, max(1, height - win_h + 1))),
        int(rng.integers(0, max(1, width - win_w + 1))),
        win_h,
        win_w,
    )


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))

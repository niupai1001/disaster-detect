from __future__ import annotations

from dataclasses import dataclass


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


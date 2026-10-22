from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


BATCH_RE = re.compile(
    r"epoch (?P<epoch>\d+)/(?P<max_epochs>\d+) batch (?P<batch>\d+)/(?P<total_batches>\d+) "
    r"\((?P<pct>[0-9.]+)%\) loss=(?P<loss>[0-9.eE+-]+) avg_loss=(?P<avg_loss>[0-9.eE+-]+)"
)
VALIDATION_RE = re.compile(
    r"epoch (?P<epoch>\d+)/(?P<max_epochs>\d+) validation loss=(?P<loss>[0-9.eE+-]+) "
    r"mean_iou=(?P<mean_iou>[0-9.eE+-]+) foreground_recall=(?P<foreground_recall>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class TrainingProgress:
    batch_rows: list[dict]
    validation_rows: list[dict]


def parse_training_progress(text: str) -> TrainingProgress:
    batch_rows = []
    validation_rows = []
    step = 0
    for line in text.splitlines():
        batch_match = BATCH_RE.search(line)
        if batch_match:
            step += 1
            row = _convert_numbers(batch_match.groupdict())
            row["step"] = step
            batch_rows.append(row)
            continue
        validation_match = VALIDATION_RE.search(line)
        if validation_match:
            row = _convert_numbers(validation_match.groupdict())
            row["step"] = step
            validation_rows.append(row)
    return TrainingProgress(batch_rows=batch_rows, validation_rows=validation_rows)


def write_training_curves(run_dir: Path, *, output_png: Path | None = None, output_csv: Path | None = None) -> dict:
    log_path = run_dir / "logs" / "training_progress.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Training progress log not found: {log_path}")
    progress = parse_training_progress(log_path.read_text(encoding="utf-8"))
    if not progress.batch_rows and not progress.validation_rows:
        raise ValueError(f"No training curve points found in {log_path}")
    output_png = output_png or run_dir / "training_curves.png"
    output_csv = output_csv or run_dir / "training_curves.csv"
    _write_csv(output_csv, progress)
    _write_png(output_png, progress)
    return {
        "log": str(log_path),
        "png": str(output_png),
        "csv": str(output_csv),
        "batch_points": len(progress.batch_rows),
        "validation_points": len(progress.validation_rows),
    }


def _convert_numbers(row: dict[str, str]) -> dict:
    converted = {}
    for key, value in row.items():
        converted[key] = float(value) if "." in value or "e" in value.lower() else int(value)
    return converted


def _write_csv(path: Path, progress: TrainingProgress) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["kind", "step", "epoch", "batch", "loss", "avg_loss", "mean_iou", "foreground_recall"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in progress.batch_rows:
            writer.writerow(
                {
                    "kind": "batch",
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "batch": row["batch"],
                    "loss": row["loss"],
                    "avg_loss": row["avg_loss"],
                }
            )
        for row in progress.validation_rows:
            writer.writerow(
                {
                    "kind": "validation",
                    "step": row["step"],
                    "epoch": row["epoch"],
                    "loss": row["loss"],
                    "mean_iou": row["mean_iou"],
                    "foreground_recall": row["foreground_recall"],
                }
            )


def _write_png(path: Path, progress: TrainingProgress) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 760
    margin_l, margin_r, margin_t, margin_b = 80, 40, 60, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 20), "Segmentation Training Curves", fill=(0, 0, 0))
    draw.rectangle((margin_l, margin_t, width - margin_r, height - margin_b), outline=(190, 190, 190))
    for i in range(6):
        y = margin_t + i * plot_h / 5
        draw.line((margin_l, y, width - margin_r, y), fill=(235, 235, 235))
        label = f"{1 - i / 5:.1f}"
        draw.text((20, y - 7), label, fill=(90, 90, 90))
    max_step = max(
        [row["step"] for row in progress.batch_rows + progress.validation_rows] or [1]
    )

    def xy(step: float, value: float) -> tuple[float, float]:
        x = margin_l + (step / max(1, max_step)) * plot_w
        y = margin_t + (1 - max(0.0, min(1.0, value))) * plot_h
        return x, y

    batch_points = [xy(row["step"], row["avg_loss"]) for row in progress.batch_rows]
    _draw_polyline(draw, batch_points, fill=(40, 105, 180), width=3)
    validation_loss = [xy(row["step"], row["loss"]) for row in progress.validation_rows]
    _draw_polyline(draw, validation_loss, fill=(220, 120, 20), width=3)
    miou = [xy(row["step"], row["mean_iou"]) for row in progress.validation_rows]
    _draw_polyline(draw, miou, fill=(40, 150, 80), width=3)
    recall = [xy(row["step"], row["foreground_recall"]) for row in progress.validation_rows]
    _draw_polyline(draw, recall, fill=(180, 40, 130), width=3)
    legend_x = margin_l
    legend_y = height - 55
    for label, color in [
        ("train avg_loss", (40, 105, 180)),
        ("validation loss", (220, 120, 20)),
        ("mean_iou", (40, 150, 80)),
        ("foreground_recall", (180, 40, 130)),
    ]:
        draw.line((legend_x, legend_y + 8, legend_x + 28, legend_y + 8), fill=color, width=4)
        draw.text((legend_x + 36, legend_y), label, fill=(0, 0, 0))
        legend_x += 230
    image.save(path)


def _draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], *, fill, width: int) -> None:
    if len(points) < 2:
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=fill)
        return
    draw.line(points, fill=fill, width=width)

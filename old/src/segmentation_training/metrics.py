from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassMetric:
    class_id: int
    tp: int
    fp: int
    fn: int
    iou: float
    dice: float
    precision: float
    recall: float


def compute_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    class_ids: list[int],
    ignore_index: int = 255,
) -> np.ndarray:
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    class_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    valid = labels != ignore_index
    for label, prediction in zip(labels[valid].ravel(), predictions[valid].ravel()):
        if int(label) in class_to_index and int(prediction) in class_to_index:
            matrix[class_to_index[int(label)], class_to_index[int(prediction)]] += 1
    return matrix


def encode_label(labels: np.ndarray, source_class_ids: list[int], *, ignore_index: int = 255) -> np.ndarray:
    encoded = np.full(labels.shape, ignore_index, dtype=np.uint8)
    encoded[labels == ignore_index] = ignore_index
    for encoded_id, source_class_id in enumerate(source_class_ids):
        encoded[labels == source_class_id] = encoded_id
    return encoded


def decode_prediction(
    encoded_predictions: np.ndarray,
    source_class_ids: list[int],
    *,
    ignore_index: int = 255,
) -> np.ndarray:
    decoded = np.full(encoded_predictions.shape, ignore_index, dtype=np.uint8)
    for encoded_id, source_class_id in enumerate(source_class_ids):
        decoded[encoded_predictions == encoded_id] = source_class_id
    return decoded


def per_class_metrics(confusion_matrix: np.ndarray, *, class_ids: list[int]) -> list[ClassMetric]:
    metrics = []
    for idx, class_id in enumerate(class_ids):
        tp = int(confusion_matrix[idx, idx])
        fp = int(confusion_matrix[:, idx].sum() - tp)
        fn = int(confusion_matrix[idx, :].sum() - tp)
        iou = _safe_div(tp, tp + fp + fn)
        dice = _safe_div(2 * tp, 2 * tp + fp + fn)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        metrics.append(ClassMetric(class_id, tp, fp, fn, iou, dice, precision, recall))
    return metrics


def summarize_area(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    class_ids: list[int],
    ignore_index: int = 255,
) -> dict[int, dict[str, int]]:
    valid = labels != ignore_index
    summary = {}
    for class_id in class_ids:
        summary[class_id] = {
            "label_pixels": int(np.logical_and(valid, labels == class_id).sum()),
            "predicted_pixels": int(np.logical_and(valid, predictions == class_id).sum()),
        }
    return summary


def prediction_summary_rows(
    labels: list[np.ndarray],
    predictions: list[np.ndarray],
    *,
    sample_ids: list[str],
    foreground_class_ids: list[int],
    ignore_index: int = 255,
) -> list[dict]:
    if not (len(labels) == len(predictions) == len(sample_ids)):
        raise ValueError("labels, predictions, and sample_ids must have the same length")
    rows = []
    for sample_id, label, prediction in zip(sample_ids, labels, predictions):
        if label.shape != prediction.shape:
            raise ValueError(f"{sample_id}: label and prediction shapes differ")
        valid = label != ignore_index
        label_fg = np.logical_and(valid, np.isin(label, foreground_class_ids))
        pred_fg = np.logical_and(valid, np.isin(prediction, foreground_class_ids))
        tp = int(np.logical_and(label_fg, pred_fg).sum())
        fp = int(np.logical_and(~label_fg, pred_fg).sum())
        fn = int(np.logical_and(label_fg, ~pred_fg).sum())
        label_pixels = int(label_fg.sum())
        predicted_pixels = int(pred_fg.sum())
        rows.append(
            {
                "sample_id": sample_id,
                "label_foreground_pixels": label_pixels,
                "predicted_foreground_pixels": predicted_pixels,
                "predicted_label_area_ratio": _safe_div(predicted_pixels, label_pixels),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "foreground_precision": _safe_div(tp, tp + fp),
                "foreground_recall": _safe_div(tp, tp + fn),
                "foreground_iou": _safe_div(tp, tp + fp + fn),
                "foreground_dice": _safe_div(2 * tp, 2 * tp + fp + fn),
                "all_background_prediction": predicted_pixels == 0,
            }
        )
    return rows


def threshold_sweep(
    labels: np.ndarray,
    foreground_probabilities: np.ndarray,
    *,
    foreground_class_id: int,
    thresholds: list[float],
    ignore_index: int = 255,
) -> list[dict]:
    if labels.shape != foreground_probabilities.shape:
        raise ValueError("labels and foreground_probabilities must have the same shape")
    rows = []
    valid = labels != ignore_index
    label_fg = np.logical_and(valid, labels == foreground_class_id)
    for threshold in thresholds:
        pred_fg = np.logical_and(valid, foreground_probabilities >= threshold)
        tp = int(np.logical_and(label_fg, pred_fg).sum())
        fp = int(np.logical_and(~label_fg, pred_fg).sum())
        fn = int(np.logical_and(label_fg, ~pred_fg).sum())
        predicted_pixels = int(pred_fg.sum())
        label_pixels = int(label_fg.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "label_foreground_pixels": label_pixels,
                "predicted_foreground_pixels": predicted_pixels,
                "predicted_label_area_ratio": _safe_div(predicted_pixels, label_pixels),
                "precision": _safe_div(tp, tp + fp),
                "recall": _safe_div(tp, tp + fn),
                "iou": _safe_div(tp, tp + fp + fn),
                "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
            }
        )
    return rows


def probability_summary(
    labels: np.ndarray,
    foreground_probabilities: np.ndarray,
    *,
    foreground_class_id: int,
    ignore_index: int = 255,
) -> dict[str, float]:
    if labels.shape != foreground_probabilities.shape:
        raise ValueError("labels and foreground_probabilities must have the same shape")
    valid = labels != ignore_index
    label_fg = np.logical_and(valid, labels == foreground_class_id)
    label_bg = np.logical_and(valid, labels != foreground_class_id)
    values = foreground_probabilities[valid]
    fg_values = foreground_probabilities[label_fg]
    bg_values = foreground_probabilities[label_bg]
    return {
        "valid_pixels": int(valid.sum()),
        "foreground_pixels": int(label_fg.sum()),
        "probability_mean": _mean(values),
        "probability_max": _max(values),
        "foreground_probability_mean": _mean(fg_values),
        "foreground_probability_max": _max(fg_values),
        "background_probability_mean": _mean(bg_values),
        "background_probability_max": _max(bg_values),
        "probability_p95": _percentile(values, 95),
        "probability_p99": _percentile(values, 99),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0

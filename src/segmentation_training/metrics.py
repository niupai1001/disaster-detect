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


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


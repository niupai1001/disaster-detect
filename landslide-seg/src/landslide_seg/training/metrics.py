"""分割与检测指标（严格排除无效/未知像素）。"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def _flatten_valid(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """只保留有效像素。

    注意：**不能**用 ``p &= v`` 的写法——那样无效像素会变成 (0,0)，
    从而被计入 TN，导致 FPR/OA 被稀释。必须真正把它们剔除。
    """
    p = pred.reshape(-1).astype(bool)
    g = gt.reshape(-1).astype(bool)
    if valid is not None:
        v = valid.reshape(-1).astype(bool)
        p = p[v]
        g = g[v]
    return p, g


def confusion(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> Dict[str, int]:
    p, g = _flatten_valid(pred, gt, valid)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> float:
    c = confusion(pred, gt, valid)
    denom = c["tp"] + c["fp"] + c["fn"]
    return float(c["tp"] / denom) if denom else float("nan")


def dice(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> float:
    c = confusion(pred, gt, valid)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return float(2 * c["tp"] / denom) if denom else float("nan")


def precision_recall(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None):
    c = confusion(pred, gt, valid)
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else float("nan")
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan")
    return float(p), float(r)


def summarize(counters: Iterable[Dict[str, int]]) -> Dict[str, float]:
    """把逐样本混淆计数聚合为整体指标。"""
    tp = sum(c["tp"] for c in counters)
    fp = sum(c["fp"] for c in counters)
    fn = sum(c["fn"] for c in counters)
    tn = sum(c["tn"] for c in counters)
    iou_v = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    dice_v = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "iou": iou_v,
        "dice": dice_v,
        "precision": prec,
        "recall": rec,
        "false_positive_rate": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

"""pooled 指标口径测试（执行计划 §2.3 / P0 单元测试清单）。

锁住五件事：
1. 前景并集为 0 时 IoU **未定义**；交集为0但并集非0时IoU为0；
2. 完全重叠 / 零交集 / 阈值边界（prob == 0.5 判为正）；
3. 无效像素必须**真正剔除**，不能变成 TN 稀释 FPR；
4. pooled（先累加 TP/FP/FN 再算）**不等于** 逐瓦片 IoU 的算术平均；
5. 同一批 TP/FP/FN 下 `F1 = 2·IoU / (1 + IoU)`。

以上任何一条被改动，历史数值就不可比，因此必须失败而不是静默通过。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from landslide_seg.training import metrics
from landslide_seg.training.segformer_metrics import evaluate_tile, summarize_tiles


def _tile(gt_px: list[tuple[int, int]], pred_px: list[tuple[int, int]], size: int = 16):
    gt = np.zeros((size, size), dtype=bool)
    pr = np.zeros((size, size), dtype=np.float32)
    for y, x in gt_px:
        gt[y, x] = True
    for y, x in pred_px:
        pr[y, x] = 0.9
    return pr, gt, np.ones((size, size), dtype=bool)


# ---------------------------------------------------------------- 1. 未定义
def test_all_zero_gt_and_pred_iou_is_undefined():
    zeros = np.zeros((8, 8), dtype=bool)
    prob = np.zeros((8, 8), dtype=np.float32)
    assert math.isnan(metrics.iou(zeros, zeros))
    r = evaluate_tile(prob, zeros, np.ones((8, 8), dtype=bool))
    assert r["tp"] == r["fp"] == r["fn"] == 0
    s = summarize_tiles([r])
    v = s["pixel"]["iou"]
    assert v is None or math.isnan(v), "全负瓦片的前景 IoU 必须未定义，不能以 1.0 拉高 macro IoU"
    assert s["n_negative_tiles"] == 1
    assert s["n_tiles"] == 1


# ---------------------------------------------------------------- 2. 边界情形
def test_perfect_overlap():
    prob, gt, valid = _tile([(2, 2), (3, 3)], [(2, 2), (3, 3)])
    assert metrics.iou(prob >= 0.5, gt) == pytest.approx(1.0)
    assert metrics.dice(prob >= 0.5, gt) == pytest.approx(1.0)
    p, r = metrics.precision_recall(prob >= 0.5, gt)
    assert p == pytest.approx(1.0) and r == pytest.approx(1.0)


def test_zero_intersection():
    prob, gt, _ = _tile([(2, 2), (3, 3)], [(6, 6), (7, 7)])
    assert metrics.iou(prob >= 0.5, gt) == pytest.approx(0.0)
    assert metrics.dice(prob >= 0.5, gt) == pytest.approx(0.0)


def test_threshold_is_inclusive_at_half():
    prob = np.zeros((4, 4), dtype=np.float32)
    prob[1, 1] = 0.5
    gt = np.zeros((4, 4), dtype=bool)
    gt[1, 1] = True
    r = evaluate_tile(prob, gt, np.ones((4, 4), dtype=bool), threshold=0.5)
    assert r["tp"] == 1 and r["fp"] == 0, "prob == 0.5 应判为正类（>= 阈值）"


# ---------------------------------------------------------------- 3. ignore
def test_ignore_pixels_are_removed_not_counted_as_tn():
    prob = np.zeros((4, 4), dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    valid = np.zeros((4, 4), dtype=bool)
    # 全部 4 个前景像素都在无效区：应完全不进入混淆矩阵
    gt[0, :4] = True
    valid[1:, :] = True
    c = metrics.confusion(prob >= 0.5, gt, valid)
    assert c["fn"] == 0 and c["tp"] == 0
    assert c["tn"] == 12, "无效区不能计入 TN"


def test_ignore_pixels_do_not_dilute_fpr():
    prob = np.zeros((4, 4), dtype=np.float32)
    gt = np.zeros((4, 4), dtype=bool)
    prob[0, :4] = 0.9  # 4 个误报
    valid_all = np.ones((4, 4), dtype=bool)
    valid_half = np.zeros((4, 4), dtype=bool)
    valid_half[:2, :] = True  # 只保留一半有效区
    fpr_all = metrics.summarize([metrics.confusion(prob >= 0.5, gt, valid_all)])["false_positive_rate"]
    fpr_half = metrics.summarize([metrics.confusion(prob >= 0.5, gt, valid_half)])["false_positive_rate"]
    assert fpr_all == pytest.approx(4 / 16)
    assert fpr_half == pytest.approx(4 / 8), "剔除无效区后 FPR 应按有效像素重算"


# ---------------------------------------------------------------- 4. pooled
def test_pooled_is_not_tile_macro():
    # 瓦片 A：大目标，预测刚好覆盖一半 → tile IoU = 0.5
    gt_a = np.zeros((8, 8), dtype=bool)
    prob_a = np.zeros((8, 8), dtype=np.float32)
    gt_a[:4, :4] = True          # 16 px
    prob_a[:2, :4] = 0.9         # 8 px
    # 瓦片 B：小目标，预测完全正确 → tile IoU = 1.0
    gt_b = np.zeros((8, 8), dtype=bool)
    prob_b = np.zeros((8, 8), dtype=np.float32)
    gt_b[0, 0] = True
    prob_b[0, 0] = 0.9
    rows = [
        evaluate_tile(prob_a, gt_a, np.ones((8, 8), dtype=bool)),
        evaluate_tile(prob_b, gt_b, np.ones((8, 8), dtype=bool)),
    ]
    tile_iou = [r["tp"] / (r["tp"] + r["fp"] + r["fn"]) for r in rows]
    pooled = summarize_tiles(rows)["pixel"]["iou"]
    macro = sum(tile_iou) / len(tile_iou)
    assert pooled == pytest.approx(9 / 17)   # (8+1) / (16+1+... ) 按累加计数
    assert macro == pytest.approx((0.5 + 1.0) / 2)
    assert abs(pooled - macro) > 0.2, "pooled 与 tile-macro 必须是两个数，不能混用"


# ---------------------------------------------------------------- 5. F1 ↔ IoU
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_f1_iou_identity(seed):
    rng = np.random.default_rng(seed)
    pred = rng.random((32, 32)) > 0.7
    gt = rng.random((32, 32)) > 0.8
    i = metrics.iou(pred, gt)
    d = metrics.dice(pred, gt)
    assert d == pytest.approx(2 * i / (1 + i), rel=1e-9)


def test_f1_iou_identity_on_pooled_counts():
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(5):
        pred = rng.random((16, 16)) > 0.75
        gt = rng.random((16, 16)) > 0.85
        rows.append(evaluate_tile(pred.astype(np.float32), gt, np.ones((16, 16), dtype=bool)))
    s = summarize_tiles(rows)
    assert s["pixel"]["dice"] == pytest.approx(2 * s["pixel"]["iou"] / (1 + s["pixel"]["iou"]), rel=1e-9)


# ---------------------------------------------------------------- 6. 负瓦片
def test_negative_tile_flags_false_positive():
    gt = np.zeros((8, 8), dtype=bool)
    prob = np.zeros((8, 8), dtype=np.float32)
    prob[1, 1] = 0.9
    r = evaluate_tile(prob, gt, np.ones((8, 8), dtype=bool))
    assert r["is_negative_tile"] is True
    assert r["pred_positive_on_negative"] is True
    assert r["fn"] == 0

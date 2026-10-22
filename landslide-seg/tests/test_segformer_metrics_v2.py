"""metric v2 回归用例：tiny 分母、尺度像素召回、无效区改变连通性。

这些用例只验证指标代码的口径，不是模型效果数据。
"""

from __future__ import annotations

import numpy as np

from landslide_seg.training.segformer_metrics import (
    evaluate_tile,
    evaluate_tile_v2,
    summarize_tiles_v2,
)


def _canvas() -> np.ndarray:
    return np.zeros((128, 128), dtype=bool)


def test_tiny_matched_evaluable_is_recorded():
    gt = _canvas()
    gt[60:62, 60:62] = True  # 4 px，tiny
    valid = np.ones_like(gt)
    row = evaluate_tile_v2(gt.astype(np.float32), gt, valid)
    assert row["tiny"]["n_objects"] == 1
    assert row["tiny"]["n_objects_evaluable"] == 1
    assert row["tiny"]["n_matched_evaluable"] == 1
    assert row["tiny"]["object_recall_evaluable"] == 1.0
    assert row["tiny"]["pixel_recall"] == 1.0

    summary = summarize_tiles_v2([row])
    tiny = summary["objects"]["tiny"]
    assert tiny["n_matched_evaluable"] == 1
    assert tiny["object_recall_evaluable"] == 1.0


def test_scale_pixel_recall_does_not_require_object_match():
    gt = _canvas()
    gt[20:40, 20:40] = True   # 400 px -> large
    gt[80:90, 80:90] = True   # 100 px -> medium
    valid = np.ones_like(gt)
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[80:90, 80:90] = 1.0            # medium 完整命中
    prob[20, 20:40] = 1.0               # large 只命中一行（20 px，IoU≈0.048 <0.25）
    row = evaluate_tile_v2(prob, gt, valid)
    large = row["bins"]["large"]
    assert large["n_objects"] == 1
    # 对象未匹配，但 20 个正确像素必须计入尺度像素召回
    assert large["n_matched_evaluable"] == 0
    assert abs(large["pixel_recall"] - 20 / 400) < 1e-6
    medium = row["bins"]["medium"]
    assert medium["n_matched_evaluable"] == 1
    assert medium["pixel_recall"] == 1.0


def test_invalid_prediction_is_masked_before_components():
    gt = _canvas()
    gt[40:50, 80:90] = True       # 100 px，距有效区边界 20 px，不算截断
    valid = np.zeros_like(gt)
    valid[:, 60:] = True          # 左 60 列无效
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[40:50, 0:90] = 1.0       # 预测从无效区一路延伸到 GT（一个 900 px 大连通域）

    old = evaluate_tile(prob, gt, valid)
    new = evaluate_tile_v2(prob, gt, valid)
    # 旧口径：整段无效区预测把对象撑到 900 px，IoU=0.111，匹配失败
    assert old["n_matched"] == 0
    # v2：先遮蔽无效区，预测在有效区内是 300 px（100 TP + 200 FP），IoU=0.333，匹配成功
    assert new["n_matched_evaluable"] == 1
    assert new["tp"] == 100 and new["fp"] == 200
    assert new["tiny"]["n_objects"] == 0


def test_prediction_only_in_invalid_area_is_not_a_false_positive():
    gt = _canvas()
    valid = np.ones_like(gt)
    valid[:, :64] = False
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[10:20, 10:20] = 1.0  # 全部落在无效区
    row = evaluate_tile_v2(prob, gt, valid)
    assert row["pred_px"] == 0
    assert row["fp"] == 0
    assert row["pred_positive_on_negative"] is False

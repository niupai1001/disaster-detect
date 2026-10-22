"""指标口径测试：无效像素剔除、尺度分档、对象匹配、截断、边界 F1、负瓦片。"""

from __future__ import annotations

import numpy as np
import pytest

from landslide_seg.training.segformer_metrics import (
    assign_bin,
    boundary_metrics,
    evaluate_tile,
    label_objects,
    object_table,
    summarize_tiles,
)


def _gt_with_rect(rows=slice(20, 40), cols=slice(30, 50)) -> np.ndarray:
    gt = np.zeros((128, 128), dtype=bool)
    gt[rows, cols] = True
    return gt


def test_assign_bin_boundaries():
    assert assign_bin(49) == "small"
    assert assign_bin(50) == "medium"
    assert assign_bin(199) == "medium"
    assert assign_bin(200) == "large"


def test_perfect_prediction():
    gt = _gt_with_rect()
    valid = np.ones_like(gt, dtype=bool)
    row = evaluate_tile(gt.astype(np.float32), gt, valid, sample_id="a", event_id="e")
    assert row["tp"] == int(gt.sum())
    assert row["fp"] == 0 and row["fn"] == 0
    assert row["n_gt_objects"] == 1 and row["n_matched"] == 1
    summary = summarize_tiles([row])
    assert summary["pixel"]["iou"] == pytest.approx(1.0)
    assert summary["objects"]["object_recall"] == pytest.approx(1.0)
    assert summary["boundary"]["f1"] == pytest.approx(1.0, abs=1e-6)


def test_invalid_pixels_are_not_counted_as_tn():
    gt = np.zeros((128, 128), dtype=bool)
    valid = np.zeros((128, 128), dtype=bool)
    valid[50:70, 50:70] = True
    pred = np.ones((128, 128), dtype=bool)
    row = evaluate_tile(pred.astype(np.float32), gt, valid)
    assert row["fp"] == int(valid.sum())
    assert row["tn"] == 0
    assert row["valid_px"] == int(valid.sum())


def test_truncated_object_detection():
    gt = np.zeros((128, 128), dtype=bool)
    gt[0:10, 0:10] = True  # 触左上角
    labels, n = label_objects(gt)
    objs = object_table(labels, n, np.ones_like(gt, dtype=bool))
    assert len(objs) == 1 and objs[0]["truncated"] is True

    gt2 = _gt_with_rect()
    labels2, n2 = label_objects(gt2)
    objs2 = object_table(labels2, n2, np.ones_like(gt2, dtype=bool))
    assert objs2[0]["truncated"] is False

    # 与无效区相邻也算截断
    valid = np.ones_like(gt2, dtype=bool)
    valid[20:25, 30:35] = False
    labels3, n3 = label_objects(gt2)
    objs3 = object_table(labels3, n3, valid)
    assert objs3[0]["truncated"] is True


def test_object_recall_half():
    gt = np.zeros((128, 128), dtype=bool)
    gt[20:40, 30:50] = True
    gt[60:80, 80:100] = True
    pred = np.zeros((128, 128), dtype=bool)
    pred[20:40, 30:50] = True  # 只匹配第一个
    valid = np.ones_like(gt, dtype=bool)
    row = evaluate_tile(pred.astype(np.float32), gt, valid)
    assert row["n_gt_objects"] == 2
    assert row["n_matched"] == 1
    summary = summarize_tiles([row])
    assert summary["objects"]["object_recall"] == pytest.approx(0.5)
    assert summary["objects"]["object_recall_evaluable"] == pytest.approx(0.5)


def test_tiny_object_counted_separately():
    gt = np.zeros((128, 128), dtype=bool)
    gt[50:52, 50:51] = True  # 2 px
    gt[60:62, 60:62] = True  # 4 px
    gt[80:100, 80:100] = True  # 400 px
    valid = np.ones_like(gt, dtype=bool)
    row = evaluate_tile(gt.astype(np.float32), gt, valid)
    summary = summarize_tiles([row])
    assert summary["objects"]["tiny_1_4px"]["n_objects"] == 2


def test_empty_pred_and_empty_gt_boundary():
    gt = np.zeros((128, 128), dtype=bool)
    valid = np.ones_like(gt, dtype=bool)
    row = evaluate_tile(np.zeros_like(gt, dtype=np.float32), gt, valid)
    assert row["boundary"]["status"] == "both_empty"
    summary = summarize_tiles([row])
    assert summary["boundary"]["f1"] is None
    assert summary["pixel"]["iou"] is None
    assert summary["negative_tiles"]["n"] == 1


def test_negative_tile_false_alarm():
    gt = np.zeros((128, 128), dtype=bool)
    pred = np.zeros((128, 128), dtype=bool)
    pred[10:20, 10:20] = True
    valid = np.ones_like(gt, dtype=bool)
    row = evaluate_tile(pred.astype(np.float32), gt, valid)
    assert row["is_negative_tile"] is True
    assert row["pred_positive_on_negative"] is True
    summary = summarize_tiles([row])
    assert summary["negative_tiles"]["tile_false_alarm_rate"] == pytest.approx(1.0)


def test_boundary_with_missing_prediction_is_zero():
    gt = _gt_with_rect()
    valid = np.ones_like(gt, dtype=bool)
    row = evaluate_tile(np.zeros_like(gt, dtype=np.float32), gt, valid)
    assert row["boundary"]["status"] == "pred_empty"
    summary = summarize_tiles([row])
    assert summary["boundary"]["f1"] == 0.0


def test_boundary_metrics_tolerance():
    gt = _gt_with_rect()
    pred = _gt_with_rect(rows=slice(21, 41), cols=slice(30, 50))  # 下移 1 px
    valid = np.ones_like(gt, dtype=bool)
    b = boundary_metrics(pred, gt, valid, tol=1)
    assert b["prec_tp"] == b["pred_px"] or b["prec_tp"] > 0
    assert b["rec_tp"] > 0

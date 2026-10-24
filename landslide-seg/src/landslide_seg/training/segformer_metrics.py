"""SegFormer 评测指标：像素 / 尺度 / 对象召回 / 边界 F1 / 负瓦片。

只依赖 numpy + scipy（环境里没有 scikit-image）。所有函数的输入都是单张
``128×128`` 的数组：``prob`` 为 sigmoid 概率、``gt`` 为 {0,1} 掩膜、``valid`` 为 bool。

口径（对应执行计划 v2 §8）：
- 像素：复用 ``training/metrics.py``，无效像素**真正剔除**，不计入 TN；
- 尺度：GT 8 邻接连通域，按**对象面积**分档（1 px = 100 m²）：
  small <50 px、medium 50–200 px、large ≥200 px；1–4 px 另列 tiny；
- 对象召回：GT 与预测各做连通域，一对一**贪心**匹配（按 IoU 降序），
  IoU≥0.25 主、≥0.5 严格；触瓦片边缘或与无效区相邻的对象记为"截断"，主指标用未截断对象；
- 边界 F1：边界 = 掩膜 XOR 1 px 腐蚀；排除距无效区/瓦片边缘 ≤容差 的部分；
  微平均（把所有瓦片的边界像素加总）而不是对逐瓦片 F1 取平均；
- 负瓦片：只对 GT 全 0 的瓦片统计"是否出现预测正像素"。本数据集只有 2 个负瓦片，
  因此瓦片级 FPR 只能报 N/A，另给背景像素 FPR。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage

from .metrics import confusion

_NEIGHBOR8 = np.ones((3, 3), dtype=bool)


# --------------------------------------------------------------------------
# 连通域与对象属性
# --------------------------------------------------------------------------
def label_objects(mask: np.ndarray, connectivity: int = 2) -> tuple[np.ndarray, int]:
    """8 邻接（默认）/4 邻接连通域标记。"""
    structure = _NEIGHBOR8 if connectivity == 2 else None
    labels, n = ndimage.label(np.asarray(mask).astype(bool), structure=structure)
    return labels, int(n)


def object_table(labels: np.ndarray, n: int, valid: np.ndarray | None = None) -> list[dict]:
    """每个对象的 id / 面积 / bbox / 是否截断。

    截断判据：对象像素触及瓦片边缘，或与 ``valid=False`` 的像素相邻（1 px）。
    """
    if n <= 0:
        return []
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=n + 1)[1:]
    slices = ndimage.find_objects(labels)

    bad = np.zeros(labels.shape, dtype=bool)
    bad[0, :] = bad[-1, :] = bad[:, 0] = bad[:, -1] = True
    if valid is not None:
        bad = ndimage.binary_dilation(bad | (~np.asarray(valid, dtype=bool)), iterations=1)
    hit = set(np.unique(labels[bad & (labels > 0)]).tolist())

    out: list[dict] = []
    for i in range(1, n + 1):
        sl = slices[i - 1]
        if sl is None:  # pragma: no cover - ndimage 保证非空
            continue
        out.append(
            {
                "id": i,
                "area": int(areas[i - 1]),
                "bbox": [int(sl[0].start), int(sl[1].start), int(sl[0].stop), int(sl[1].stop)],
                "truncated": i in hit,
            }
        )
    return out


def match_objects(
    gt_labels: np.ndarray,
    gt_objects: Sequence[dict],
    pred_labels: np.ndarray,
    pred_objects: Sequence[dict],
    iou_thr: float = 0.25,
) -> dict[int, dict]:
    """一对一贪心匹配。返回 ``{gt_id: {pred_id, iou, inter}}``。"""
    if not gt_objects or not pred_objects:
        return {}
    pred_area = {o["id"]: o["area"] for o in pred_objects}
    pairs: list[tuple[float, int, int, int]] = []
    for g in gt_objects:
        ids, counts = np.unique(pred_labels[gt_labels == g["id"]], return_counts=True)
        for pid, inter in zip(ids.tolist(), counts.tolist()):
            if pid == 0:
                continue
            iou = inter / max(g["area"] + pred_area[pid] - inter, 1)
            if iou >= iou_thr:
                pairs.append((float(iou), int(g["id"]), int(pid), int(inter)))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    out: dict[int, dict] = {}
    for iou, gid, pid, inter in pairs:
        if gid in used_g or pid in used_p:
            continue
        used_g.add(gid)
        used_p.add(pid)
        out[gid] = {"pred_id": pid, "iou": iou, "inter": inter}
    return out


# --------------------------------------------------------------------------
# 边界
# --------------------------------------------------------------------------
def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    eroded = ndimage.binary_erosion(mask, structure=_NEIGHBOR8)
    return mask & ~eroded


def boundary_metrics(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None, tol: int = 1
) -> dict:
    """单张图的边界计数。返回 ``gt_px/pred_px/prec_tp/rec_tp/status``。"""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    shape = gt.shape
    edge = np.zeros(shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    excluded = edge.copy()
    if valid is not None:
        excluded |= ~np.asarray(valid, dtype=bool)
    keep = ~ndimage.binary_dilation(excluded, iterations=int(tol))

    gb = boundary_pixels(gt) & keep
    pb = boundary_pixels(pred) & keep
    res = {"gt_px": int(gb.sum()), "pred_px": int(pb.sum()), "prec_tp": 0, "rec_tp": 0, "status": "ok"}
    if res["gt_px"] == 0 and res["pred_px"] == 0:
        res["status"] = "both_empty"
        return res
    if res["gt_px"] == 0:
        res["status"] = "gt_empty"
    if res["pred_px"] == 0:
        res["status"] = "pred_empty"
    if res["pred_px"] > 0 and res["gt_px"] > 0:
        dist_to_gt = ndimage.distance_transform_edt(~gb)
        res["prec_tp"] = int((dist_to_gt[pb] <= tol).sum())
    if res["gt_px"] > 0 and res["pred_px"] > 0:
        dist_to_pred = ndimage.distance_transform_edt(~pb)
        res["rec_tp"] = int((dist_to_pred[gb] <= tol).sum())
    return res


# --------------------------------------------------------------------------
# 尺度分档
# --------------------------------------------------------------------------
def assign_bin(area_px: int, thresholds_px: Sequence[float] = (50.0, 200.0)) -> str:
    low, high = float(thresholds_px[0]), float(thresholds_px[1])
    if area_px < low:
        return "small"
    if area_px < high:
        return "medium"
    return "large"


def _empty_bin() -> dict:
    return {"n": 0, "n_eval": 0, "matched": 0, "matched_eval": 0, "px": 0, "px_hit": 0}


def evaluate_tile(
    prob: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float = 0.5,
    thresholds_px: Sequence[float] = (50.0, 200.0),
    iou_threshold: float = 0.25,
    iou_strict: float = 0.5,
    boundary_tol: int = 1,
    sample_id: str | None = None,
    event_id: str | None = None,
) -> dict:
    """单张瓦片的全部指标（JSON 友好）。"""
    pred = np.asarray(prob) >= float(threshold)
    gt = np.asarray(gt).astype(bool)
    valid = np.asarray(valid).astype(bool)

    counts = confusion(pred, gt, valid)
    gt_lab, gt_n = label_objects(gt)
    pr_lab, pr_n = label_objects(pred)
    gt_objs = object_table(gt_lab, gt_n, valid)
    pr_objs = object_table(pr_lab, pr_n, valid)
    m_main = match_objects(gt_lab, gt_objs, pr_lab, pr_objs, iou_threshold)
    m_strict = match_objects(gt_lab, gt_objs, pr_lab, pr_objs, iou_strict)

    bins = {name: _empty_bin() for name in ("small", "medium", "large")}
    tiny = _empty_bin()
    for obj in gt_objs:
        hit = m_main.get(obj["id"])
        inter = hit["inter"] if hit else 0
        b = assign_bin(obj["area"], thresholds_px)
        rec = bins[b]
        rec["n"] += 1
        rec["px"] += obj["area"]
        rec["px_hit"] += inter
        if hit:
            rec["matched"] += 1
        if not obj["truncated"]:
            rec["n_eval"] += 1
            if hit:
                rec["matched_eval"] += 1
        if 1 <= obj["area"] <= 4:
            tiny["n"] += 1
            tiny["px"] += obj["area"]
            tiny["px_hit"] += inter
            if hit:
                tiny["matched"] += 1
            if not obj["truncated"]:
                tiny["n_eval"] += 1

    return {
        "sample_id": sample_id,
        "event_id": event_id,
        "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
        "gt_px": int(gt.sum()),
        "pred_px": int(pred.sum()),
        "valid_px": int(valid.sum()),
        "n_gt_objects": gt_n,
        "n_pred_objects": pr_n,
        "n_gt_objects_evaluable": sum(1 for o in gt_objs if not o["truncated"]),
        "n_truncated_objects": sum(1 for o in gt_objs if o["truncated"]),
        "n_matched": len(m_main),
        "n_matched_strict": len(m_strict),
        "n_matched_evaluable": sum(
            1 for o in gt_objs if (not o["truncated"]) and o["id"] in m_main
        ),
        "bins": bins,
        "tiny": tiny,
        "boundary": boundary_metrics(pred, gt, valid, tol=boundary_tol),
        "is_negative_tile": bool(gt.sum() == 0),
        "pred_positive_on_negative": bool(gt.sum() == 0 and pred.any()),
    }


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        if precision == 0 or recall == 0:
            return 0.0
        return None
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def summarize_tiles(rows: Iterable[dict]) -> dict[str, Any]:
    """把逐瓦片结果聚合成一份报告。"""
    rows = list(rows)
    if not rows:
        return {"n_tiles": 0}

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)

    bins_out: dict[str, dict] = {}
    for name in ("small", "medium", "large"):
        n = sum(r["bins"][name]["n"] for r in rows)
        n_eval = sum(r["bins"][name]["n_eval"] for r in rows)
        matched = sum(r["bins"][name]["matched"] for r in rows)
        matched_eval = sum(r["bins"][name]["matched_eval"] for r in rows)
        px = sum(r["bins"][name]["px"] for r in rows)
        px_hit = sum(r["bins"][name]["px_hit"] for r in rows)
        bins_out[name] = {
            "n_objects": n,
            "n_objects_evaluable": n_eval,
            "n_matched": matched,
            "n_matched_evaluable": matched_eval,
            "object_recall": _safe_div(matched, n),
            "object_recall_evaluable": _safe_div(matched_eval, n_eval),
            "pixel_recall": _safe_div(px_hit, px),
            "gt_px": px,
        }

    tiny = {
        "n_objects": sum(r["tiny"]["n"] for r in rows),
        "n_objects_evaluable": sum(r["tiny"]["n_eval"] for r in rows),
        "n_matched": sum(r["tiny"]["matched"] for r in rows),
        "pixel_recall": _safe_div(
            sum(r["tiny"]["px_hit"] for r in rows), sum(r["tiny"]["px"] for r in rows)
        ),
    }

    n_gt = sum(r["n_gt_objects"] for r in rows)
    n_gt_eval = sum(r["n_gt_objects_evaluable"] for r in rows)
    n_matched = sum(r["n_matched"] for r in rows)
    n_matched_eval = sum(r["n_matched_evaluable"] for r in rows)
    n_matched_strict = sum(r["n_matched_strict"] for r in rows)

    gb = sum(r["boundary"]["gt_px"] for r in rows)
    pb = sum(r["boundary"]["pred_px"] for r in rows)
    prec_tp = sum(r["boundary"]["prec_tp"] for r in rows)
    rec_tp = sum(r["boundary"]["rec_tp"] for r in rows)
    b_prec = _safe_div(prec_tp, pb)
    b_rec = _safe_div(rec_tp, gb)
    if gb == 0 and pb == 0:
        b_f1 = None
    elif gb > 0 and pb == 0:
        b_f1 = 0.0
    elif gb == 0 and pb > 0:
        b_f1 = 0.0
    else:
        b_f1 = _f1(b_prec, b_rec)

    neg = [r for r in rows if r["is_negative_tile"]]
    neg_hit = sum(1 for r in neg if r["pred_positive_on_negative"])

    return {
        "n_tiles": len(rows),
        "n_positive_tiles": sum(1 for r in rows if not r["is_negative_tile"]),
        "n_negative_tiles": len(neg),
        "pixel": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": iou, "dice": dice, "precision": precision, "recall": recall,
            "false_positive_rate": fpr,
        },
        "objects": {
            "n_gt": n_gt,
            "n_gt_evaluable": n_gt_eval,
            "n_matched": n_matched,
            "n_matched_evaluable": n_matched_eval,
            "n_matched_strict": n_matched_strict,
            "object_recall": _safe_div(n_matched, n_gt),
            "object_recall_evaluable": _safe_div(n_matched_eval, n_gt_eval),
            "object_recall_strict": _safe_div(n_matched_strict, n_gt),
            "bins": bins_out,
            "tiny_1_4px": tiny,
        },
        "boundary": {
            "gt_px": gb, "pred_px": pb,
            "precision": b_prec, "recall": b_rec, "f1": b_f1,
            "tiles_both_empty": sum(1 for r in rows if r["boundary"]["status"] == "both_empty"),
            "tiles_gt_empty": sum(1 for r in rows if r["boundary"]["status"] == "gt_empty"),
            "tiles_pred_empty": sum(1 for r in rows if r["boundary"]["status"] == "pred_empty"),
        },
        "negative_tiles": {
            "n": len(neg),
            "n_with_pred_positive": neg_hit,
            "tile_false_alarm_rate": (_safe_div(neg_hit, len(neg)) if neg else None),
            "note": (
                "本数据集只有 2 个可靠负瓦片（均在 val），比率不可解释；"
                "请改用 pixel.false_positive_rate（背景像素级）。"
                if 0 < len(neg) <= 2
                else "瓦片级误报率需要足够多的经核实负瓦片"
            ),
        },
    }


def summarize_by_event(rows: Iterable[dict]) -> dict[str, dict]:
    rows = list(rows)
    events = sorted({r.get("event_id") or "unknown" for r in rows})
    out: dict[str, dict] = {}
    for ev in events:
        sub = [r for r in rows if (r.get("event_id") or "unknown") == ev]
        out[ev] = summarize_tiles(sub)
    return out


# ==========================================================================
# metric v2
#
# 审阅发现旧口径有三处问题（旧函数保持不变，便于新旧对照）：
# 1. 预测连通域直接建在原始预测上，无效区里的预测会把相邻对象连成一片，
#    改变连通性、面积与匹配 IoU；截断 GT 过滤并不能消除这个影响。
# 2. 各尺度的 pixel_recall 只累加"匹配上的对象"的交集，未匹配但预测正确的
#    前景像素被丢掉，不是通常意义的尺度像素召回。
# 3. tiny 的 matched_eval 从未递增，也没有输出可评价对象召回。
#
# v2 的固定口径：
# - GT 对象的面积/身份仍来自原始掩膜；截断判定用 valid（触边或邻无效区）；
# - **先** pred &= valid，再建预测连通域；
# - 主对象指标分母 = 未截断 GT，只与该子集做一对一贪心匹配；
# - 尺度像素召回 = (预测 ∩ 对象 ∩ valid) / (对象 ∩ valid)，与是否匹配无关；
# - tiny（1–4 px）同时给 matched_all / matched_evaluable 与两种召回。
# ==========================================================================

METRIC_VERSION = "v2"


def evaluate_tile_v2(
    prob: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float = 0.5,
    thresholds_px: Sequence[float] = (50.0, 200.0),
    iou_threshold: float = 0.25,
    iou_strict: float = 0.5,
    boundary_tol: int = 1,
    sample_id: str | None = None,
    event_id: str | None = None,
) -> dict:
    """单张瓦片的 v2 指标（见文件末尾的口径说明）。"""
    pred_prob = np.asarray(prob, dtype=np.float32)
    gt_b = np.asarray(gt).astype(bool)
    valid_b = np.asarray(valid).astype(bool)
    pred = (pred_prob >= float(threshold)) & valid_b  # 先遮蔽无效区

    counts = confusion(pred, gt_b, valid_b)

    gt_lab, gt_n = label_objects(gt_b)
    gt_objs = object_table(gt_lab, gt_n, valid_b)
    pr_lab, pr_n = label_objects(pred)
    pr_objs = object_table(pr_lab, pr_n, None)  # pred 已被 valid 遮蔽

    gt_eval = [o for o in gt_objs if not o["truncated"]]
    m_all = match_objects(gt_lab, gt_objs, pr_lab, pr_objs, iou_threshold)
    m_eval = match_objects(gt_lab, gt_eval, pr_lab, pr_objs, iou_threshold)
    m_strict = match_objects(gt_lab, gt_eval, pr_lab, pr_objs, iou_strict)

    def bin_stats(predicate) -> dict:
        objs = [o for o in gt_objs if predicate(o["area"])]
        ids = [o["id"] for o in objs]
        px = hit = 0
        if ids:
            region = np.isin(gt_lab, ids) & valid_b
            px = int(region.sum())
            hit = int((pred & region).sum())
        ev = [o for o in objs if not o["truncated"]]
        stats = {
            "n_objects": len(objs),
            "n_objects_evaluable": len(ev),
            "n_matched_all": sum(1 for o in objs if o["id"] in m_all),
            "n_matched_evaluable": sum(1 for o in ev if o["id"] in m_eval),
            "gt_px": px,
            "hit_px": hit,
            "pixel_recall": (hit / px) if px else None,
        }
        stats["object_recall_all"] = (
            stats["n_matched_all"] / stats["n_objects"] if stats["n_objects"] else None
        )
        stats["object_recall_evaluable"] = (
            stats["n_matched_evaluable"] / stats["n_objects_evaluable"]
            if stats["n_objects_evaluable"] else None
        )
        return stats

    bins = {
        name: bin_stats(lambda a, name=name: assign_bin(a, thresholds_px) == name)
        for name in ("small", "medium", "large")
    }
    tiny = bin_stats(lambda a: 1 <= a <= 4)

    fine_bins = {
        name: bin_stats(lambda a, lo=lo, hi=hi: a >= lo and (hi is None or a < hi))
        for name, lo, hi in (("1_4", 1, 5), ("5_9", 5, 10), ("10_19", 10, 20),
                             ("20_49", 20, 50), ("50_199", 50, 200), ("ge200", 200, None))
    }

    return {
        "metric_version": METRIC_VERSION,
        "sample_id": sample_id,
        "event_id": event_id,
        "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
        "gt_px": int(gt_b.sum()),
        "pred_px": int(pred.sum()),
        "valid_px": int(valid_b.sum()),
        "n_gt_objects": gt_n,
        "n_gt_objects_evaluable": len(gt_eval),
        "n_truncated_objects": gt_n - len(gt_eval),
        "n_pred_objects": pr_n,
        "n_matched_all": len(m_all),
        "n_matched_evaluable": len(m_eval),
        "n_matched_strict_evaluable": len(m_strict),
        "bins": bins,
        "tiny": tiny,
        "fine_bins": fine_bins,
        "use_bins": {"1_4": tiny, "5_25": bin_stats(lambda a: 5 <= a <= 25),
                     "26_plus": bin_stats(lambda a: a >= 26)},
        "boundary": boundary_metrics(pred, gt_b, valid_b, tol=boundary_tol),
        "is_negative_tile": bool(gt_b.sum() == 0),
        "pred_positive_on_negative": bool(gt_b.sum() == 0 and pred.any()),
    }


def _agg_items(items: Sequence[dict]) -> dict:
    n = sum(x["n_objects"] for x in items)
    n_eval = sum(x["n_objects_evaluable"] for x in items)
    matched_all = sum(x["n_matched_all"] for x in items)
    matched_eval = sum(x["n_matched_evaluable"] for x in items)
    gt_px = sum(x["gt_px"] for x in items)
    hit_px = sum(x["hit_px"] for x in items)
    return {
        "n_objects": n,
        "n_objects_evaluable": n_eval,
        "n_matched_all": matched_all,
        "n_matched_evaluable": matched_eval,
        "object_recall_all": (matched_all / n) if n else None,
        "object_recall_evaluable": (matched_eval / n_eval) if n_eval else None,
        "gt_px": gt_px,
        "hit_px": hit_px,
        "pixel_recall": (hit_px / gt_px) if gt_px else None,
    }


def summarize_tiles_v2(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {"metric_version": METRIC_VERSION, "n_tiles": 0}

    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)

    n_gt = sum(r["n_gt_objects"] for r in rows)
    n_gt_eval = sum(r["n_gt_objects_evaluable"] for r in rows)
    n_matched = sum(r["n_matched_all"] for r in rows)
    n_matched_eval = sum(r["n_matched_evaluable"] for r in rows)
    n_matched_strict = sum(r["n_matched_strict_evaluable"] for r in rows)

    gb = sum(r["boundary"]["gt_px"] for r in rows)
    pb = sum(r["boundary"]["pred_px"] for r in rows)
    prec_tp = sum(r["boundary"]["prec_tp"] for r in rows)
    rec_tp = sum(r["boundary"]["rec_tp"] for r in rows)
    b_prec = _safe_div(prec_tp, pb)
    b_rec = _safe_div(rec_tp, gb)
    if gb == 0 and pb == 0:
        b_f1 = None
    elif gb > 0 and pb == 0:
        b_f1 = 0.0
    elif gb == 0 and pb > 0:
        b_f1 = 0.0
    else:
        b_f1 = _f1(b_prec, b_rec)

    neg = [r for r in rows if r["is_negative_tile"]]
    neg_hit = sum(1 for r in neg if r["pred_positive_on_negative"])

    return {
        "metric_version": METRIC_VERSION,
        "n_tiles": len(rows),
        "n_positive_tiles": sum(1 for r in rows if not r["is_negative_tile"]),
        "n_negative_tiles": len(neg),
        "pixel": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": iou, "dice": dice,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "false_positive_rate": _safe_div(fp, fp + tn),
        },
        "objects": {
            "n_gt": n_gt,
            "n_gt_evaluable": n_gt_eval,
            "n_matched_all": n_matched,
            "n_pred": sum(r["n_pred_objects"] for r in rows),
            "unmatched_prediction_objects": sum(r["n_pred_objects"] - r["n_matched_all"] for r in rows),
            "unmatched_prediction_objects_per_tile": sum(r["n_pred_objects"] - r["n_matched_all"] for r in rows) / len(rows),
            "n_matched_evaluable": n_matched_eval,
            "object_recall_all": (n_matched / n_gt) if n_gt else None,
            "object_recall_evaluable": (n_matched_eval / n_gt_eval) if n_gt_eval else None,
            "object_recall_strict_evaluable": (n_matched_strict / n_gt_eval) if n_gt_eval else None,
            "bins": {name: _agg_items([r["bins"][name] for r in rows]) for name in ("small", "medium", "large")},
            "tiny": _agg_items([r["tiny"] for r in rows]),
            "fine_bins": {name: _agg_items([r["fine_bins"][name] for r in rows])
                          for name in rows[0].get("fine_bins", {})},
            "use_bins": {name: _agg_items([r['use_bins'][name] for r in rows])
                         for name in rows[0].get('use_bins', {})},
        },
        "boundary": {
            "gt_px": gb, "pred_px": pb,
            "precision": b_prec, "recall": b_rec, "f1": b_f1,
            "tiles_both_empty": sum(1 for r in rows if r["boundary"]["status"] == "both_empty"),
            "tiles_gt_empty": sum(1 for r in rows if r["boundary"]["status"] == "gt_empty"),
            "tiles_pred_empty": sum(1 for r in rows if r["boundary"]["status"] == "pred_empty"),
        },
        "negative_tiles": {
            "n": len(neg),
            "n_with_pred_positive": neg_hit,
            "tile_false_alarm_rate": (_safe_div(neg_hit, len(neg)) if neg else None),
        },
    }


def summarize_by_event_v2(rows: Iterable[dict]) -> dict[str, dict]:
    rows = list(rows)
    events = sorted({r.get("event_id") or "unknown" for r in rows})
    return {
        ev: summarize_tiles_v2([r for r in rows if (r.get("event_id") or "unknown") == ev])
        for ev in events
    }


def pixel_average_precision(prob: np.ndarray, target: np.ndarray) -> float | None:
    """Exact noninterpolated binary AP; equal scores are one threshold group.

    Inputs must already contain valid pixels only. All-negative target is
    undefined; constant scores with positives return their prevalence.
    """
    p = np.asarray(prob).ravel()
    y = np.asarray(target, dtype=bool).ravel()
    if p.size != y.size or not np.isfinite(p).all():
        raise ValueError("AP requires aligned finite scores")
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-p, kind="stable")
    scores, labels = p[order], y[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(labels, dtype=np.int64)[ends]
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / positives)

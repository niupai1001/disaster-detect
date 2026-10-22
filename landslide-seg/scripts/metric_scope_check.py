"""口径换算：同一份预测，分别在「仅 valid 像素」与「全像素」下重算指标。

为什么需要：官方 `src/lit_module.py` 用
``probs.flatten()`` vs ``(targets>0.5).flatten()``，即**全部 128×128 像素**，
不过滤 SCL/无效像素；本项目主口径只统计 valid 像素。
直接拿官方表值减我们的主口径数值，会把口径差异算进「模型差距」。
本脚本用已保存的概率图，把两个口径都算出来，使与官方表值的对照尽量同口径。

用法::

    python scripts/metric_scope_check.py --pred <predictions_best_test.npz>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.training.segformer_metrics import pixel_average_precision  # noqa: E402


def metrics(prob: np.ndarray, gt: np.ndarray, scope: np.ndarray) -> dict:
    p = prob[scope] >= 0.5
    g = gt[scope]
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec and (prec + rec) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    ap = pixel_average_precision(prob[scope], gt[scope])
    return {"n_pixels": int(scope.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": iou, "dice": dice, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fpr, "ap": ap}


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--pred", required=True)
    ap_.add_argument("--out", default=None)
    args = ap_.parse_args()

    z = np.load(resolve_path(args.pred), allow_pickle=True)
    prob = z["probability"].astype(np.float32)
    gt = z["gt"].astype(bool)
    valid = z["valid"].astype(bool)

    all_scope = np.ones_like(gt, dtype=bool)
    res = {
        "pred": str(resolve_path(args.pred)).replace("\\", "/"),
        "n_tiles": int(prob.shape[0]),
        "valid_pixel_fraction": float(valid.mean()),
        "valid_only": metrics(prob, gt, valid),
        "all_pixels": metrics(prob, gt, all_scope),
    }
    # 逐瓦片池化（与全局池化略不同，用于检查是否受单块主导）
    per_tile = []
    for i in range(prob.shape[0]):
        m = metrics(prob[i], gt[i], valid[i])
        per_tile.append(m["iou"])
    vals = [v for v in per_tile if v is not None]
    res["valid_only"]["tile_mean_iou"] = float(np.mean(vals)) if vals else None
    res["valid_only"]["tile_median_iou"] = float(np.median(vals)) if vals else None

    print(json.dumps({k: (v if k != "pred" else Path(v).name) for k, v in res.items()},
                     indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

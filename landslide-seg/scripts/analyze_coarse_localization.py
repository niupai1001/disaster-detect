"""P0：粗定位独立评价（C 与 F 分开）。

对 CF0/CF1 的同一 best checkpoint（按最终 IoU 选出，不称其为粗任务最优）报告：
- C 与 F 的 IoU/Dice/P/R（图块宏平均与跨图汇总两种口径）、预测面积；
- C 的尺度分档对象召回；
- 粗概率 PR 曲线（阈值在 Hiroshima 上选定后冻结，Italy 仅诊断）；
- 5%/10%/20% 候选预算下的区域 Precision、GT 覆盖、对象触及率、对象内部覆盖比例。

用法::

    python scripts/analyze_coarse_localization.py --experiments CF0 CF1 --splits val test
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
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    evaluate_tile_v2,
    label_objects,
    summarize_tiles_v2,
)
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402

BUDGETS = (0.05, 0.10, 0.20)
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def _counts(prob, gt, valid, thr):
    pred = (prob >= thr) & valid
    g = gt & valid
    tp = int((pred & g).sum())
    fp = int((pred & ~g).sum())
    fn = int((~pred & g).sum())
    return tp, fp, fn, int(pred.sum())


def _tile_iou(t):
    d = t["tp"] + t["fp"] + t["fn"]
    return (t["tp"] / d) if d else None


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = (2 * p * r / (p + r)) if (p and r) else None
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return {"precision": p, "recall": r, "f1": f1, "iou": iou, "dice": dice}


def _budget_stats(prob, gt, valid, budget):
    v = np.asarray(valid, dtype=bool)
    g = np.asarray(gt, dtype=bool) & v
    p = np.asarray(prob, dtype=np.float32)
    n = int(v.sum())
    if n == 0:
        return None
    order = np.argsort(-p[v], kind="stable")
    k = max(1, int(round(n * budget)))
    cand = np.zeros(n, dtype=bool)
    cand[order[:k]] = True
    cf = np.zeros_like(v)
    cf[v] = cand
    gt_n = int(g.sum())
    inter = int((cf & g).sum())
    labels, n_obj = label_objects(g)
    hit = 0
    interior = []
    if n_obj:
        present = set(np.unique(labels[cf & g]).tolist())
        for o in range(1, n_obj + 1):
            m = labels == o
            ov = int((m & cf).sum())
            interior.append(ov / int(m.sum()))
            if o in present:
                hit += 1
    return {
        "regional_precision": (inter / k) if k else None,
        "gt_coverage": (inter / gt_n) if gt_n else None,
        "object_hit_rate": (hit / n_obj) if n_obj else None,
        "object_interior_coverage": float(np.mean(interior)) if interior else None,
        "n_objects": n_obj,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "coarse_to_fine" / "base.yaml"))
    ap.add_argument("--experiments", nargs="+", default=["CF0", "CF1"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_segformer_config(args.config)["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    records_all = read_manifest(resolve_path(cfg["data"]["manifest"]))
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 第一遍：收集粗概率曲线与候选预算，阈值只在 val 上选
    data = {}
    for exp in args.experiments:
        ckpt = root / exp / "1234" / "best.pt"
        if not ckpt.is_file():
            print(f"[skip] {ckpt}")
            continue
        model, meta = load_checkpoint(ckpt, device=str(device))
        for split in args.splits:
            recs = filter_split(records_all, split)
            ds = S2SegDataset(recs, stats, preload=True)
            tiles_c, tiles_f = [], []
            pr_counts = {t: [0, 0, 0] for t in THRESHOLDS}  # pooled tp/fp/fn
            budgets = {b: [] for b in BUDGETS}
            with torch.no_grad():
                for i in range(0, len(ds), args.batch_size):
                    b = collate([ds[j] for j in range(i, min(i + args.batch_size, len(ds)))])
                    out = model(b["image"].to(device))
                    c = torch.sigmoid(out["base_logits"].float()[:, 0]).cpu().numpy()
                    f = torch.sigmoid(out["logits"].float()[:, 0]).cpu().numpy()
                    gt = b["mask"].numpy().astype(bool)
                    valid = b["valid"].numpy().astype(bool)
                    for k, sid in enumerate(b["sample_id"]):
                        common = dict(
                            threshold=0.5, iou_threshold=0.25, iou_strict=0.5, boundary_tol=1,
                            sample_id=sid, event_id=b["event_id"][k],
                        )
                        tiles_c.append(evaluate_tile_v2(c[k], gt[k], valid[k], **common))
                        tiles_f.append(evaluate_tile_v2(f[k], gt[k], valid[k], **common))
                        for t in THRESHOLDS:
                            tp, fp, fn, _ = _counts(c[k], gt[k], valid[k], t)
                            pr_counts[t][0] += tp
                            pr_counts[t][1] += fp
                            pr_counts[t][2] += fn
                        for bud in BUDGETS:
                            s = _budget_stats(c[k], gt[k], valid[k], bud)
                            if s:
                                budgets[bud].append(s)
            data[(exp, split)] = {
                "tiles_c": tiles_c, "tiles_f": tiles_f,
                "pr_counts": pr_counts, "budgets": budgets,
                "mean_area": float(np.mean([t["pred_px"] for t in tiles_f])),
                "mean_area_c": float(np.mean([t["pred_px"] for t in tiles_c])),
            }
            del ds
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 阈值选择：在 val 上最大化粗像素 F1（若 val 未评则用 0.5）
    best_thr = 0.5
    for exp in args.experiments:
        key = (exp, "val")
        if key not in data:
            continue
        best = (-1.0, 0.5)
        for t, (tp, fp, fn) in data[key]["pr_counts"].items():
            f1 = _prf(tp, fp, fn)["f1"]
            if f1 is not None and f1 > best[0]:
                best = (f1, float(t))
        best_thr = best[1] if best[0] >= 0 else 0.5
        print(f"[threshold] {exp} 在 val 上选 th={best_thr} (coarse F1={best[0]:.4f})")
        break  # 部署冻结阈值取同一值，报告里注明

    report = {"frozen_threshold": best_thr, "threshold_source": "val (Hiroshima), coarse pixel F1", "experiments": {}}
    for exp in args.experiments:
        report["experiments"][exp] = {}
        for split in args.splits:
            key = (exp, split)
            if key not in data:
                continue
            d = data[key]
            fc = summarize_tiles_v2(d["tiles_c"])
            ff = summarize_tiles_v2(d["tiles_f"])
            rows = {}
            for name, s, area in (("coarse", fc, d["mean_area_c"]), ("final", ff, d["mean_area"])):
                rows[name] = {
                    "pooled": {k: s["pixel"][k] for k in ("iou", "dice", "precision", "recall")},
                    "macro": {
                        "iou": float(np.mean([_tile_iou(t) for t in (d["tiles_c"] if name == "coarse" else d["tiles_f"]) if _tile_iou(t) is not None])),
                    },
                    "pred_px_mean": area,
                    "bins": s["objects"]["bins"],
                    "tiny": s["objects"]["tiny"],
                }
            pr = {
                str(t): _prf(tp, fp, fn)
                for t, (tp, fp, fn) in d["pr_counts"].items()
            }
            buds = {}
            for bud in BUDGETS:
                items = d["budgets"][bud]
                buds[f"{int(bud*100)}pct"] = {
                    k: (float(np.mean([x[k] for x in items if x[k] is not None])) if any(x[k] is not None for x in items) else None)
                    for k in ("regional_precision", "gt_coverage", "object_hit_rate", "object_interior_coverage")
                }
                buds[f"{int(bud*100)}pct"]["n_tiles"] = len(items)
            report["experiments"][exp][split] = {"coarse": rows["coarse"], "final": rows["final"], "pr_curve": pr, "budgets": buds}

    out_dir = resolve_path("outputs/segformer/coarse_localization")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coarse_localization.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["# 粗定位独立评价（C 与 F 分开）", "", f"冻结阈值：{best_thr}（来源：val 粗像素 F1）", ""]
    for exp, splits in report["experiments"].items():
        lines += [f"## {exp}", "", "| split | 输出 | IoU(pooled) | Dice | P | R | 预测面积(px) |", "|---|---|---|---|---|---|---|"]
        for split, r in splits.items():
            for name in ("coarse", "final"):
                p = r[name]["pooled"]
                lines.append(f"| {split} | {name} | {p['iou']:.4f} | {p['dice']:.4f} | {p['precision']:.4f} | {p['recall']:.4f} | {r[name]['pred_px_mean']:.1f} |")
        lines += ["", "| split | 预算 | 区域P | GT覆盖 | 对象触及 | 对象内部覆盖 |", "|---|---|---|---|---|---|"]
        for split, r in splits.items():
            for b, v in r["budgets"].items():
                lines.append(f"| {split} | {b} | {v['regional_precision']:.4f} | {v['gt_coverage']:.4f} | {v['object_hit_rate']:.4f} | {v['object_interior_coverage']:.4f} |")
        lines.append("")
    (out_dir / "coarse_localization.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

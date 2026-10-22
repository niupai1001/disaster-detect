"""主模型（单期/多期 × 光谱）的统一评测：best/last × VAL/TEST。

指标口径与主评测一致（``segformer_metrics`` v2）：
pooled IoU/Dice/Precision/Recall、AP、逐事件、对象召回（IoU≥0.25 / 严格≥0.5）、
可评对象、尺度分档、1 px 容差边界 F1、负瓦片与逐图块 CSV。
另外保留“GT 连通域内任意预测像素命中率”作为附加诊断，不冒充对象召回。

用法::

    python scripts/evaluate_temporal.py --checkpoint outputs/.../best.pt --splits val,test --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import filter_split, load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.spectral_temporal import build_temporal_model  # noqa: E402
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    evaluate_tile_v2, label_objects, pixel_average_precision,
    summarize_by_event_v2, summarize_tiles_v2,
)


def _rounded(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


def _any_hit(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict:
    """附加诊断：GT 连通域内是否存在任一预测像素（旧口径，不是对象召回）。"""
    lab, n = label_objects(gt & valid, 2)
    if n == 0:
        return {"n_gt_objects": 0, "n_any_hit": 0, "any_pixel_hit_rate": None}
    hit = np.unique(lab[pred & valid])
    hit = hit[hit > 0]
    return {"n_gt_objects": int(n), "n_any_hit": int(len(hit)),
            "any_pixel_hit_rate": float(len(hit) / n)}


@torch.no_grad()
def evaluate_split(model, ds, cfg, device, *, split: str, batch_size: int | None = None,
                   threshold: float = 0.5) -> tuple[dict, list[dict], dict]:
    s = cfg["segformer"]
    bsz = int(batch_size or s["eval"].get("batch_size", 8))
    scale_ha = s["eval"].get("scale_thresholds_ha", [0.5, 2.0])
    thresholds_px = [float(x) * 100.0 for x in scale_ha]
    rows: list[dict] = []
    probs, gts, valids, full_probs, full_gts, ids, events = [], [], [], [], [], [], []
    model.eval()
    for start in range(0, len(ds), bsz):
        chunk = list(range(start, min(start + bsz, len(ds))))
        batch = collate_v6([ds[i] for i in chunk])
        logits = model(batch["image"].to(device), batch["dates"].to(device),
                       batch["keep"].to(device), batch["anchor"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        valid = batch["valid"].numpy().astype(bool)
        for i, sid in enumerate(batch["sample_id"]):
            r = evaluate_tile_v2(
                prob[i], gt[i], valid[i], threshold=threshold, thresholds_px=thresholds_px,
                iou_threshold=float(s["eval"]["object_iou_threshold"]),
                iou_strict=float(s["eval"]["object_iou_strict"]),
                boundary_tol=int(s["eval"]["boundary_tol_px"]),
                sample_id=sid, event_id=batch["event_id"][i],
            )
            r["any_pixel_hit"] = _any_hit(gt[i], prob[i] >= threshold, valid[i])
            rows.append(r)
            probs.append(prob[i][valid[i]])
            gts.append(gt[i][valid[i]])
            valids.append(valid[i])
            full_probs.append(prob[i])
            full_gts.append(gt[i])
            ids.append(sid)
            events.append(batch["event_id"][i])
    overall = summarize_tiles_v2(rows)
    overall["pixel"]["average_precision"] = float(pixel_average_precision(
        np.concatenate(probs), np.concatenate(gts)))
    by_event = summarize_by_event_v2(rows)
    for ev in by_event:
        mask = [i for i, e in enumerate(events) if e == ev]
        pp = np.concatenate([probs[i] for i in mask])
        gg = np.concatenate([gts[i] for i in mask])
        by_event[ev]["pixel"]["average_precision"] = float(pixel_average_precision(pp, gg))
    # 全像素口径仅作历史对照，不用于选模
    all_counts = []
    for i in range(len(rows)):
        p = (np.asarray(full_probs[i]) >= threshold)
        g = np.asarray(full_gts[i]).astype(bool)
        all_counts.append({"tp": int((p & g).sum()), "fp": int((p & ~g).sum()),
                           "fn": int((~p & g).sum()), "tn": int((~p & ~g).sum())})
    tp = sum(c["tp"] for c in all_counts); fp = sum(c["fp"] for c in all_counts)
    fn = sum(c["fn"] for c in all_counts)
    overall["all_pixels"] = {"iou": tp / (tp + fp + fn) if tp + fp + fn else None}
    preds = {
        "sample_ids": np.asarray(ids), "events": np.asarray(events),
        "probability": np.stack(full_probs), "gt": np.stack(full_gts),
        "valid": np.stack(valids),
    }
    return overall, rows, {"by_event": by_event, "predictions": preds}


def _latency(model, cfg, device, iters: int = 20) -> dict:
    mode = str(cfg["segformer"]["temporal"]["mode"])
    t = 1 if mode == "single" else 15
    x = torch.randn(1, t, 10, 128, 128, device=device)
    dates = torch.arange(t, device=device, dtype=torch.float32)[None] + 1500
    keep = torch.ones(1, t, dtype=torch.bool, device=device)
    anchor = torch.zeros(1, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x, dates, keep, anchor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x, dates, keep, anchor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    return {"mode": mode, "frames": t, "batch": 1,
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "mean_ms": round(float(np.mean(lat)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag-prefix", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = resolve_path(ckpt_path)
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    s = cfg["segformer"]
    stats = load_stats(resolve_path(s["data"]["stats"]))
    model = build_temporal_model(s, stats)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    mode = str(s["temporal"]["mode"])
    threshold = float(s["eval"]["threshold"])

    trainval = read_manifest(resolve_path(s["data"]["manifest"]))
    test_rows = read_manifest(resolve_path(s["data"]["test_manifest"]))
    protocol_path = resolve_path(s["data"]["protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    want = {"val": "val", "test": "test"}
    run_dir = ckpt_path.parent
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.tag_prefix or ckpt_path.stem
    report = {"checkpoint": str(ckpt_path).replace("\\", "/"),
              "checkpoint_sha256": file_sha256(ckpt_path),
              "mode": mode, "threshold": threshold,
              "role": {"val": "DEV", "test": "DEVELOPMENT_BENCHMARK"},
              "metric_version": "v2"}

    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        rows = filter_split(trainval, split) if split == "val" else test_rows
        if split == "test":
            rows = [r for r in test_rows if r["split"] == "test"]
        if not rows:
            raise RuntimeError(f"split={split} 为空")
        # 协议成员校验：身份/缓存/日期/时间帧必须与锁定协议一致
        for rec in rows:
            member = protocol["members"].get(rec["sample_id"])
            if member is None:
                raise ValueError(f"协议缺少成员 {rec['sample_id']}")
            for k in ("event_id", "cache", "cache_sha256", "date", "time_index"):
                if rec.get(k) != member.get(k):
                    raise ValueError(f"{rec['sample_id']} 的 {k} 与协议不一致")
            if member["role"] != want[split]:
                raise ValueError(f"{rec['sample_id']} 角色 {member['role']} != {split}")
        ds = TemporalV6(rows, stats, mode=mode, augment=False, seed=int(s["seed"]),
                        root=s["data"]["temporal_root"], quality=s["data"]["quality"])
        t0 = time.time()
        overall, tile_rows, extra = evaluate_split(model, ds, cfg, dev, split=split,
                                                   batch_size=args.batch_size, threshold=threshold)
        by_event = extra["by_event"]
        report[split] = {"n_samples": len(rows), "overall": overall, "by_event": by_event,
                         "seconds": round(time.time() - t0, 2)}
        if split == "val":
            report["val_event_mean_iou"] = float(np.mean(
                [v["pixel"]["iou"] for v in by_event.values()
                 if v["pixel"]["iou"] is not None and np.isfinite(v["pixel"]["iou"])]))
        # 逐图块 CSV
        per_tile = []
        for r in tile_rows:
            b = r["boundary"]
            if b["status"] == "both_empty":
                bf1 = None
            elif b["gt_px"] == 0 or b["pred_px"] == 0:
                bf1 = 0.0
            else:
                p = b["prec_tp"] / b["pred_px"]; rc = b["rec_tp"] / b["gt_px"]
                bf1 = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
            per_tile.append({
                "sample_id": r["sample_id"], "event_id": r["event_id"],
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
                "gt_px": r["gt_px"], "pred_px": r["pred_px"], "valid_px": r["valid_px"],
                "iou": _rounded(r["tp"] / max(r["tp"] + r["fp"] + r["fn"], 1)),
                "precision": _rounded(r["tp"] / max(r["tp"] + r["fp"], 1)),
                "recall": _rounded(r["tp"] / max(r["tp"] + r["fn"], 1)),
                "n_gt_objects": r["n_gt_objects"],
                "n_gt_objects_evaluable": r["n_gt_objects_evaluable"],
                "n_matched_evaluable": r["n_matched_evaluable"],
                "n_matched_strict_evaluable": r["n_matched_strict_evaluable"],
                "boundary_f1": _rounded(bf1),
                "any_pixel_hit_rate": r["any_pixel_hit"]["any_pixel_hit_rate"],
            })
        csv_path = out_dir / f"per_tile_{prefix}_{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_tile[0].keys()))
            writer.writeheader()
            writer.writerows(per_tile)
        (out_dir / f"objects_{prefix}_{split}.json").write_text(
            json.dumps([{k: r[k] for k in ("sample_id", "event_id", "bins", "tiny", "fine_bins")}
                        for r in tile_rows], indent=1), encoding="utf-8")
        if args.save_predictions:
            np.savez_compressed(out_dir / f"predictions_{prefix}_{split}.npz", **extra["predictions"])
        print(f"[{mode}] {prefix}_{split}: n={len(rows)} "
              f"pooled IoU={overall['pixel']['iou']:.4f} AP={overall['pixel']['average_precision']:.4f} "
              f"objRec={overall['objects']['object_recall_evaluable']:.4f} "
              f"bF1={overall['boundary']['f1']:.4f}")

    report["efficiency"] = {"params": ckpt.get("params"),
                            "latency_b1": _latency(model, cfg, dev),
                            "train_seconds": ckpt.get("elapsed_seconds")}
    (out_dir / f"eval_{prefix}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_event",)}, indent=1,
                     ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()

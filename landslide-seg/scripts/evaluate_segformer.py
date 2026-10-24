"""评测入口：加载 best/last，按 split 输出像素 / 尺度 / 对象 / 边界 / 负瓦片指标。

产物（写到 ``<run_dir>/eval/``）：
- ``eval_<split>.json``：总体 + 逐事件汇总 + 效率数据；
- ``per_tile_<split>.csv``：逐瓦片混淆计数与派生指标；
- ``figures/*.png``（可选）：RGB / GT / 概率 / 预测 / TP-FP-FN。

用法::

    python scripts/evaluate_segformer.py --checkpoint outputs/segformer/s2_10band_v1/A0/1234/best.pt --split test
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

try:  # Windows 控制台/管道默认 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import resolve_path, file_sha256  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    evaluate_tile,
    summarize_by_event,
    summarize_tiles,
    evaluate_tile_v2,
    summarize_tiles_v2,
    summarize_by_event_v2,
    pixel_average_precision,
)
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def _safe(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def _rounded(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 6)


def _rgb(x10: np.ndarray) -> np.ndarray:
    rgb = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _latency(model, device, iters: int = 100, in_chans: int = 10) -> dict:
    x = torch.randn(1, int(in_chans), 128, 128, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    return {
        "batch": 1,
        "size": [128, 128],
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "mean_ms": round(float(np.mean(lat)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument('--band-subset', default='full', choices=['full','rgb','rgb_nir','six','unseen5'])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--figures", type=int, default=0, help="额外输出多少张案例图")
    ap.add_argument("--manifest", default=None, help="覆盖清单路径（用于 DEV/FINAL/DIAG 分事件评测）")
    ap.add_argument("--tag", default=None, help="输出文件后缀，默认等于 --split；用于避免多次评测互相覆盖")
    ap.add_argument("--skip-efficiency", action="store_true")
    ap.add_argument("--save-predictions", action="store_true", help="保存有效区、GT与概率，供失败案例复查")
    ap.add_argument("--role", default=None)
    ap.add_argument("--allow-frozen-final", action="store_true", help="仅在研究决策冻结后显式评FINAL")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = resolve_path(ckpt_path)
    model, ckpt = load_checkpoint(ckpt_path, device=args.device)
    cfg = ckpt["config"]["segformer"]
    device = next(model.parameters()).device

    threshold = float(args.threshold if args.threshold is not None else cfg["eval"]["threshold"])
    scale_ha = cfg["eval"].get("scale_thresholds_ha", [0.5, 2.0])
    thresholds_px = [float(x) * 100.0 for x in scale_ha]  # 1 ha = 10000 m²，1 px = 100 m²

    manifest = resolve_path(args.manifest) if args.manifest else resolve_path(cfg["data"]["manifest"])
    tag = args.tag or args.split
    stats_path = resolve_path(cfg["data"]["stats"])
    expected_stats = ckpt.get("meta", {}).get("stats_sha256")
    if expected_stats and file_sha256(stats_path) != expected_stats:
        raise ValueError("Normalization statistics differ from training checkpoint")
    records = filter_split(read_manifest(manifest), args.split)
    if not records:
        raise RuntimeError(f"split={args.split} 为空")
    protocol_path = cfg["data"].get("protocol")
    if protocol_path:
        protocol = json.loads(resolve_path(protocol_path).read_text(encoding="utf-8"))
        for rec in records:
            member = protocol["members"].get(rec["sample_id"])
            if member is None or any(rec.get(k) != member.get(k) for k in ("event_id", "cache", "cache_sha256", "date", "time_index")):
                raise ValueError("Unknown or modified evaluation member")
            if member["role"] == "final" and not args.allow_frozen_final:
                raise ValueError("FINAL is frozen during development")
    stats = ckpt.get("stats") or load_stats(stats_path)
    metric_tile = evaluate_tile_v2 if protocol_path else evaluate_tile
    metric_summary = summarize_tiles_v2 if protocol_path else summarize_tiles
    event_summary = summarize_by_event_v2 if protocol_path else summarize_by_event
    dataset_class = S2SegDataset
    if cfg['model'].get('type') == 'use':
        from landslide_seg.data.use_dataset import USESegDataset
        dataset_class = USESegDataset
    ds = dataset_class(records, stats, augment=False, preload=True,
                      extra=str(cfg["model"].get("extra_channel", "none")),
                      sidecar=cfg["data"].get("extra_sidecar"))
    if cfg['model'].get('type') == 'use':
        ds.eval_subset = args.band_subset
        quality_hash = ckpt.get('meta', {}).get('use_quality_sha256')
        if quality_hash is not None and file_sha256(ds.quality_root / 'index.json') != quality_hash:
            raise ValueError('USE input quality index differs from checkpoint')
        if not args.skip_efficiency:
            raise ValueError('USE效率需实际观测对象；请显式--skip-efficiency，训练峰值另报')
    elif args.band_subset != 'full':
        raise ValueError('Variable band subsets require USE')

    rows: list[dict] = []
    ap_parts = {}
    predictions, truths, validity = [], [], []
    batch_size = 8 if cfg['model'].get('type') == 'use' else 32
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch = collate([ds[i] for i in range(start, min(start + batch_size, len(ds)))])
            image = batch["image"].to(device)
            logits = model(image)["logits"].float()
            prob = torch.sigmoid(logits[:, 0]).cpu().numpy()
            gt = batch["mask"].numpy().astype(bool)
            valid = batch["valid"].numpy().astype(bool)
            for i, sid in enumerate(batch["sample_id"]):
                rows.append(
                    metric_tile(
                        prob[i], gt[i], valid[i],
                        threshold=threshold,
                        thresholds_px=thresholds_px,
                        iou_threshold=float(cfg["eval"]["object_iou_threshold"]),
                        iou_strict=float(cfg["eval"]["object_iou_strict"]),
                        boundary_tol=int(cfg["eval"]["boundary_tol_px"]),
                        sample_id=sid,
                        event_id=batch["event_id"][i],
                    )
                )
                ev = batch["event_id"][i]
                ap_parts.setdefault(ev, [[], []])
                ap_parts[ev][0].append(prob[i][valid[i]])
                ap_parts[ev][1].append(gt[i][valid[i]])
            if args.save_predictions:
                predictions.append(prob)
                truths.append(gt)
                validity.append(valid)

    overall = metric_summary(rows)
    by_event = event_summary(rows)
    if protocol_path:
        for summary in [overall, *by_event.values()]:
            summary["negative_tiles"] = {
                "status": "not_independently_verified", "tile_false_alarm_rate": None,
                "label_empty_tile_counts": summary["negative_tiles"],
            }
    event_prob, event_gt = [], []
    for ev, (ps, ys) in ap_parts.items():
        pp, yy = np.concatenate(ps), np.concatenate(ys)
        by_event[ev]["pixel"]["average_precision"] = pixel_average_precision(pp, yy)
        event_prob.append(pp)
        event_gt.append(yy)
    overall["pixel"]["average_precision"] = pixel_average_precision(np.concatenate(event_prob), np.concatenate(event_gt))
    ap_parts.clear()

    run_dir = ckpt_path.parent
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_tile = []
    for r in rows:
        denom = r["tp"] + r["fp"] + r["fn"]
        b = r["boundary"]
        if b["status"] == "both_empty":
            b_f1 = None
        elif b["gt_px"] == 0 or b["pred_px"] == 0:
            b_f1 = 0.0
        else:
            p = b["prec_tp"] / b["pred_px"]
            rc = b["rec_tp"] / b["gt_px"]
            b_f1 = (2 * p * rc / (p + rc)) if (p + rc) > 0 else 0.0
        per_tile.append(
            {
                "sample_id": r["sample_id"],
                "event_id": r["event_id"],
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
                "gt_px": r["gt_px"], "pred_px": r["pred_px"], "valid_px": r["valid_px"],
                "iou": _rounded(_safe(r["tp"], denom)),
                "dice": _rounded(_safe(2 * r["tp"], 2 * r["tp"] + r["fp"] + r["fn"])),
                "precision": _rounded(_safe(r["tp"], r["tp"] + r["fp"])),
                "recall": _rounded(_safe(r["tp"], r["tp"] + r["fn"])),
                "n_gt_objects": r["n_gt_objects"],
                "n_gt_objects_evaluable": r["n_gt_objects_evaluable"],
                "n_truncated_objects": r["n_truncated_objects"],
                "n_matched": r.get("n_matched", r.get("n_matched_all")),
                "n_matched_evaluable": r["n_matched_evaluable"],
                "boundary_f1": None if b_f1 is None else round(float(b_f1), 6),
                "is_negative_tile": r["is_negative_tile"],
                "pred_positive_on_negative": r["pred_positive_on_negative"],
            }
        )
    csv_path = out_dir / f"per_tile_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_tile[0].keys()))
        writer.writeheader()
        writer.writerows(per_tile)

    eff = {
        "params": ckpt.get("params"),
        "checkpoint": str(ckpt_path).replace("\\", "/"),
        "threshold": threshold,
        "scale_thresholds_px": thresholds_px,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "latency_b1": None if args.skip_efficiency else _latency(
            model, device, in_chans=int(cfg["model"].get("in_chans", 10))),
    }
    if device.type == "cuda" and not args.skip_efficiency:
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(torch.randn(1, int(cfg["model"].get("in_chans", 10)), 128, 128, device=device))
        eff["inference_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        eff["inference_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 1)

    event_ious = [v["pixel"]["iou"] for v in by_event.values() if v["pixel"]["iou"] is not None and np.isfinite(v["pixel"]["iou"])]
    report = {
        'band_subset': args.band_subset,
        "checkpoint": str(ckpt_path).replace("\\", "/"),
        "split": tag,
        "threshold": threshold,
        "n_samples": len(rows),
        "overall": overall,
        "by_event": by_event,
        "role": args.role or {"train": "TRAIN", "val": "DEV", "test": "UNSPECIFIED_EVALUATION"}[args.split],
        "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "stats_sha256": file_sha256(stats_path), "checkpoint_sha256": file_sha256(ckpt_path),
        "metric_version": overall.get("metric_version", "v1"),
        "event_mean_iou": float(np.mean(event_ious)) if event_ious else None,
        "efficiency": eff,
        "note": "test split 不参与任何调参/阈值/checkpoint 选择；阈值扫描只在 val 上做",
    }
    json_path = out_dir / f"eval_{tag}.json"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.save_predictions:
        np.savez_compressed(out_dir / f"predictions_{tag}.npz", sample_ids=np.asarray([r["sample_id"] for r in rows]),
                            probability=np.concatenate(predictions), gt=np.concatenate(truths), valid=np.concatenate(validity))
    (out_dir / f"objects_{tag}.json").write_text(json.dumps([{k: r[k] for k in ("sample_id", "event_id", "bins", "tiny", "fine_bins", "use_bins") if k in r} for r in rows], indent=1), encoding="utf-8")

    # ---- 可选案例图 ----
    if args.figures > 0:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            picks = rows[: args.figures]
            fig, axes = plt.subplots(len(picks), 5, figsize=(15, 3 * len(picks)), squeeze=False)
            for i, r in enumerate(picks):
                rec = next(x for x in records if x["sample_id"] == r["sample_id"])
                raw = np.load(rec["cache"])["x10"].astype(np.float32)
                # 重新前向一次拿概率
                b = collate([S2SegDataset([rec], stats, preload=False)[0]])
                with torch.no_grad():
                    p = torch.sigmoid(model(b["image"].to(device))["logits"].float()[0, 0]).cpu().numpy()
                gt = b["mask"].numpy()[0].astype(bool)
                v = b["valid"].numpy()[0].astype(bool)
                pred = (p >= threshold) & v
                err = np.zeros((*gt.shape, 3), dtype=np.float32)
                err[..., 1] = (gt & pred & v).astype(np.float32)
                err[..., 0] = ((~gt) & pred & v).astype(np.float32)
                err[..., 2] = (gt & (~pred) & v).astype(np.float32)
                panels = [
                    (_rgb(raw), f"{r['sample_id']} RGB"),
                    (gt, "GT"),
                    (p, "prob"),
                    (pred, "pred"),
                    (err, "TP / FP / FN"),
                ]
                for j, (img, title) in enumerate(panels):
                    ax = axes[i][j]
                    if img.ndim == 3:
                        ax.imshow(np.clip(img, 0, 1))
                    else:
                        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(title, fontsize=9)
                    ax.axis("off")
            fig.tight_layout()
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(exist_ok=True)
            fig.savefig(fig_dir / f"cases_{tag}.png", dpi=110)
            plt.close(fig)
        except Exception as exc:
            report["figure_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[saved] {json_path}\n[saved] {csv_path}")


if __name__ == "__main__":
    main()

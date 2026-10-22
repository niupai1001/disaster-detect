"""U-TAE 真实日期版统一评测：best/last × VAL/TEST，指标与主评测同口径（v2）。

附加输出旧“GT 连通域任意像素命中率”作为诊断，不当作对象召回。

用法::

    python scripts/evaluate_utae_v6.py --mode multi --checkpoints best,last --device cuda:1
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
from torch.utils.data import DataLoader  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import filter_split, load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    evaluate_tile_v2, label_objects, pixel_average_precision,
    summarize_by_event_v2, summarize_tiles_v2,
)

STATS = "manifests/official_single_geo/stats_dem.json"
TRAINVAL = "manifests/official_single/trainval.jsonl"
TEST = "manifests/official_single/test.jsonl"
OUT_ROOT = "outputs/utae_v6"
THRESH = 0.5


def _rounded(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


def _any_hit(gt, pred, valid):
    lab, n = label_objects(gt & valid, 2)
    if n == 0:
        return {"n_gt_objects": 0, "n_any_hit": 0, "any_pixel_hit_rate": None}
    hit = np.unique(lab[pred & valid]); hit = hit[hit > 0]
    return {"n_gt_objects": int(n), "n_any_hit": int(len(hit)),
            "any_pixel_hit_rate": float(len(hit) / n)}


@torch.no_grad()
def run_split(model, rows, stats, mode, device) -> tuple[dict, list[dict], dict]:
    ds = TemporalV6(rows, stats, mode=mode, augment=False, seed=1234, dem=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_v6)
    tile_rows, probs, gts, valids, full_p, full_g, ids, events = [], [], [], [], [], [], [], []
    model.eval()
    for batch in loader:
        logits = model(batch["image"].to(device), batch["dates"].to(device),
                       batch["keep"].to(device), batch["anchor"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = batch["mask"].numpy().astype(bool)
        va = batch["valid"].numpy().astype(bool)
        for i, sid in enumerate(batch["sample_id"]):
            pred = prob[i] >= THRESH
            r = evaluate_tile_v2(prob[i], gt[i], va[i], threshold=THRESH,
                                 thresholds_px=(50.0, 200.0), iou_threshold=0.25,
                                 iou_strict=0.5, boundary_tol=1,
                                 sample_id=sid, event_id=batch["event_id"][i])
            r["any_pixel_hit"] = _any_hit(gt[i], pred, va[i])
            tile_rows.append(r)
            probs.append(prob[i][va[i]]); gts.append(gt[i][va[i]]); valids.append(va[i])
            full_p.append(prob[i]); full_g.append(gt[i])
            ids.append(sid); events.append(batch["event_id"][i])
    overall = summarize_tiles_v2(tile_rows)
    overall["pixel"]["average_precision"] = float(pixel_average_precision(
        np.concatenate(probs), np.concatenate(gts)))
    by_event = summarize_by_event_v2(tile_rows)
    for ev in by_event:
        ix = [i for i, e in enumerate(events) if e == ev]
        by_event[ev]["pixel"]["average_precision"] = float(pixel_average_precision(
            np.concatenate([probs[i] for i in ix]), np.concatenate([gts[i] for i in ix])))
    # 旧全像素口径（诊断，不用于选模）
    tp = fp = fn = 0
    for p, g in zip(full_p, full_g):
        q = p >= THRESH
        tp += int((q & g).sum()); fp += int((q & ~g).sum()); fn += int((~q & g).sum())
    overall["all_pixels"] = {"iou": tp / (tp + fp + fn) if tp + fp + fn else None}
    preds = {"sample_ids": np.asarray(ids), "events": np.asarray(events),
             "probability": np.stack(full_p), "gt": np.stack(full_g),
             "valid": np.stack(valids)}
    return overall, tile_rows, {"by_event": by_event, "predictions": preds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "multi", "history"), required=True)
    ap.add_argument("--checkpoints", default="best,last")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    run_dir = ROOT / OUT_ROOT / f"utae_{args.mode}_realdates" / "1234"
    stats = load_stats(resolve_path(STATS))
    rows = read_manifest(resolve_path(TRAINVAL))
    val = filter_split(rows, "val")
    test = read_manifest(resolve_path(TEST))
    protocol = json.loads(resolve_path("manifests/official_single/protocol.json").read_text(encoding="utf-8"))
    model = UTAERealDates(in_channels=11, num_classes=1).to(dev)
    out = {}
    (run_dir / "eval").mkdir(exist_ok=True)
    for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        p = run_dir / f"{ck}.pt"
        if not p.is_file():
            print(f"[skip] {p}")
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"]); model.eval()
        out[ck] = {}
        for split, split_rows in (("val", val), ("test", test)):
            for rec in split_rows:
                member = protocol["members"].get(rec["sample_id"])
                if member is None or member["role"] != split:
                    raise ValueError(f"{rec['sample_id']} 协议角色不符")
            t0 = time.time()
            overall, tile_rows, extra = run_split(model, split_rows, stats, args.mode, dev)
            tag = f"{ck}_{split}"
            out[ck][split] = {"n_samples": len(split_rows), "overall": overall,
                              "by_event": extra["by_event"],
                              "seconds": round(time.time() - t0, 2),
                              "checkpoint_sha256": file_sha256(p)}
            per_tile = []
            for r in tile_rows:
                b = r["boundary"]
                if b["status"] == "both_empty":
                    bf1 = None
                elif b["gt_px"] == 0 or b["pred_px"] == 0:
                    bf1 = 0.0
                else:
                    pre = b["prec_tp"] / b["pred_px"]; rec = b["rec_tp"] / b["gt_px"]
                    bf1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0
                per_tile.append({
                    "sample_id": r["sample_id"], "event_id": r["event_id"],
                    "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
                    "gt_px": r["gt_px"], "pred_px": r["pred_px"], "valid_px": r["valid_px"],
                    "iou": _rounded(r["tp"] / max(r["tp"] + r["fp"] + r["fn"], 1)),
                    "n_gt_objects": r["n_gt_objects"],
                    "n_gt_objects_evaluable": r["n_gt_objects_evaluable"],
                    "n_matched_evaluable": r["n_matched_evaluable"],
                    "n_matched_strict_evaluable": r["n_matched_strict_evaluable"],
                    "boundary_f1": _rounded(bf1),
                    "any_pixel_hit_rate": r["any_pixel_hit"]["any_pixel_hit_rate"],
                })
            with (run_dir / "eval" / f"per_tile_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(per_tile[0].keys()))
                w.writeheader(); w.writerows(per_tile)
            (run_dir / "eval" / f"objects_{tag}.json").write_text(
                json.dumps([{k: r[k] for k in ("sample_id", "event_id", "bins", "tiny", "fine_bins")}
                            for r in tile_rows], indent=1), encoding="utf-8")
            if args.save_predictions:
                np.savez_compressed(run_dir / "eval" / f"predictions_{tag}.npz", **extra["predictions"])
            (run_dir / "eval" / f"eval_{tag}.json").write_text(
                json.dumps(out[ck][split], indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[{args.mode}] {tag}: pooled={overall['pixel']['iou']:.4f} "
                  f"AP={overall['pixel']['average_precision']:.4f} "
                  f"objRec={overall['objects']['object_recall_evaluable']:.4f} "
                  f"bF1={overall['boundary']['f1']:.4f} "
                  f"anyHit={np.mean([r['any_pixel_hit']['any_pixel_hit_rate'] for r in tile_rows]):.4f}")
    (run_dir / "eval_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()

"""在项目数据划分（event_ood_v4r1）上评测 U-TAE：DEV 全指标，口径与 SegFormer 主评测一致。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    evaluate_tile_v2, pixel_average_precision, summarize_by_event_v2, summarize_tiles_v2,
)

from train_utae_cv import load_cv_config  # noqa: E402

THRESH = 0.5


@torch.no_grad()
def run_split(model, rows, stats, mode, device, root, quality, n_frames=None):
    ds = TemporalV6(rows, stats, mode=mode, augment=False, seed=1234, dem=False,
                    n_frames=n_frames, root=root, quality=quality)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_v6)
    tiles, probs, gts, ids, events = [], [], [], [], []
    model.eval()
    for b in loader:
        logits = model(b["image"].to(device), b["dates"].to(device), b["keep"].to(device),
                       b["anchor"].to(device))["logits"].float()
        prob = logits[:, 0].sigmoid().cpu().numpy()
        gt = b["mask"].numpy().astype(bool)
        va = b["valid"].numpy().astype(bool)
        for i, sid in enumerate(b["sample_id"]):
            r = evaluate_tile_v2(prob[i], gt[i], va[i], threshold=THRESH,
                                 thresholds_px=(50.0, 200.0), iou_threshold=0.25,
                                 iou_strict=0.5, boundary_tol=1,
                                 sample_id=sid, event_id=b["event_id"][i])
            tiles.append(r)
            probs.append(prob[i][va[i]])
            gts.append(gt[i][va[i]])
            ids.append(sid)
            events.append(b["event_id"][i])
    overall = summarize_tiles_v2(tiles)
    overall["pixel"]["average_precision"] = float(pixel_average_precision(
        np.concatenate(probs), np.concatenate(gts)))
    by_event = summarize_by_event_v2(tiles)
    for ev in by_event:
        ix = [i for i, e in enumerate(events) if e == ev]
        by_event[ev]["pixel"]["average_precision"] = float(pixel_average_precision(
            np.concatenate([probs[i] for i in ix]), np.concatenate([gts[i] for i in ix])))
    return overall, by_event, tiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=("single", "multi", "history"))
    ap.add_argument("--checkpoints", default="best,last")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg = load_cv_config(resolve_path(args.config))
    prot = cfg["protocol"]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / cfg["output_dir"] / str(cfg["fold"]) / args.mode / str(cfg["seed"])
    rows = [r for r in read_manifest(resolve_path(prot["pool"])) if r["split"] == "val"]
    stats = load_stats(resolve_path(prot["stats"]))
    model = UTAERealDates(in_channels=int(cfg["data"]["in_channels"]), num_classes=1,
                          positional=str(cfg.get("model", {}).get("positional", "dates")),
                          aggregation=str(cfg.get("model", {}).get("aggregation", "attention"))).to(dev)
    out = {}
    for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        p = run_dir / f"{ck}.pt"
        if not p.is_file():
            print(f"[skip] {p}")
            continue
        state = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        t0 = time.time()
        overall, by_event, tiles = run_split(model, rows, stats, args.mode, dev,
                                             prot.get("temporal_root", "data/s2_temporal_v5"),
                                             prot.get("quality") or "data/s2_temporal_v4r1_none",
                                             n_frames=cfg.get("temporal", {}).get("n_frames"))
        px = overall["pixel"]
        out[ck] = {"n_samples": len(rows), "overall": overall, "by_event": by_event,
                   "epoch": int(state["epoch"]), "seconds": round(time.time() - t0, 1),
                   "checkpoint_sha256": file_sha256(p)}
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval" / f"eval_{ck}_dev.json").write_text(
            json.dumps(out[ck], indent=1, ensure_ascii=False), encoding="utf-8")
        per_tile = [{"sample_id": r["sample_id"], "event_id": r["event_id"],
                     "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
                     "n_gt_objects": r["n_gt_objects"], "n_matched_evaluable": r["n_matched_evaluable"]}
                    for r in tiles]
        with (run_dir / "eval" / f"per_tile_{ck}_dev.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_tile[0].keys()))
            w.writeheader()
            w.writerows(per_tile)
        print(f"[{args.mode}/{ck}] epoch {state['epoch']} IoU {px['iou']:.4f} AP "
              f"{px['average_precision']:.4f} P {px['precision']:.4f} R {px['recall']:.4f} "
              f"objRec {overall['objects']['object_recall_evaluable']:.4f} "
              f"bF1 {overall['boundary']['f1']:.4f}", flush=True)
    (run_dir / "eval_dev_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()

"""四折事件协议 · U-TAE 统一评测（best/last × INNER/OUTER）。

指标口径与主评测完全一致：直接复用 ``evaluate_utae_v6.run_split``（同一
``evaluate_tile_v2``、同一对象匹配与边界容差、同一 AP 计算），只把
"数据来自哪一折"变成配置。

用法::

    python scripts/evaluate_utae_cv.py --config configs/utae/event_cv/F1.yaml --mode multi --device cuda:0
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
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import file_sha256, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.models.utae_dates import UTAERealDates  # noqa: E402

from evaluate_utae_v6 import _rounded, run_split  # noqa: E402
from train_utae_cv import load_cv_config  # noqa: E402


def per_tile_rows(tile_rows: list[dict]) -> list[dict]:
    out = []
    for r in tile_rows:
        b = r["boundary"]
        if b["status"] == "both_empty":
            bf1 = None
        elif b["gt_px"] == 0 or b["pred_px"] == 0:
            bf1 = 0.0
        else:
            pre = b["prec_tp"] / b["pred_px"]
            rec = b["rec_tp"] / b["gt_px"]
            bf1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0
        out.append({
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, choices=("single", "multi", "history"))
    ap.add_argument("--checkpoints", default="best,last")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_cv_config(resolve_path(args.config), args.set)
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    prot = cfg["protocol"]
    run_dir = ROOT / cfg["output_dir"] / str(cfg["fold"]) / args.mode / str(cfg["seed"])
    pool = read_manifest(resolve_path(prot["pool"]))
    inner = [r for r in pool if r["protocol_role"] == prot["roles"]["inner"]]
    outer = read_manifest(resolve_path(prot["outer"]))
    stats = load_stats(resolve_path(prot["stats"]))
    model = UTAERealDates(in_channels=int(cfg["data"]["in_channels"]), num_classes=1).to(dev)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    out: dict = {}
    for ck in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        p = run_dir / f"{ck}.pt"
        if not p.is_file():
            print(f"[skip] {p}")
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if ckpt.get("config_sha256") != __import__("train_utae_cv").config_sha(cfg):
            raise ValueError("checkpoint 与本配置不一致")
        model.load_state_dict(ckpt["model"])
        model.eval()
        out[ck] = {}
        for split, rows in (("inner", inner), ("outer", outer)):
            t0 = time.time()
            overall, tile_rows, extra = run_split(model, rows, stats, args.mode, dev)
            tag = f"{ck}_{split}"
            out[ck][split] = {"n_samples": len(rows), "overall": overall,
                              "by_event": extra["by_event"],
                              "seconds": round(time.time() - t0, 2),
                              "checkpoint_sha256": file_sha256(p),
                              "checkpoint_epoch": int(ckpt["epoch"])}
            rows_out = per_tile_rows(tile_rows)
            with (run_dir / "eval" / f"per_tile_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
                w.writeheader()
                w.writerows(rows_out)
            (run_dir / "eval" / f"objects_{tag}.json").write_text(
                json.dumps([{k: r[k] for k in ("sample_id", "event_id", "bins", "tiny", "fine_bins")}
                            for r in tile_rows], indent=1), encoding="utf-8")
            if args.save_predictions:
                np.savez_compressed(run_dir / "eval" / f"predictions_{tag}.npz", **extra["predictions"])
            (run_dir / "eval" / f"eval_{tag}.json").write_text(
                json.dumps(out[ck][split], indent=1, ensure_ascii=False), encoding="utf-8")
            px = overall["pixel"]
            print(f"[{cfg['fold']}/{args.mode}] {tag}: pooled={px['iou']:.4f} AP={px['average_precision']:.4f} "
                  f"dice={px['dice']:.4f} P={px['precision']:.4f} R={px['recall']:.4f} "
                  f"objRec={overall['objects']['object_recall_evaluable']:.4f} "
                  f"bF1={overall['boundary']['f1']:.4f}", flush=True)
    (run_dir / "eval_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

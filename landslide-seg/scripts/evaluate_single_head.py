"""N1/N2（及 N0）的冻结评测：pooled + macro、分档对象、边界、负瓦片。"""

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
from landslide_seg.training.segformer_metrics import evaluate_tile_v2, summarize_tiles_v2  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def _latency(model, device, iters=50):
    import time
    x = torch.randn(1, 10, 128, 128, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
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
    return {"p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2)}


def _tile_iou(t):
    d = t["tp"] + t["fp"] + t["fn"]
    return t["tp"] / d if d else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "single_head" / "base.yaml"))
    ap.add_argument("--experiments", nargs="+", default=["CF0_NOAUX", "SINGLE_SUM", "SINGLE_AFF"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_segformer_config(args.config)["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    recs_all = read_manifest(resolve_path(cfg["data"]["manifest"]))
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out = {"version": cfg["data"]["version"], "seed": args.seed, "experiments": {}}
    for exp in args.experiments:
        ckpt = root / exp / str(args.seed) / "best.pt"
        if not ckpt.is_file():
            print(f"[skip] {ckpt}")
            continue
        model, meta = load_checkpoint(ckpt, device=str(device))
        out["experiments"][exp] = {"split": {}, "params": meta.get("params"), "latency_b1": _latency(model, device)}
        for split in args.splits:
            recs = filter_split(recs_all, split)
            ds = S2SegDataset(recs, stats, preload=True)
            rows = []
            with torch.no_grad():
                for i in range(0, len(ds), 32):
                    b = collate([ds[j] for j in range(i, min(i + 32, len(ds)))])
                    o = model(b["image"].to(device))
                    prob = torch.sigmoid(o["logits"].float()[:, 0]).cpu().numpy()
                    gt = b["mask"].numpy().astype(bool)
                    valid = b["valid"].numpy().astype(bool)
                    for k, sid in enumerate(b["sample_id"]):
                        rows.append(evaluate_tile_v2(
                            prob[k], gt[k], valid[k],
                            threshold=0.5, iou_threshold=0.25, iou_strict=0.5, boundary_tol=1,
                            sample_id=sid, event_id=b["event_id"][k],
                        ))
            s = summarize_tiles_v2(rows)
            macro = float(np.mean([_tile_iou(r) for r in rows if _tile_iou(r) is not None]))
            out["experiments"][exp]["split"][split] = {
                "n_tiles": len(rows),
                "pooled": s["pixel"],
                "macro_iou": macro,
                "objects": s["objects"],
                "boundary": s["boundary"],
                "negative_tiles": s["negative_tiles"],
            }
            print(f"[{exp} {split}] pooled IoU={s['pixel']['iou']:.4f} macro={macro:.4f} obj={s['objects']['object_recall_evaluable']:.4f}", flush=True)
            del ds
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = resolve_path("outputs/segformer/single_head")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_v2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["# 单头模型评测（metric v2，seed 1234）", ""]
    for exp, r in out["experiments"].items():
        lines += [f"## {exp}", "", "| split | IoU(pooled) | IoU(macro) | Dice | P | R | 对象召回 | 边界F1 |", "|---|---|---|---|---|---|---|---|"]
        for split, s in r["split"].items():
            p = s["pooled"]
            lines.append(
                f"| {split} | {p['iou']:.4f} | {s['macro_iou']:.4f} | {p['dice']:.4f} | "
                f"{p['precision']:.4f} | {p['recall']:.4f} | {s['objects']['object_recall_evaluable']:.4f} | {s['boundary']['f1']:.4f} |"
            )
        lines.append("")
    (out_dir / "eval_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

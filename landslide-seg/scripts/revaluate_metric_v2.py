"""用 metric v2 重评全部现有 best checkpoint（不重新选模型、不覆盖旧结果）。

产物：
- 每个 run 下 ``eval_v2/eval_<split>.json`` 与 ``per_tile_<split>.csv``；
- ``outputs/segformer/<version>/ablation_eval_metric_v2.json``（逐 run + 逐组聚合）；
- ``outputs/segformer/<version>/ablation_table_metric_v2.md``。

用法::

    python scripts/revaluate_metric_v2.py --splits val test
    python scripts/revaluate_metric_v2.py --splits train val test --experiments B1 --seeds 1234
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
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

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import (  # noqa: E402
    S2SegDataset,
    collate,
    filter_split,
    load_stats,
    read_manifest,
)
from landslide_seg.training.segformer_metrics import (  # noqa: E402
    METRIC_VERSION,
    evaluate_tile_v2,
    summarize_by_event_v2,
    summarize_tiles_v2,
)
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def _safe(a, b):
    return (a / b) if b else None


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "values": []}
    return {
        "n": len(vals),
        "mean": round(float(stats.mean(vals)), 5),
        "std": round(float(stats.pstdev(vals)), 5) if len(vals) > 1 else 0.0,
        "values": [round(float(v), 5) for v in vals],
    }


def _per_tile_row(r: dict) -> dict:
    denom = r["tp"] + r["fp"] + r["fn"]
    return {
        "sample_id": r["sample_id"],
        "event_id": r["event_id"],
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        "gt_px": r["gt_px"], "pred_px": r["pred_px"], "valid_px": r["valid_px"],
        "iou": round(_safe(r["tp"], denom) or 0.0, 6),
        "dice": round(_safe(2 * r["tp"], 2 * r["tp"] + r["fp"] + r["fn"]) or 0.0, 6),
        "precision": round(_safe(r["tp"], r["tp"] + r["fp"]) or 0.0, 6),
        "recall": round(_safe(r["tp"], r["tp"] + r["fn"]) or 0.0, 6),
        "n_gt_objects": r["n_gt_objects"],
        "n_gt_objects_evaluable": r["n_gt_objects_evaluable"],
        "n_matched_evaluable": r["n_matched_evaluable"],
        "n_matched_strict_evaluable": r["n_matched_strict_evaluable"],
        "tiny_n_evaluable": r["tiny"]["n_objects_evaluable"],
        "tiny_n_matched_evaluable": r["tiny"]["n_matched_evaluable"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    ap.add_argument("--experiments", nargs="+", default=["A0", "A1", "B0", "B1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 2026, 3407])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    base = load_segformer_config(args.config)["segformer"]
    version = base["data"]["version"]
    root = resolve_path(base["output_dir"]) / version
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    runs: list[dict] = []
    for exp in args.experiments:
        for seed in args.seeds:
            run_dir = root / exp / str(seed)
            best = run_dir / "best.pt"
            if not best.is_file():
                print(f"[skip] {best}")
                continue
            model, ckpt = load_checkpoint(best, device=str(device))
            cfg = ckpt["config"]["segformer"]
            records_all = read_manifest(resolve_path(cfg["data"]["manifest"]))
            stats_data = load_stats(resolve_path(cfg["data"]["stats"]))
            for split in args.splits:
                records = filter_split(records_all, split)
                if not records:
                    continue
                ds = S2SegDataset(records, stats_data, preload=True)
                rows: list[dict] = []
                with torch.no_grad():
                    for i in range(0, len(ds), args.batch_size):
                        batch = collate([ds[j] for j in range(i, min(i + args.batch_size, len(ds)))])
                        logits = model(batch["image"].to(device))["logits"].float()
                        prob = torch.sigmoid(logits[:, 0]).cpu().numpy()
                        gt = batch["mask"].numpy().astype(bool)
                        valid = batch["valid"].numpy().astype(bool)
                        for k, sid in enumerate(batch["sample_id"]):
                            rows.append(
                                evaluate_tile_v2(
                                    prob[k], gt[k], valid[k],
                                    threshold=float(cfg["eval"]["threshold"]),
                                    iou_threshold=float(cfg["eval"]["object_iou_threshold"]),
                                    iou_strict=float(cfg["eval"]["object_iou_strict"]),
                                    boundary_tol=int(cfg["eval"]["boundary_tol_px"]),
                                    sample_id=sid,
                                    event_id=batch["event_id"][k],
                                )
                            )
                overall = summarize_tiles_v2(rows)
                by_event = summarize_by_event_v2(rows)
                out_dir = run_dir / "eval_v2"
                out_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "metric_version": METRIC_VERSION,
                    "checkpoint": str(best).replace("\\", "/"),
                    "split": split,
                    "n_samples": len(rows),
                    "overall": overall,
                    "by_event": by_event,
                }
                (out_dir / f"eval_{split}.json").write_text(
                    json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
                )
                per_tile = [_per_tile_row(r) for r in rows]
                with (out_dir / f"per_tile_{split}.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(per_tile[0].keys()))
                    writer.writeheader()
                    writer.writerows(per_tile)
                runs.append(
                    {
                        "experiment": exp,
                        "seed": seed,
                        "split": split,
                        "checkpoint": str(best).replace("\\", "/"),
                        "overall": overall,
                        "by_event": by_event,
                    }
                )
                print(
                    f"[{exp} {seed} {split}] IoU={overall['pixel']['iou']:.4f} "
                    f"obj_eval={overall['objects']['object_recall_evaluable']:.4f} "
                    f"tiny={overall['objects']['tiny']['n_matched_evaluable']}/"
                    f"{overall['objects']['tiny']['n_objects_evaluable']}",
                    flush=True,
                )
            del model, ckpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    agg: dict = {}
    for exp in args.experiments:
        agg[exp] = {}
        for split in args.splits:
            sub = [r for r in runs if r["experiment"] == exp and r["split"] == split]
            if not sub:
                continue
            entry = {"n_runs": len(sub)}
            entry["pixel.iou"] = _mean_std([r["overall"]["pixel"]["iou"] for r in sub])
            entry["pixel.dice"] = _mean_std([r["overall"]["pixel"]["dice"] for r in sub])
            entry["pixel.precision"] = _mean_std([r["overall"]["pixel"]["precision"] for r in sub])
            entry["pixel.recall"] = _mean_std([r["overall"]["pixel"]["recall"] for r in sub])
            entry["objects.object_recall_evaluable"] = _mean_std(
                [r["overall"]["objects"]["object_recall_evaluable"] for r in sub]
            )
            entry["objects.object_recall_strict_evaluable"] = _mean_std(
                [r["overall"]["objects"]["object_recall_strict_evaluable"] for r in sub]
            )
            entry["boundary.f1"] = _mean_std([r["overall"]["boundary"]["f1"] for r in sub])
            entry["tiny.object_recall_evaluable"] = _mean_std(
                [r["overall"]["objects"]["tiny"]["object_recall_evaluable"] for r in sub]
            )
            entry["tiny.pixel_recall"] = _mean_std(
                [r["overall"]["objects"]["tiny"]["pixel_recall"] for r in sub]
            )
            bins = {}
            for name in ("small", "medium", "large"):
                bins[name] = {
                    "object_recall_evaluable": _mean_std(
                        [r["overall"]["objects"]["bins"][name]["object_recall_evaluable"] for r in sub]
                    ),
                    "pixel_recall": _mean_std(
                        [r["overall"]["objects"]["bins"][name]["pixel_recall"] for r in sub]
                    ),
                }
            entry["bins"] = bins
            agg[exp][split] = entry

    out = {
        "metric_version": METRIC_VERSION,
        "version": version,
        "experiments": args.experiments,
        "seeds": args.seeds,
        "splits": args.splits,
        "aggregate": agg,
        "runs": runs,
    }
    out_path = root / f"ablation_eval_metric_{METRIC_VERSION}.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = [f"# 四组结果（metric {METRIC_VERSION}，均值 ± 标准差）", ""]
    for split in args.splits:
        lines += [f"## {split}", ""]
        lines += [
            "| 组 | IoU | Dice | P | R | 对象召回(未截断) | 边界 F1 | tiny 对象召回 | tiny 像素召回 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for exp in args.experiments:
            e = agg.get(exp, {}).get(split)
            if not e:
                continue

            def cell(key):
                d = e[key]
                if not d["n"]:
                    return "N/A"
                return f"{d['mean']:.3f} ± {d['std']:.3f}"

            lines.append(
                f"| {exp} | {cell('pixel.iou')} | {cell('pixel.dice')} | {cell('pixel.precision')} | "
                f"{cell('pixel.recall')} | {cell('objects.object_recall_evaluable')} | "
                f"{cell('boundary.f1')} | {cell('tiny.object_recall_evaluable')} | {cell('tiny.pixel_recall')} |"
            )
        lines.append("")
    table_path = root / f"ablation_table_metric_{METRIC_VERSION}.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"[saved] {out_path}\n[saved] {table_path}")


if __name__ == "__main__":
    main()

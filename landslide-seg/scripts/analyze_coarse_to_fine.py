"""P4：粗/细阶段分析、定位覆盖诊断与固定案例。

对 A1-AUX / CF1 / CF0 / U0 的 best checkpoint：
- 同一 checkpoint 分别评估**最终输出**与**粗输出（base_logits）**（metric v2）；
- 定位覆盖：按粗概率给有效像素排序，在 5%/10%/20% 面积预算下报告
  GT 像素覆盖率与"对象被候选区触及"的比例（平分时按像素索引稳定排序）；
- 可选输出案例图（RGB / GT / 粗概率 / 细概率 / 粗 mask / 细 mask）。

用法::

    python scripts/analyze_coarse_to_fine.py --experiments A1_AUX CF1 CF0 U0 --splits val test
"""

from __future__ import annotations

import argparse
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


def _coverage(prob: np.ndarray, gt: np.ndarray, valid: np.ndarray, budgets=(0.05, 0.10, 0.20)) -> dict:
    """按粗概率排序，返回各预算下的 GT 像素覆盖与对象命中。"""
    v = np.asarray(valid, dtype=bool)
    g = np.asarray(gt, dtype=bool) & v
    p = np.asarray(prob, dtype=np.float32)
    flat_p = p[v]
    n = flat_p.size
    if n == 0:
        return {}
    order = np.argsort(-flat_p, kind="stable")
    gt_flat = g[v]
    labels, n_obj = label_objects(g)
    obj_ids = [o for o in range(1, n_obj + 1)]
    out = {}
    for b in budgets:
        k = max(1, int(round(n * b)))
        cand = np.zeros(n, dtype=bool)
        cand[order[:k]] = True
        # 映射回原图
        cand_full = np.zeros_like(v)
        cand_full[v] = cand
        gt_total = int(gt_flat.sum())
        hit = int((cand_full & g).sum())
        obj_hit = 0
        if n_obj:
            present = set(np.unique(labels[cand_full & g]).tolist())
            obj_hit = sum(1 for o in obj_ids if o in present)
        key = f"budget_{int(b*100)}pct"
        out[key] = {
            "pixel_coverage": (hit / gt_total) if gt_total else None,
            "object_hit_rate": (obj_hit / n_obj) if n_obj else None,
            "n_objects": n_obj,
        }
    return out


def _latency(model, device, iters=50):
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
    return {
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "coarse_to_fine" / "base.yaml"))
    ap.add_argument("--experiments", nargs="+", default=["A1_AUX", "CF1", "CF0", "U0"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--figures", type=int, default=0, help="每个 split 输出多少张粗/细对照图")
    args = ap.parse_args()

    base_cfg = load_segformer_config(args.config)["segformer"]
    version = base_cfg["data"]["version"]
    root = resolve_path(base_cfg["output_dir"]) / version
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records_all = read_manifest(resolve_path(base_cfg["data"]["manifest"]))
    stats = load_stats(resolve_path(base_cfg["data"]["stats"]))

    report = {"version": version, "seed": args.seed, "splits": {}, "cost": {}}
    for split in args.splits:
        records = filter_split(records_all, split)
        ds = S2SegDataset(records, stats, preload=True)
        rows = {}
        for exp in args.experiments:
            ckpt = root / exp / str(args.seed) / "best.pt"
            if not ckpt.is_file():
                print(f"[skip] {ckpt}")
                continue
            model, meta = load_checkpoint(ckpt, device=str(device))
            cfg = meta["config"]["segformer"]
            row_final, row_coarse = [], []
            coverage = {b: {"pixel": [], "obj": []} for b in (5, 10, 20)}
            with torch.no_grad():
                for i in range(0, len(ds), args.batch_size):
                    batch = collate([ds[j] for j in range(i, min(i + args.batch_size, len(ds)))])
                    out = model(batch["image"].to(device))
                    fine = torch.sigmoid(out["logits"].float()[:, 0]).cpu().numpy()
                    coarse = torch.sigmoid(out["base_logits"].float()[:, 0]).cpu().numpy()
                    gt = batch["mask"].numpy().astype(bool)
                    valid = batch["valid"].numpy().astype(bool)
                    for k, sid in enumerate(batch["sample_id"]):
                        common = dict(
                            threshold=float(cfg["eval"]["threshold"]),
                            iou_threshold=float(cfg["eval"]["object_iou_threshold"]),
                            iou_strict=float(cfg["eval"]["object_iou_strict"]),
                            boundary_tol=int(cfg["eval"]["boundary_tol_px"]),
                            sample_id=sid,
                            event_id=batch["event_id"][k],
                        )
                        row_final.append(evaluate_tile_v2(fine[k], gt[k], valid[k], **common))
                        row_coarse.append(evaluate_tile_v2(coarse[k], gt[k], valid[k], **common))
                        cov = _coverage(coarse[k], gt[k], valid[k])
                        for b in (5, 10, 20):
                            v = cov.get(f"budget_{b}pct", {})
                            if v.get("pixel_coverage") is not None:
                                coverage[b]["pixel"].append(v["pixel_coverage"])
                            if v.get("object_hit_rate") is not None:
                                coverage[b]["obj"].append(v["object_hit_rate"])
            final = summarize_tiles_v2(row_final)
            coarse_s = summarize_tiles_v2(row_coarse)
            rows[exp] = {
                "final": {
                    "iou": final["pixel"]["iou"], "dice": final["pixel"]["dice"],
                    "precision": final["pixel"]["precision"], "recall": final["pixel"]["recall"],
                    "object_recall_evaluable": final["objects"]["object_recall_evaluable"],
                    "boundary_f1": final["boundary"]["f1"],
                },
                "coarse": {
                    "iou": coarse_s["pixel"]["iou"], "dice": coarse_s["pixel"]["dice"],
                    "precision": coarse_s["pixel"]["precision"], "recall": coarse_s["pixel"]["recall"],
                    "object_recall_evaluable": coarse_s["objects"]["object_recall_evaluable"],
                    "boundary_f1": coarse_s["boundary"]["f1"],
                },
                "final_bins": final["objects"]["bins"],
                "final_tiny": final["objects"]["tiny"],
                "coarse_bins": coarse_s["objects"]["bins"],
                "negative_tiles": final["negative_tiles"],
                "params": meta.get("params"),
                "latency_b1": _latency(model, device),
                "localization_coverage": {
                    f"budget_{b}pct": {
                        "pixel_coverage_mean": round(float(np.mean(coverage[b]["pixel"])), 4) if coverage[b]["pixel"] else None,
                        "object_hit_rate_mean": round(float(np.mean(coverage[b]["obj"])), 4) if coverage[b]["obj"] else None,
                        "n_tiles": len(coverage[b]["pixel"]),
                    }
                    for b in (5, 10, 20)
                },
            }
            summary_path = root / exp / str(args.seed) / "train_summary.json"
            train_summary = {}
            if summary_path.is_file():
                train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report["cost"][exp] = {
                "params": meta.get("params"),
                "latency_b1": rows[exp]["latency_b1"],
                "train_peak_mib": train_summary.get("peak_alloc_mib"),
                "train_seconds": train_summary.get("seconds"),
                "best_epoch": train_summary.get("best_epoch"),
                "model_identity": (meta.get("meta") or {}).get("model_identity"),
            }
            print(f"[{split} {exp}] final IoU={final['pixel']['iou']:.4f} coarse IoU={coarse_s['pixel']['iou']:.4f}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["splits"][split] = {"per_experiment": rows}

    out_dir = resolve_path("outputs/segformer/coarse_to_fine")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coarse_to_fine_analysis.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["# 粗/细阶段与定位覆盖", ""]
    for split in args.splits:
        lines += [f"## {split}", "",
                  "| 组 | 最终 IoU | 粗 IoU | Δ(细-粗) | 最终对象召回 | 粗对象召回 | 最终边界F1 |",
                  "|---|---|---|---|---|---|---|"]
        for exp, r in report["splits"][split]["per_experiment"].items():
            f, c = r["final"], r["coarse"]
            d = (f["iou"] - c["iou"]) if (f["iou"] is not None and c["iou"] is not None) else None
            lines.append(
                f"| {exp} | {f['iou']:.4f} | {c['iou']:.4f} | {d:+.4f} | "
                f"{f['object_recall_evaluable']:.4f} | {c['object_recall_evaluable']:.4f} | {f['boundary_f1']:.4f} |"
            )
        lines += ["", "定位覆盖（按粗概率排序）：", "",
                  "| 组 | 预算 | GT 像素覆盖率 | 对象命中率 |", "|---|---|---|---|"]
        for exp, r in report["splits"][split]["per_experiment"].items():
            for b in (5, 10, 20):
                v = r["localization_coverage"][f"budget_{b}pct"]
                lines.append(f"| {exp} | {b}% | {v['pixel_coverage_mean']} | {v['object_hit_rate_mean']} |")
        lines.append("")
    (out_dir / "coarse_to_fine_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

"""测试集案例可视化：按"大目标命中 / 中位 / 漏检 / 误报 / 小目标检出"选片，出对照图。

用法::

    python scripts/visualize_segformer_cases.py --checkpoint outputs/segformer/s2_10band_v1/B1/1234/best.pt
    python scripts/visualize_segformer_cases.py --checkpoint .../B1/1234/best.pt --compare .../A0/1234/best.pt
"""

from __future__ import annotations

import argparse
import json
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
from landslide_seg.training.segformer_metrics import evaluate_tile  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def stretch(rgb: np.ndarray) -> np.ndarray:
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def infer_all(model, records, stats, device) -> np.ndarray:
    ds = S2SegDataset(records, stats, preload=True)
    probs = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = collate([ds[j] for j in range(i, min(i + 32, len(ds)))])
            p = torch.sigmoid(model(batch["image"].to(device))["logits"].float()[:, 0])
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def load_arrays(records) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x10 = np.stack([np.load(r["cache"])["x10"].astype(np.float32) for r in records])
    gt = np.stack([np.load(r["cache"])["mask"].astype(bool) for r in records])
    valid = np.stack([np.load(r["cache"])["valid"].astype(bool) for r in records])
    return x10, gt, valid


def metrics_for(prob, gt, valid, records, thr=0.5) -> list[dict]:
    rows = []
    for i, rec in enumerate(records):
        r = evaluate_tile(
            prob[i], gt[i], valid[i], threshold=thr, sample_id=rec["sample_id"],
            event_id=rec["event_id"],
        )
        denom = r["tp"] + r["fp"] + r["fn"]
        rows.append(
            {
                "sample_id": rec["sample_id"],
                "event_id": rec["event_id"],
                "gt_px": int(gt[i].sum()),
                "pred_px": int((prob[i] >= thr).sum()),
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                "iou": (r["tp"] / denom) if denom else None,
                "n_gt_objects": r["n_gt_objects"],
                "n_matched": r["n_matched"],
            }
        )
    return rows


def pick_cases(rows: list[dict]) -> list[tuple[str, dict]]:
    """确定性选片：大目标命中、中位、漏检、误报、小目标检出。"""
    hits = [r for r in rows if (r["iou"] or 0) > 0]
    picks: list[tuple[str, dict]] = []

    def add(label, row):
        if row is not None and all(row["sample_id"] != p[1]["sample_id"] for p in picks):
            picks.append((label, row))

    # 1) 大目标且命中
    large = [r for r in hits if r["gt_px"] >= 150]
    add("large hit", max(large, key=lambda r: r["iou"]) if large else None)
    # 2) 中位 IoU
    if hits:
        s = sorted(hits, key=lambda r: r["iou"])
        add("median IoU", s[len(s) // 2])
    # 3) 漏检：GT 不小但 IoU 为 0
    misses = [r for r in rows if r["gt_px"] >= 80 and (r["iou"] or 0) == 0]
    add("miss", max(misses, key=lambda r: r["gt_px"]) if misses else None)
    # 4) 误报最多
    fps = [r for r in rows if r["fp"] > 0]
    add("false alarms", max(fps, key=lambda r: r["fp"]) if fps else None)
    # 5) 小目标但检出
    small = [r for r in hits if 0 < r["gt_px"] < 50]
    add("small detected", max(small, key=lambda r: r["iou"]) if small else None)
    return picks


def draw_cases(picks, x10, gt, valid, prob, out_png: Path, title: str, thr=0.5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(picks)
    fig, axes = plt.subplots(n, 6, figsize=(16, 3.0 * n), squeeze=False)
    for row, (label, rec) in enumerate(picks):
        idx = rec["_idx"]
        rgb = stretch(x10[idx][[2, 1, 0]].transpose(1, 2, 0))
        false = stretch(x10[idx][[9, 7, 2]].transpose(1, 2, 0))
        p = prob[idx]
        pred = p >= thr
        g = gt[idx]
        err = np.zeros((*g.shape, 3), dtype=np.float32)
        err[..., 1] = (g & pred)
        err[..., 0] = (~g & pred)
        err[..., 2] = (g & ~pred)
        panels = [
            (rgb, f"{rec['sample_id']}  RGB"),
            (false, "B12/B8A/B04"),
            (g, f"GT  {rec['gt_px']} px"),
            (p, "prob"),
            (pred, f"pred  IoU={rec['iou'] if rec['iou'] is not None else 0:.3f}"),
            (err, f"{label}  TP/FP/FN"),
        ]
        for j, (img, t) in enumerate(panels):
            ax = axes[row][j]
            if img.ndim == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap="magma" if j == 3 else "gray", vmin=0, vmax=1)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def draw_compare(picks, x10, gt, prob_a, prob_b, name_a, name_b, out_png: Path, thr=0.5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(picks)
    fig, axes = plt.subplots(2 * n, 4, figsize=(11, 2.8 * 2 * n), squeeze=False)
    for row, (label, rec) in enumerate(picks):
        idx = rec["_idx"]
        rgb = stretch(x10[idx][[2, 1, 0]].transpose(1, 2, 0))
        g = gt[idx]
        for m, (name, prob) in enumerate(((name_a, prob_a), (name_b, prob_b))):
            pred = prob[idx] >= thr
            err = np.zeros((*g.shape, 3), dtype=np.float32)
            err[..., 1] = (g & pred)
            err[..., 0] = (~g & pred)
            err[..., 2] = (g & ~pred)
            panels = [
                (rgb, f"{rec['sample_id']} RGB") if m == 0 else (rgb, f"{name}"),
                (g, "GT"),
                (pred, f"{name} pred"),
                (err, f"{name} TP/FP/FN  ({label})"),
            ]
            for j, (img, t) in enumerate(panels):
                ax = axes[row * 2 + m][j]
                if img.ndim == 3:
                    ax.imshow(np.clip(img, 0, 1))
                else:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(t, fontsize=8)
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--compare", default=None, help="可选：另一个 checkpoint，用于同图对照")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_a = Path(args.checkpoint)
    if not ckpt_a.is_absolute():
        ckpt_a = resolve_path(ckpt_a)
    model_a, meta_a = load_checkpoint(ckpt_a, device="cuda:0")
    cfg = meta_a["config"]["segformer"]
    name_a = cfg["experiment"]
    device = next(model_a.parameters()).device

    records = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), args.split)
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    x10, gt, valid = load_arrays(records)
    prob_a = infer_all(model_a, records, stats, device)
    rows_a = metrics_for(prob_a, gt, valid, records)

    prob_b = None
    name_b = None
    rows_b = None
    if args.compare:
        ckpt_b = Path(args.compare)
        if not ckpt_b.is_absolute():
            ckpt_b = resolve_path(ckpt_b)
        model_b, meta_b = load_checkpoint(ckpt_b, device="cuda:0")
        name_b = meta_b["config"]["segformer"]["experiment"]
        prob_b = infer_all(model_b, records, stats, device)
        rows_b = metrics_for(prob_b, gt, valid, records)
        del model_b

    picks = pick_cases(rows_a)
    out_dir = Path(args.out) if args.out else (
        resolve_path("outputs/segformer") / cfg["data"]["version"] / "audit_cases"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched = []
    for label, rec in picks:
        idx = next(i for i, r in enumerate(records) if r["sample_id"] == rec["sample_id"])
        item = dict(rec)
        item["_idx"] = idx
        item["case"] = label
        if rows_b:
            item["compare"] = next(r for r in rows_b if r["sample_id"] == rec["sample_id"])
        enriched.append((label, item))

    tag = f"{name_a}"
    if name_b:
        tag += f"_vs_{name_b}"
    draw_cases(
        enriched, x10, gt, valid, prob_a,
        out_dir / f"{tag}_{args.split}_cases.png",
        f"{name_a} on {args.split} — {len(enriched)} audit cases (thr=0.5)",
    )
    if prob_b is not None and name_b:
        draw_compare(
            enriched, x10, gt, prob_b, prob_a, name_b, name_a,
            out_dir / f"{name_b}_vs_{name_a}_{args.split}_compare.png",
        )

    table = []
    for label, rec in enriched:
        row = {
            "case": label,
            "sample_id": rec["sample_id"],
            "gt_px": rec["gt_px"],
            "iou": round(rec["iou"], 4) if rec["iou"] is not None else None,
            "tp": rec["tp"], "fp": rec["fp"], "fn": rec["fn"],
            "n_gt_objects": rec["n_gt_objects"],
            "n_matched": rec["n_matched"],
        }
        if "compare" in rec:
            c = rec["compare"]
            row["compare"] = {
                "experiment": name_b,
                "iou": round(c["iou"], 4) if c["iou"] is not None else None,
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                "n_matched": c["n_matched"],
            }
        table.append(row)

    (out_dir / f"{tag}_{args.split}_cases.json").write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_a).replace("\\", "/"),
                "compare": str(args.compare) if args.compare else None,
                "split": args.split,
                "n_cases": len(table),
                "cases": table,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(table, indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

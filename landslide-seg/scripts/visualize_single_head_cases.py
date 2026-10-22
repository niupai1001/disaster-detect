"""N1 vs N2 固定案例：找回、N1 误检减少、N2 新增误检、边界修正。"""

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
from landslide_seg.training.segformer_metrics import evaluate_tile_v2  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def _rgb(x10):
    rgb = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _err(gt, pred):
    out = np.zeros(gt.shape + (3,), dtype=np.float32)
    out[..., 1] = (gt & pred)
    out[..., 0] = (~gt & pred)
    out[..., 2] = (gt & ~pred)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_segformer_config(ROOT / "configs" / "segformer" / "single_head" / "base.yaml")["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    device = torch.device(args.device)
    m1, _ = load_checkpoint(root / "SINGLE_SUM" / str(args.seed) / "best.pt", device=str(device))
    m2, _ = load_checkpoint(root / "SINGLE_AFF" / str(args.seed) / "best.pt", device=str(device))
    m1.eval(); m2.eval()
    recs = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), args.split)
    ds = S2SegDataset(recs, load_stats(resolve_path(cfg["data"]["stats"])), preload=True)

    rows = []
    cache = {}
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            b = collate([ds[j] for j in range(i, min(i + 32, len(ds)))])
            img = b["image"].to(device)
            p1 = torch.sigmoid(m1(img)["logits"].float()[:, 0]).cpu().numpy()
            p2 = torch.sigmoid(m2(img)["logits"].float()[:, 0]).cpu().numpy()
            gt = b["mask"].numpy().astype(bool)
            valid = b["valid"].numpy().astype(bool)
            for k, sid in enumerate(b["sample_id"]):
                r1 = evaluate_tile_v2(p1[k], gt[k], valid[k], threshold=0.5, iou_threshold=0.25, iou_strict=0.5, boundary_tol=1, sample_id=sid, event_id=b["event_id"][k])
                r2 = evaluate_tile_v2(p2[k], gt[k], valid[k], threshold=0.5, iou_threshold=0.25, iou_strict=0.5, boundary_tol=1, sample_id=sid, event_id=b["event_id"][k])
                rows.append({
                    "idx": i + k, "sample_id": sid,
                    "i1": r1["tp"] / max(r1["tp"] + r1["fp"] + r1["fn"], 1),
                    "i2": r2["tp"] / max(r2["tp"] + r2["fp"] + r2["fn"], 1),
                    "fp1": r1["fp"], "fp2": r2["fp"], "gt": int(gt[k].sum()),
                    "b1": r1["boundary"], "b2": r2["boundary"],
                })
                cache[i + k] = (p1[k], p2[k], gt[k], valid[k])

    def bf1(b):
        if b.get("status") != "ok":
            return None
        p = b["prec_tp"] / b["pred_px"] if b["pred_px"] else 0.0
        r = b["rec_tp"] / b["gt_px"] if b["gt_px"] else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    picks = []
    rec = [r for r in rows if r["i1"] < 0.05 and r["i2"] > 0.3]
    if not rec:
        rec = [r for r in rows if r["i1"] < 0.15 and r["i2"] - r["i1"] > 0.2]
    if rec:
        picks.append(("N2 improves over N1", max(rec, key=lambda r: r["i2"] - r["i1"])))
    elif rows:
        best = max(rows, key=lambda r: r["i2"] - r["i1"])
        picks.append(("N2 best delta", best))
    red = [r for r in rows if r["fp1"] > 0]
    if red:
        picks.append(("N1 FP reduced by N2", max(red, key=lambda r: r["fp1"] - r["fp2"])))
    add = [r for r in rows if r["fp2"] > r["fp1"]]
    if add:
        picks.append(("N2 adds FP", max(add, key=lambda r: r["fp2"] - r["fp1"])))
    bfix = [r for r in rows if bf1(r["b2"]) is not None and bf1(r["b1"]) is not None and bf1(r["b2"]) > bf1(r["b1"])]
    if bfix:
        picks.append(("boundary improved", max(bfix, key=lambda r: bf1(r["b2"]) - bf1(r["b1"]))))
    if not picks:
        picks.append(("largest GT", max(rows, key=lambda r: r["gt"])))

    fig, axes = plt.subplots(len(picks), 8, figsize=(21, 2.9 * len(picks)), squeeze=False)
    for row, (label, r) in enumerate(picks):
        p1, p2, gt, valid = cache[r["idx"]]
        x10 = ds[r["idx"]]["image"].numpy().astype(np.float32) if not torch.is_tensor(ds[r["idx"]]["image"]) else ds[r["idx"]]["image"].numpy()
        panels = [
            (_rgb(x10), f"{r['sample_id']} RGB"),
            (gt, f"GT {r['gt']} px"),
            (p1, "N1 SUM prob"),
            (p2, "N2 AFF prob"),
            (p1 >= 0.5, f"N1 IoU={r['i1']:.3f}"),
            (p2 >= 0.5, f"N2 IoU={r['i2']:.3f}"),
            (_err(gt, p2 >= 0.5), "N2 error TP/FP/FN"),
            (np.abs(p2 - p1), "|N2-N1|"),
        ]
        for j, (img, title) in enumerate(panels):
            ax = axes[row][j]
            if img.ndim == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap="magma" if j in (2, 3, 7) else "gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row][0].set_ylabel(label)

    out_dir = resolve_path("outputs/segformer/single_head")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"SINGLE_{args.split}_cases.png", dpi=110)
    plt.close(fig)
    (out_dir / f"SINGLE_{args.split}_cases.json").write_text(
        json.dumps([{"label": l, **{k: v for k, v in r.items() if k != "b1" and k != "b2"}} for l, r in picks], indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print([l for l, _ in picks])
    print(f"[saved] {out_dir / f'SINGLE_{args.split}_cases.png'}")


if __name__ == "__main__":
    main()

"""粗/细阶段固定案例：CF1 的粗输出、精分输出与 GT 对照。

按确定性规则选 3 类案例：
1. 粗阶段漏检（粗 IoU=0）但精分找回；
2. 粗阶段误检最多、精分减少（coarse_fp - fine_fp 最大）；
3. 精分反而变差（coarse_iou - fine_iou 最大）。

用法::

    python scripts/visualize_coarse_to_fine_cases.py --experiment CF1 --split val
"""

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
from landslide_seg.training.segformer_metrics import confusion, evaluate_tile_v2  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def _iou(prob, gt, valid, thr=0.5):
    c = confusion(prob >= thr, gt, valid)
    d = c["tp"] + c["fp"] + c["fn"]
    return (c["tp"] / d) if d else 0.0


def _rgb(x10):
    rgb = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _false_color(x10):
    # B12 / B8A / B04（短波红外—红边—红），滑坡裸土通常更亮
    bands = np.stack([x10[9], x10[7], x10[2]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(bands, [2, 98], axis=(0, 1))
    return np.clip((bands - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _bf1(b):
    if b.get("status") != "ok":
        return None
    p = b["prec_tp"] / b["pred_px"] if b["pred_px"] else 0.0
    r = b["rec_tp"] / b["gt_px"] if b["gt_px"] else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _error_map(gt, pred):
    out = np.zeros(gt.shape + (3,), dtype=np.float32)
    out[..., 1] = (gt & pred).astype(np.float32)   # TP 绿
    out[..., 0] = (~gt & pred).astype(np.float32)  # FP 红
    out[..., 2] = (gt & ~pred).astype(np.float32)  # FN 蓝
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="CF1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "coarse_to_fine" / "base.yaml"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_segformer_config(args.config)["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    ckpt = root / args.experiment / str(args.seed) / "best.pt"
    model, meta = load_checkpoint(ckpt, device="cuda:0")
    model.eval()
    records = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), args.split)
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    ds = S2SegDataset(records, stats, preload=True)

    rows = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            batch = collate([ds[j] for j in range(i, min(i + 32, len(ds)))])
            out = model(batch["image"].cuda())
            fine = torch.sigmoid(out["logits"].float()[:, 0]).cpu().numpy()
            coarse = torch.sigmoid(out["base_logits"].float()[:, 0]).cpu().numpy()
            gt = batch["mask"].numpy().astype(bool)
            valid = batch["valid"].numpy().astype(bool)
            for k, sid in enumerate(batch["sample_id"]):
                ci = confusion(coarse[k] >= 0.5, gt[k], valid[k])
                fi = confusion(fine[k] >= 0.5, gt[k], valid[k])
                cb = evaluate_tile_v2(
                    coarse[k], gt[k], valid[k], threshold=0.5, iou_threshold=0.25,
                    iou_strict=0.5, boundary_tol=1, sample_id=sid, event_id=batch["event_id"][k],
                )["boundary"]
                fb = evaluate_tile_v2(
                    fine[k], gt[k], valid[k], threshold=0.5, iou_threshold=0.25,
                    iou_strict=0.5, boundary_tol=1, sample_id=sid, event_id=batch["event_id"][k],
                )["boundary"]
                rows.append(
                    {
                        "idx": i + k,
                        "sample_id": sid,
                        "coarse_iou": _iou(coarse[k], gt[k], valid[k]),
                        "fine_iou": _iou(fine[k], gt[k], valid[k]),
                        "coarse_fp": ci["fp"],
                        "fine_fp": fi["fp"],
                        "coarse_bf1": _bf1(cb),
                        "fine_bf1": _bf1(fb),
                        "gt_px": int(gt[k].sum()),
                    }
                )

    picks = []
    rec = [r for r in rows if r["coarse_iou"] == 0 and r["fine_iou"] > 0.2]
    if rec:
        picks.append(("coarse miss -> fine recovers", max(rec, key=lambda r: r["fine_iou"])))
    fp = [r for r in rows if r["coarse_fp"] > 0]
    if fp:
        picks.append(("coarse FP reduced", max(fp, key=lambda r: r["coarse_fp"] - r["fine_fp"])))
    worse = [r for r in rows if r["coarse_iou"] > r["fine_iou"]]
    if worse:
        picks.append(("fine worse than coarse", max(worse, key=lambda r: r["coarse_iou"] - r["fine_iou"])))
    bfix = [
        r for r in rows
        if r["fine_bf1"] is not None and r["coarse_bf1"] is not None
        and r["fine_bf1"] > r["coarse_bf1"]
    ]
    if bfix:
        picks.append(("boundary improved", max(bfix, key=lambda r: r["fine_bf1"] - r["coarse_bf1"])))
    if not picks:
        picks.append(("largest GT", max(rows, key=lambda r: r["gt_px"])))

    fig, axes = plt.subplots(len(picks), 8, figsize=(21, 2.9 * len(picks)), squeeze=False)
    for row, (label, r) in enumerate(picks):
        i = r["idx"]
        d = np.load(records[i]["cache"])
        x10 = d["x10"].astype(np.float32)
        gt = d["mask"].astype(bool)
        with torch.no_grad():
            b = collate([S2SegDataset([records[i]], stats, preload=False)[0]])
            o = model(b["image"].cuda())
            cp = torch.sigmoid(o["base_logits"].float()[0, 0]).cpu().numpy()
            fp_ = torch.sigmoid(o["logits"].float()[0, 0]).cpu().numpy()
        panels = [
            (_rgb(x10), f"{r['sample_id']} RGB"),
            (_false_color(x10), "false color B12/B8A/B04"),
            (gt, f"GT {r['gt_px']} px"),
            (cp, "coarse prob"),
            (cp >= 0.5, f"coarse IoU={r['coarse_iou']:.3f}"),
            (fp_ >= 0.5, f"fine IoU={r['fine_iou']:.3f}"),
            (_error_map(gt, fp_ >= 0.5), "error: TP green / FP red / FN blue"),
            (np.abs(fp_ - cp), "|fine prob - coarse prob|"),
        ]
        for j, (img, title) in enumerate(panels):
            ax = axes[row][j]
            if img.ndim == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap="magma" if j in (3, 7) else "gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row][0].set_ylabel(label)

    out_dir = resolve_path("outputs/segformer/coarse_to_fine")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.experiment}_{args.split}_stage_cases.png", dpi=110)
    plt.close(fig)
    (out_dir / f"{args.experiment}_{args.split}_stage_cases.json").write_text(
        json.dumps([{"label": l, **r} for l, r in picks], indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps([{"label": l, **r} for l, r in picks], indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir / f'{args.experiment}_{args.split}_stage_cases.png'}")


if __name__ == "__main__":
    main()

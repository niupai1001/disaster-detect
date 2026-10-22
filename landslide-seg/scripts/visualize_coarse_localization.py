"""粗定位结果图：粗概率、粗 mask、候选预算区域与最终 mask。

案例（确定性选择）：
1. 小目标被粗阶段命中；
2. 粗阶段漏检、最终输出找回；
3. 粗阶段大块误检（小 GT）；
4. 大目标、粗阶段覆盖较好。

用法::

    python scripts/visualize_coarse_localization.py --experiment CF1 --split val
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
from landslide_seg.training.segformer_metrics import evaluate_tile_v2  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402

COARSE_THR = 0.4  # P0 在 Hiroshima 上冻结的定位阈值
FINAL_THR = 0.5


def _rgb(x10):
    rgb = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _overlay(rgb, cand, gt, color=(1.0, 0.1, 0.1), alpha=0.45):
    out = rgb.copy()
    out[cand] = (1 - alpha) * out[cand] + alpha * np.array(color)
    # GT 轮廓用亮绿
    edge = np.zeros_like(gt)
    edge[1:, :] |= gt[1:, :] & ~gt[:-1, :]
    edge[:-1, :] |= gt[:-1, :] & ~gt[1:, :]
    edge[:, 1:] |= gt[:, 1:] & ~gt[:, :-1]
    edge[:, :-1] |= gt[:, :-1] & ~gt[:, 1:]
    out[edge] = (0.1, 1.0, 0.1)
    return out


def _iou(prob, gt, valid, thr):
    tp = int(((prob >= thr) & gt & valid).sum())
    fp = int(((prob >= thr) & ~gt & valid).sum())
    fn = int(((prob < thr) & gt & valid).sum())
    d = tp + fp + fn
    return tp / d if d else 0.0, tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="CF1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_segformer_config(ROOT / "configs" / "segformer" / "coarse_to_fine" / "base.yaml")["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    model, meta = load_checkpoint(root / args.experiment / str(args.seed) / "best.pt", device=args.device)
    model.eval()
    recs = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), args.split)
    ds = S2SegDataset(recs, load_stats(resolve_path(cfg["data"]["stats"])), preload=True)

    rows = []
    cache = {}
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            b = collate([ds[j] for j in range(i, min(i + 32, len(ds)))])
            out = model(b["image"].to(args.device))
            c = torch.sigmoid(out["base_logits"].float()[:, 0]).cpu().numpy()
            f = torch.sigmoid(out["logits"].float()[:, 0]).cpu().numpy()
            gt = b["mask"].numpy().astype(bool)
            valid = b["valid"].numpy().astype(bool)
            for k, sid in enumerate(b["sample_id"]):
                ev = evaluate_tile_v2(
                    c[k], gt[k], valid[k], threshold=COARSE_THR, iou_threshold=0.25,
                    iou_strict=0.5, boundary_tol=1, sample_id=sid, event_id=b["event_id"][k],
                )
                fi, _, _, _ = _iou(f[k], gt[k], valid[k], FINAL_THR)
                ci, _, _, _ = _iou(c[k], gt[k], valid[k], COARSE_THR)
                small = ev["bins"]["small"]
                rows.append({
                    "idx": i + k, "sample_id": sid, "gt_px": int(gt[k].sum()),
                    "coarse_iou": ci, "final_iou": fi,
                    "coarse_fp": ev["fp"], "small_matched": small.get("n_matched", 0) if small else 0,
                    "small_total": small.get("n_objects", 0) if small else 0,
                    "event": b["event_id"][k],
                })
                cache[i + k] = (c[k], f[k], gt[k], valid[k])

    picks = []
    small_hit = [r for r in rows if 0 < r["gt_px"] < 80 and r["coarse_iou"] > 0.2]
    if small_hit:
        picks.append(("small object localized (coarse)", max(small_hit, key=lambda r: r["coarse_iou"])))
    else:
        rescue = [r for r in rows if 0 < r["gt_px"] < 80 and r["final_iou"] > 0.2]
        if rescue:
            picks.append(("small object found by final", max(rescue, key=lambda r: r["final_iou"])))
    miss = [r for r in rows if r["coarse_iou"] == 0 and r["final_iou"] > 0.2]
    if miss:
        picks.append(("coarse miss, final recovers", max(miss, key=lambda r: r["final_iou"])))
    fp = [r for r in rows if r["gt_px"] < 200 and r["coarse_fp"] > 500]
    if fp:
        picks.append(("coarse false positive", max(fp, key=lambda r: r["coarse_fp"])))
    large = [r for r in rows if r["gt_px"] > 500]
    if large:
        picks.append(("large object covered", max(large, key=lambda r: r["coarse_iou"])))
    if len(picks) < args.rows and rows:
        med = sorted(rows, key=lambda r: r["coarse_iou"])[len(rows) // 2]
        picks.append(("median coarse IoU tile", med))
    for r in rows:
        if len(picks) >= args.rows:
            break
        if r not in [p for _, p in picks]:
            picks.append(("typical tile", r))
    picks = picks[: args.rows]

    fig, axes = plt.subplots(len(picks), 7, figsize=(18.5, 2.9 * len(picks)), squeeze=False)
    for row, (label, r) in enumerate(picks):
        c, f, gt, valid = cache[r["idx"]]
        rgb = _rgb(ds[r["idx"]]["image"].numpy().astype(np.float32))
        v = valid
        n = int(v.sum())
        order = np.argsort(-c[v], kind="stable")
        cands = {}
        for b in (0.05, 0.20):
            k = max(1, int(round(n * b)))
            m = np.zeros(n, dtype=bool)
            m[order[:k]] = True
            cm = np.zeros_like(v)
            cm[v] = m
            cands[b] = cm
        panels = [
            (rgb, f"{r['sample_id']} RGB"),
            (gt, f"GT {r['gt_px']} px"),
            (c, f"coarse P (IoU={r['coarse_iou']:.3f})"),
            (c >= COARSE_THR, f"coarse mask @{COARSE_THR}"),
            (_overlay(rgb, cands[0.05], gt), "5% budget (red), GT edge (green)"),
            (_overlay(rgb, cands[0.20], gt), "20% budget (red), GT edge (green)"),
            (f >= FINAL_THR, f"final mask @{FINAL_THR} (IoU={r['final_iou']:.3f})"),
        ]
        for j, (img, title) in enumerate(panels):
            ax = axes[row][j]
            if img.ndim == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap="magma" if j == 2 else "gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row][0].set_ylabel(label)

    out_dir = resolve_path("outputs/segformer/coarse_localization")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = out_dir / f"{args.experiment}_{args.split}_localization.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)
    (out_dir / f"{args.experiment}_{args.split}_localization.json").write_text(
        json.dumps([{"label": l, **r} for l, r in picks], indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps([{"label": l, **{k: v for k, v in r.items()}} for l, r in picks], indent=1, ensure_ascii=False))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

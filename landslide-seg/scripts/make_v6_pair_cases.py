"""主模型对照案例：同一批 TEST 图块的 RGB / GT / 单期预测 / 多期预测 / 误差。

列：RGB、GT、单期二值预测、15 期二值预测、单期误差、15 期误差。
误差色：绿=TP、红=FP、蓝=FN。每行一个固定案例，行首标 sample_id、事件与两模型 IoU。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _rgb(x10):
    rgb = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _load(path):
    z = np.load(path)
    prob, gt, valid = z["probability"], z["gt"], z["valid"]
    ids, events = [str(x) for x in z["sample_ids"]], [str(x) for x in z["events"]]
    return {sid: (prob[i], gt[i], valid[i], events[i]) for i, sid in enumerate(ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="单期 predictions npz")
    ap.add_argument("--b", required=True, help="多期 predictions npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B = _load(args.a), _load(args.b)
    cache = {}
    for mf in ("manifests/official_single/trainval.jsonl", "manifests/official_single/test.jsonl"):
        p = ROOT / mf
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    cache[r["sample_id"]] = r["cache"]

    order, seen = [], set()
    for sid, (p, g, v, ev) in B.items():
        if ev in seen or sid not in A:
            continue
        if 120 <= int(g.sum()) <= 2500:
            order.append(sid); seen.add(ev)
        if len(order) >= args.n:
            break
    if not order:
        order = [sid for sid in list(B)[: args.n] if sid in A]

    fig, axes = plt.subplots(len(order), 6, figsize=(17, 2.9 * len(order)), squeeze=False)
    for row, sid in enumerate(order):
        pa, ga, va, ev = A[sid]
        pb, gb, vb, _ = B[sid]
        g = ga.astype(bool)
        va = va.astype(bool); vb = vb.astype(bool)
        pred_a = (pa >= args.threshold) & va
        pred_b = (pb >= args.threshold) & vb
        def iou(pred):
            tp = int((pred & g).sum()); fp = int((pred & ~g).sum()); fn = int((~pred & g & va).sum())
            return tp / max(tp + fp + fn, 1)
        def err(pred, valid):
            e = np.zeros((128, 128, 3), np.float32)
            e[..., 1] = (pred & g).astype(np.float32)
            e[..., 0] = (pred & ~g).astype(np.float32)
            e[..., 2] = (~pred & g & valid).astype(np.float32)
            return e
        x = np.load(cache[sid])["x10"].astype(np.float32)
        panels = [(_rgb(x), "RGB"), (g, "GT"), (pred_a, f"Single IoU={iou(pred_a):.3f}"),
                  (pred_b, f"15-frame IoU={iou(pred_b):.3f}"), (err(pred_a, va), "Single TP/FP/FN"),
                  (err(pred_b, vb), "15-frame TP/FP/FN")]
        for j, (img, title) in enumerate(panels):
            ax = axes[row][j]
            if img.ndim == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row][0].text(-0.06, 0.5, f"{sid}\n{ev}", transform=axes[row][0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=7)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

"""主模型/时序模型的固定案例图：RGB / GT / 概率 / 预测 / TP-FP-FN。

从 ``evaluate_temporal.py`` 保存的 predictions npz 生成，不重新推理。
每行一个样本；列依次为 RGB、GT、预测概率、二值预测、误差（绿=TP、红=FP、蓝=FN）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _rgb(x10: np.ndarray) -> np.ndarray:
    rgb = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="manifests/official_single/trainval.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.predictions)
    prob, gt, valid = z["probability"], z["gt"], z["valid"]
    ids = [str(x) for x in z["sample_ids"]]
    events = [str(x) for x in z["events"]]
    cache = {}
    for mf in (args.manifest, "manifests/official_single/test.jsonl"):
        p = Path(ROOT / mf)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["sample_id"]] = r["cache"]

    # 选每事件一个、GT 面积适中的正样本，保证可读性
    order = []
    seen = set()
    for i, (sid, ev) in enumerate(zip(ids, events)):
        if ev in seen:
            continue
        if 80 <= int(gt[i].sum()) <= 2000:
            order.append(i); seen.add(ev)
        if len(order) >= args.n:
            break
    if not order:
        order = list(range(min(args.n, len(ids))))

    fig, axes = plt.subplots(len(order), 5, figsize=(14, 2.9 * len(order)), squeeze=False)
    for row, i in enumerate(order):
        x = np.load(cache[ids[i]])["x10"].astype(np.float32)
        pred = (prob[i] >= args.threshold) & valid[i]
        g = gt[i].astype(bool)
        err = np.zeros((128, 128, 3), dtype=np.float32)
        err[..., 1] = (pred & g).astype(np.float32)
        err[..., 0] = (pred & ~g).astype(np.float32)
        err[..., 2] = (~pred & g & valid[i]).astype(np.float32)
        tp = int((pred & g).sum()); fp = int((pred & ~g).sum()); fn = int((~pred & g & valid[i]).sum())
        iou = tp / max(tp + fp + fn, 1)
        panels = [(_rgb(x), "RGB"), (g, "GT"), (prob[i], "prob"),
                  (pred, "pred"), (err, "TP / FP / FN")]
        for j, (img, title) in enumerate(panels):
            ax = axes[row][j]
            if img.ndim == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title if row == 0 else "", fontsize=9)
            ax.axis("off")
        axes[row][0].set_ylabel("")
        axes[row][0].text(-0.08, 0.5, f"{ids[i]}\n{events[i]}\nIoU={iou:.3f}",
                          transform=axes[row][0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=7)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

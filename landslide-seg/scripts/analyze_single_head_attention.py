"""N2 注意力 W 的分布与敏感性（计划 §8）。

- W 的总体/前景内部/边界/背景统计与饱和比例；
- 固定 checkpoint 的分布改变敏感性：W→0.5（退化为相加）、W→1（仅语义 2U）、W→0（仅细节 2V）；
- 只作敏感性检查，不是重训消融，不能单独当因果证明。
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
from landslide_seg.training.segformer_metrics import boundary_pixels, confusion  # noqa: E402
from landslide_seg.training.segformer_trainer import load_checkpoint  # noqa: E402


def _stats(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "sat_lo": float((x < 0.05).mean()),
        "sat_hi": float((x > 0.95).mean()),
    }


def _iou(prob, gt, valid):
    c = confusion(prob >= 0.5, gt, valid)
    d = c["tp"] + c["fp"] + c["fn"]
    return c["tp"] / d if d else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sensitivity-tiles", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_segformer_config(ROOT / "configs" / "segformer" / "single_head" / "base.yaml")["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    model, meta = load_checkpoint(root / "SINGLE_AFF" / str(args.seed) / "best.pt", device=args.device)
    model.eval()
    recs = filter_split(read_manifest(resolve_path(cfg["data"]["manifest"])), args.split)
    stats = load_stats(resolve_path(cfg["data"]["stats"]))
    ds = S2SegDataset(recs, stats, preload=True)
    device = torch.device(args.device)

    regions = {"fg_interior": [], "boundary": [], "bg": []}
    all_w = []
    tiles = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            b = collate([ds[j] for j in range(i, min(i + 32, len(ds)))])
            out = model(b["image"].to(device), return_attn=True)
            w = out["attn"].cpu().numpy()
            gt = b["mask"].numpy().astype(bool)
            valid = b["valid"].numpy().astype(bool)
            for k in range(w.shape[0]):
                all_w.append(w[k][:, valid[k]].ravel())
                g = gt[k] & valid[k]
                bd = boundary_pixels(g)
                bd = np.asarray(bd, dtype=bool)
                interior = g & ~bd
                bg = valid[k] & ~g
                if interior.any():
                    regions["fg_interior"].append(w[k][:, interior].ravel())
                if bd.any():
                    regions["boundary"].append(w[k][:, bd].ravel())
                if bg.any():
                    regions["bg"].append(w[k][:, bg].ravel())
                tiles.append((w[k], b["image"][k], gt[k], valid[k]))
    allw = np.concatenate(all_w) if all_w else np.zeros(0)

    report = {"split": args.split, "w_all": _stats(allw)}
    for name, arr in regions.items():
        report[name] = _stats(np.concatenate(arr) if arr else np.zeros(0))

    # 敏感性：W→0.5 / 1 / 0，比较 IoU
    base_scores, s05, s1, s0 = [], [], [], []
    for w, image, gt, valid in tiles[: args.sensitivity_tiles]:
        img = image.unsqueeze(0).to(device)
        with torch.no_grad():
            p0 = torch.sigmoid(model(img)["logits"].float()[0, 0]).cpu().numpy()
            p5 = torch.sigmoid(model(img, attn_override=0.5)["logits"].float()[0, 0]).cpu().numpy()
            p1 = torch.sigmoid(model(img, attn_override=1.0)["logits"].float()[0, 0]).cpu().numpy()
            pz = torch.sigmoid(model(img, attn_override=0.0)["logits"].float()[0, 0]).cpu().numpy()
        for arr, store in ((p0, base_scores), (p5, s05), (p1, s1), (pz, s0)):
            v = _iou(arr, gt, valid)
            if v is not None:
                store.append(v)
    def m(x):
        return float(np.mean(x)) if x else None
    report["sensitivity"] = {
        "n_tiles": len(base_scores),
        "iou_model": m(base_scores),
        "iou_w05_sum": m(s05),
        "iou_w1_semantic_only": m(s1),
        "iou_w0_detail_only": m(s0),
        "delta_w05": (m(s05) - m(base_scores)) if base_scores and s05 else None,
        "delta_w1": (m(s1) - m(base_scores)) if base_scores and s1 else None,
        "delta_w0": (m(s0) - m(base_scores)) if base_scores and s0 else None,
    }

    out_dir = resolve_path("outputs/segformer/single_head")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attention_analysis.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(allw, bins=60, range=(0, 1))
    axes[0].set_title("W distribution (all valid pixels)")
    axes[0].set_xlabel("W")
    for name, arr in regions.items():
        if arr:
            axes[1].hist(np.concatenate(arr), bins=60, range=(0, 1), histtype="step", label=name)
    axes[1].set_title("W by region")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"attn_W_{args.split}.png", dpi=120)
    plt.close(fig)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

"""数据集画廊：每个事件挑一个"代表性"瓦片（正像素数接近该事件中位数），
展示 RGB / 假彩色 / GT / valid 四个面板。

用法::

    python scripts/gallery_segformer_dataset.py
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

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import read_manifest  # noqa: E402


def stretch(rgb: np.ndarray) -> np.ndarray:
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "base.yaml"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_segformer_config(args.config)["segformer"]
    records = read_manifest(resolve_path(cfg["data"]["manifest"]))

    by_event: dict[str, list[dict]] = {}
    for r in records:
        by_event.setdefault(r["event_id"], []).append(r)

    chosen = []
    for event, rows in sorted(by_event.items()):
        pos = [r for r in rows if r["mask_px"] > 0]
        target = float(np.median([r["mask_px"] for r in pos])) if pos else 0.0
        best = min(pos or rows, key=lambda r: abs(r["mask_px"] - target))
        chosen.append((event, best))

    n = len(chosen)
    fig, axes = plt.subplots(n, 4, figsize=(12, 2.5 * n), squeeze=False)
    out_rows = []
    for i, (event, rec) in enumerate(chosen):
        d = np.load(rec["cache"])
        x10 = d["x10"].astype(np.float32)
        mask = d["mask"].astype(bool)
        valid = d["valid"].astype(bool)
        rgb = stretch(x10[[2, 1, 0]].transpose(1, 2, 0))
        false = stretch(x10[[9, 7, 2]].transpose(1, 2, 0))
        panels = [
            (rgb, f"{event} / {rec['sample_id']}  RGB", None),
            (false, "B12/B8A/B04", None),
            (mask.astype(np.uint8), f"GT  {rec['mask_px']} px ({rec['mask_px']/16384*100:.1f}%)", "gray"),
            (valid.astype(np.uint8), f"valid  {rec['valid_ratio']*100:.0f}%", "Greens"),
        ]
        for j, (img, title, cmap) in enumerate(panels):
            ax = axes[i][j]
            if img.ndim == 3:
                ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        out_rows.append(
            {
                "event": event,
                "split": rec["split"],
                "sample_id": rec["sample_id"],
                "date": rec["date"],
                "event_date": rec["event_date"],
                "days_after_event": rec["days_after_event"],
                "mask_px": rec["mask_px"],
                "mask_pct": round(rec["mask_px"] / 16384 * 100, 2),
                "valid_ratio": rec["valid_ratio"],
                "cloud_fraction": rec["cloud_fraction"],
            }
        )
    fig.suptitle("s2_10band_v1 — one representative tile per event (median positive area)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_dir = resolve_path("outputs/segformer") / cfg["data"]["version"] / "audit_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "dataset_gallery.png", dpi=110)
    plt.close(fig)
    (out_dir / "dataset_gallery.json").write_text(
        json.dumps(out_rows, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    for r in out_rows:
        print(f"{r['event']:15s} {r['split']:5s} {r['sample_id']:22s} "
              f"mask={r['mask_px']:5d}px ({r['mask_pct']:5.2f}%) valid={r['valid_ratio']:.3f} "
              f"days_after={r['days_after_event']:4d}")
    print(f"[saved] {out_dir / 'dataset_gallery.png'}")


if __name__ == "__main__":
    main()

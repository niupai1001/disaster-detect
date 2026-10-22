"""训练曲线：A1 / A1-AUX / CF1 / CF0 / U0 的 loss 与 val IoU。"""

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

from landslide_seg.config import load_segformer_config, resolve_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "segformer" / "coarse_to_fine" / "base.yaml"))
    ap.add_argument("--experiments", nargs="+", default=["A1", "A1_AUX", "CF1", "CF0", "CF1_B", "U0"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_segformer_config(args.config)["segformer"]
    root = resolve_path(cfg["output_dir"]) / cfg["data"]["version"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for exp in args.experiments:
        path = root / exp / str(args.seed) / "metrics.jsonl"
        if not path.is_file():
            print(f"[skip] {path}")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["train_loss"] for r in rows], label=exp)
        axes[1].plot(epochs, [r["val_iou"] for r in rows], label=exp)
    axes[0].set_title("train loss (fine + 0.5*coarse)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("val IoU (hiroshima, best-checkpoint metric)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("IoU")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_dir = resolve_path("outputs/segformer/coarse_to_fine")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

"""固定案例对比：参考影像、GT 与四组 best 模型的概率图。

案例清单在首批训练前由参考单期 run 的错误选定（outputs/.../case_list.json），
不随新模型结果更换。用法::

    python scripts/make_temporal_control_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.config import resolve_path  # noqa: E402
from landslide_seg.data.segformer_dataset import load_stats, read_manifest  # noqa: E402
from landslide_seg.data.temporal_v6 import TemporalV6, collate_v6  # noqa: E402
from landslide_seg.models.spectral_temporal import build_temporal_model  # noqa: E402

OUT = ROOT / "outputs/segformer/v4r1_temporal_control"
ASSETS = ROOT / "docs/assets/research"
LABEL = {"single": "single T=1", "mean": "mean 15", "content": "content 15", "date": "content+date 15"}
RUN = {
    "single": "SegFormer-B0-Temporal-Single",
    "mean": "SegFormer-B0-Temporal-Mean15",
    "content": "SegFormer-B0-Temporal-Content15",
    "date": "SegFormer-B0-Temporal-Date15",
}
STATS = load_stats(ROOT / "manifests/v4r1/stats_core25.json")
BY_ID = {r["sample_id"]: r for r in read_manifest(ROOT / "manifests/v4r1/pool_100.jsonl")}


def rgb(x10: np.ndarray) -> np.ndarray:
    img = np.stack([x10[2], x10[1], x10[0]], axis=-1).astype(np.float32)
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def load_model(key: str, device: torch.device):
    ckpt = torch.load(OUT / RUN[key] / "1234" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]["segformer"]
    model = build_temporal_model(cfg, STATS)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict(model, cfg, sample_id: str, device: torch.device):
    row = BY_ID[sample_id]
    ds = TemporalV6([row], STATS, mode=cfg["temporal"]["mode"], augment=False, seed=int(cfg["seed"]),
                    root=str(resolve_path(cfg["data"]["temporal_root"])),
                    quality=str(resolve_path(cfg["data"]["quality"])))
    batch = collate_v6([ds[0]])
    logits = model(batch["image"].to(device), batch["dates"].to(device), batch["keep"].to(device),
                   batch["anchor"].to(device))["logits"]
    prob = logits[0, 0].sigmoid().cpu().numpy()
    with np.load(resolve_path(row["cache"])) as z:
        x10 = np.asarray(z["x10"])          # 单期参考影像 (10,H,W)
        mask = np.asarray(z["mask"]).astype(bool)
    return prob, mask, x10


def panel(ax, image, mask=None, prob=None, title=""):
    ax.imshow(image)
    if mask is not None and mask.any():
        ax.contour(mask, levels=[0.5], colors=["#ffe14d"], linewidths=1.2)
    if prob is not None:
        ax.imshow(np.ma.masked_less(prob, 0.5), cmap="Reds", alpha=0.45, vmin=0.5, vmax=1.0)
        ax.contour(prob, levels=[0.5], colors=["#ff3b30"], linewidths=1.1)
    ax.set_title(title, fontsize=8.5)
    ax.axis("off")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = json.loads((OUT / "case_list.json").read_text(encoding="utf-8"))["cases"]
    models = {key: load_model(key, device) for key in RUN}
    for event in ("hiroshima", "thrissur"):
        rows = [c for c in cases if c["event_id"] == event]
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3.0 * len(rows)))
        if len(rows) == 1:
            axes = axes[None]
        for i, case in enumerate(rows):
            base = None
            for j, key in enumerate(("single", "mean", "content", "date"), start=1):
                prob, mask, ref = predict(models[key][0], models[key][1], case["sample_id"], device)
                if base is None:
                    base = rgb(ref)
                    gt = mask
                tp = float(((prob >= 0.5) & gt).sum())
                fp = float(((prob >= 0.5) & ~gt).sum())
                fn = float((~((prob >= 0.5)) & gt).sum())
                iou = tp / max(tp + fp + fn, 1)
                panel(axes[i, j], base, mask=gt, prob=prob,
                      title=f"{LABEL[key]} | IoU={iou:.3f}")
            panel(axes[i, 0], base, mask=gt,
                  title=f"{case['kind']}\n{case['sample_id']} | reference + GT (yellow)")
        fig.suptitle(f"Fixed cases ({event}); columns are fixed before the four runs, "
                     f"not chosen by new results", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(ASSETS / f"temporal_control_cases_{event}.png", dpi=150)
        plt.close(fig)
        print("wrote", ASSETS / f"temporal_control_cases_{event}.png")


if __name__ == "__main__":
    main()

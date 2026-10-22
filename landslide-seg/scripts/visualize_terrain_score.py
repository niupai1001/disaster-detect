"""查看地形分支的输出：影像、DEM、坡度、地形修正项、视觉与融合概率、标注。

数据：event_ood_v4r1 的 DEV 集合（广岛 448 块 + 特里苏尔 221 块），只用于展示，
不参与任何选择。模型：地形自监督预训练 + 可学习 α 的那一组（TerrainSSL）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import load_segformer_config  # noqa: E402
from landslide_seg.data.segformer_dataset import S2SegDataset, load_stats, read_manifest  # noqa: E402
from landslide_seg.models.factory import build_model  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RUN = ROOT / "outputs/segformer/s2_10band_v4r1_p20_geo/SegFormer-B0-LateDetail-OODv4-TerrainSSL/1234"


def rgb(x10: np.ndarray, stats: dict, lo_pct=2, hi_pct=98) -> np.ndarray:
    mean = np.asarray(stats["mean"])[:10, None, None]
    std = np.asarray(stats["std"])[:10, None, None]
    dn = x10[:10].astype(np.float32) * std + mean
    img = np.stack([dn[2], dn[1], dn[0]], axis=-1)          # B04/B03/B02
    lo = np.percentile(img, lo_pct, axis=(0, 1))
    hi = np.percentile(img, hi_pct, axis=(0, 1))
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="docs/assets/research/terrain_score_examples.png")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_segformer_config(run / "resolved_config.json")
    s = cfg["segformer"]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(s).to(dev).eval()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])

    rows = [r for r in read_manifest(ROOT / s["data"]["manifest"]) if r["split"] == "val"]
    stats = load_stats(ROOT / s["data"]["stats"])
    ds = S2SegDataset(rows, stats, augment=False, preload=False,
                      extra="both", sidecar=s["data"].get("extra_sidecar"))

    # 按前景面积挑选：大、小、零，覆盖不同情形
    areas = []
    for i, r in enumerate(rows):
        item = ds[i]
        areas.append((int(item["mask"].sum()), i))
    areas.sort(reverse=True)
    picks = [areas[0][1], areas[len(areas) // 20][1], areas[len(areas) * 3 // 5][1], areas[-3][1]]
    picks = picks[: args.n]

    dem_mean = float(np.asarray(stats["mean"])[10])
    dem_std = float(np.asarray(stats["std"])[10])
    slp_mean = float(np.asarray(stats["mean"])[11])
    slp_std = float(np.asarray(stats["std"])[11])

    ncol = 7
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3.0 * ncol, 3.15 * len(picks)))
    if len(picks) == 1:
        axes = axes[None]
    titles = ["S2 真彩色（B04/B03/B02）", "DEM（米）", "坡度（度）",
              "地形修正项 α·tanh(评分)", "仅视觉分支概率", "融合后概率", "人工标注 / 有效区"]

    for row_i, idx in enumerate(picks):
        item = ds[idx]
        x = item["image"][None].to(dev)
        with torch.no_grad():
            out = model(x)
        alpha = float(out["alpha"])
        corr = (alpha * torch.tanh(out["terrain_score"]))[0, 0].cpu().numpy()
        prob_fused = torch.sigmoid(out["logits"])[0, 0].cpu().numpy()
        prob_vis = torch.sigmoid(out["prototype_logits"])[0, 0].cpu().numpy()
        x10 = item["image"].numpy()
        dem = x10[10] * dem_std + dem_mean
        slp = x10[11] * slp_std + slp_mean
        mask = item["mask"].numpy()
        valid = item["valid"].numpy()

        gt = np.zeros((*mask.shape, 3))
        gt[..., 1] = mask * 0.75
        gt[~valid] = [0.55, 0.55, 0.55]

        panels = [
            (rgb(x10, stats), None, None),
            (dem, "terrain", None),
            (slp, "magma", None),
            (corr, "coolwarm", None),
            (prob_vis, "magma", (0, 1)),
            (prob_fused, "magma", (0, 1)),
            (gt, None, None),
        ]
        for col, (data, cmap, rng) in enumerate(panels):
            ax = axes[row_i, col]
            if col == 6:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=None if rng is None else rng[0],
                               vmax=None if rng is None else rng[1])
                if rng is not None:
                    ax.set_facecolor("black")
            if row_i == 0:
                ax.set_title(titles[col], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{rows[idx]['sample_id']}\nmask {int(mask.sum())} px", fontsize=8)
            if col in (1, 2, 3):
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=6)
        if row_i == len(picks) - 1:
            pass

    fig.suptitle(f"地形分支输出（{s['experiment']}，DEV 集合，α={alpha:.3f}）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    print(out_path)

    meta = {"run": str(run.relative_to(ROOT)).replace("\\", "/"), "alpha": alpha,
            "samples": [rows[i]["sample_id"] for i in picks],
            "event_ids": [rows[i]["event_id"] for i in picks],
            "mask_px": [int(ds[i]["mask"].sum()) for i in picks],
            "split": "DEV(val)", "checkpoint": "best.pt"}
    (out_path.parent / "terrain_score_examples.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

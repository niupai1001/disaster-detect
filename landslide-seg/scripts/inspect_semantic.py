"""生成两张检查图：

1) 语义检查图：10 个真实样本 × （RGB / GT / 语义相似度图 / 属性响应条）
2) 骨干输出图：TerraMind 四层 token 与 AnySat 两尺度 dense 特征的 PCA 伪彩色

用途：让人肉眼判断"语义到底给了什么"以及"两支骨干输出了什么"。
注意：当前权重未训练，属性响应是 bias 主导的常数，图上会直接看出来。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from landslide_seg.config import load_config, set_seed  # noqa: E402
from landslide_seg.data.band_mapping import normalize_anysat  # noqa: E402
from landslide_seg.data.certified import filter_records, load_sample, read_manifest  # noqa: E402
from landslide_seg.models.build import build_from_config  # noqa: E402
from landslide_seg.models.semantic_dynamic import ATTRIBUTE_VOCAB  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pca_rgb(feat: np.ndarray, remove_spatial_mean: bool = False) -> tuple[np.ndarray, float]:
    """[C,H,W] -> ([H,W,3], 前三主成分解释方差比)。

    remove_spatial_mean=True 时先减去每个通道的空间均值，
    这样伪彩色反映的是**空间结构**，而不是通道间的常数差异。
    """
    c, h, w = feat.shape
    if remove_spatial_mean:
        feat = feat - feat.mean(axis=(1, 2), keepdims=True)
    x = feat.reshape(c, -1).T
    x = x - x.mean(0)
    u, sv, vt = np.linalg.svd(x, full_matrices=False)
    ratio = float((sv[:3] ** 2).sum() / (sv ** 2).sum())
    comp = (x @ vt[:3].T)
    comp = comp / (comp.std(0, keepdims=True) + 1e-9)
    img = comp.reshape(h, w, 3)
    img = np.clip((img + 2.5) / 5.0, 0, 1)
    return img, ratio


def rgb_of(x12: np.ndarray) -> np.ndarray:
    rgb = np.stack([x12[3], x12[2], x12[1]], -1).astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    set_seed(1234)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_from_config(cfg, device=device)
    model.eval()
    names = list(ATTRIBUTE_VOCAB)

    recs = filter_records(read_manifest(ROOT / "manifests" / "certified.jsonl"), split=args.split)[: args.n]
    rows = []
    for rec in recs:
        s = load_sample(rec)
        x12 = torch.from_numpy(s["x12"])[None].to(device)
        x10 = torch.from_numpy(s["x10"])[None].to(device)
        dates = torch.tensor([[float(s["doy"])]], device=device)
        valid = torch.from_numpy(s["valid"])[None].to(device)
        with torch.no_grad():
            out = model(x12, x10, dates, return_features=True)
            sem = model.semantic(
                out.features["pyramid"][0],
                torch.sigmoid(out.coarse_logits),
                valid.float().unsqueeze(1),
                spectral=x10,
            )
        rows.append({
            "rec": rec, "rgb": rgb_of(s["x12"]), "gt": s["mask"],
            "sem": sem.sem_map[0, 0].cpu().numpy(),
            "attr": torch.sigmoid(sem.attr_logits)[0].cpu().numpy(),
            "dyn_cos_cls": float(sem.dyn[0] @ model.semantic.class_query[0]),
        })

    # ---------- 图 1：语义检查 ----------
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 2.5 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    for i, r in enumerate(rows):
        axes[i, 0].imshow(r["rgb"])
        axes[i, 0].set_title(f"{r['rec']['sample_id']}  {r['rec']['event_id']}", fontsize=9)
        axes[i, 1].imshow(r["gt"], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"GT  {int(r['gt'].sum())} px", fontsize=9)
        im = axes[i, 2].imshow(r["sem"], cmap="viridis")
        axes[i, 2].set_title(
            f"sem_map [{r['sem'].min():+.2f},{r['sem'].max():+.2f}]  cos(dyn,cls)={r['dyn_cos_cls']:.2f}",
            fontsize=8)
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
        axes[i, 3].barh(range(len(names)), r["attr"], color="#378ADD")
        axes[i, 3].set_yticks(range(len(names)))
        axes[i, 3].set_yticklabels(names, fontsize=6)
        axes[i, 3].set_xlim(0, 1)
        axes[i, 3].axvline(0.5, color="#888780", lw=0.6, ls="--")
        axes[i, 3].set_title(f"attribute σ  (std across samples={np.std([x['attr'] for x in rows], axis=0).mean():.4f})",
                             fontsize=8)
        for ax in axes[i, :3]:
            ax.axis("off")
    fig.suptitle("语义检查：10 个真实认证样本（未训练权重——属性条几乎相同即为问题所在）", fontsize=11)
    fig.tight_layout()
    p1 = ROOT / "outputs" / "visualizations" / "semantic_inspect_10.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # ---------- 图 2：骨干输出 ----------
    rec = rows[0]["rec"]
    s = load_sample(rec)
    x12 = torch.from_numpy(s["x12"])[None].to(device)
    x10 = torch.from_numpy(s["x10"])[None].to(device)
    dates = torch.tensor([[float(s["doy"])]], device=device)
    with torch.no_grad():
        tm = model.terramind(torch.from_numpy(s["x12"])[None].to(device))
        f40 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates, patch_size_m=40)
        f160 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates, patch_size_m=160)

    feats = []
    for li, tok in enumerate(tm["tokens"]):
        feat = tok[0].transpose(0, 1).reshape(-1, *tm["grid"]).cpu().numpy()
        up = torch.nn.functional.interpolate(
            torch.from_numpy(feat)[None], size=(128, 128), mode="bilinear"
        )[0].numpy()
        feats.append((up, f"TerraMind L{model.terramind.layers[li]} 14x14x384"))
    feats.append((f40["feature"][0].cpu().numpy(), "AnySat 40m 1536x128x128"))
    feats.append((f160["feature"][0].cpu().numpy(), "AnySat 160m 1536x128x128"))

    fig, axes = plt.subplots(2, len(feats) + 1, figsize=(3.1 * (len(feats) + 1), 7.0))
    axes[0, 0].imshow(rows[0]["rgb"]); axes[0, 0].set_title("input RGB", fontsize=9); axes[0, 0].axis("off")
    axes[1, 0].imshow(rows[0]["gt"], cmap="gray"); axes[1, 0].set_title("GT mask", fontsize=9); axes[1, 0].axis("off")

    pca_ratio = {}
    for j, (feat, title) in enumerate(feats, start=1):
        img_raw, r1 = pca_rgb(feat, remove_spatial_mean=False)
        img_sub, r2 = pca_rgb(feat, remove_spatial_mean=True)
        axes[0, j].imshow(img_raw)
        axes[0, j].set_title(f"{title}\nraw PCA (PC1-3 var {r1:.1%})", fontsize=7.5)
        axes[1, j].imshow(img_sub)
        axes[1, j].set_title(f"minus spatial mean (var {r2:.1%})", fontsize=7.5)
        for r in (0, 1):
            axes[r, j].axis("off")
        pca_ratio[title] = {"raw": round(r1, 4), "demanded": round(r2, 4)}

    fig.suptitle(f"骨干实际输出（PCA 伪彩色）— {rec['sample_id']}；上排=原始，下排=去掉通道空间均值"
                 "（才看得出空间结构）", fontsize=10)
    fig.tight_layout()
    p2 = ROOT / "outputs" / "visualizations" / "backbone_features.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    summary = {
        "n_samples": len(rows),
        "attr_std_across_samples": {names[k]: round(float(np.std([r["attr"][k] for r in rows])), 6)
                                    for k in range(len(names))},
        "attr_mean": {names[k]: round(float(np.mean([r["attr"][k] for r in rows])), 4)
                      for k in range(len(names))},
        "sem_map_ranges": [[round(float(r["sem"].min()), 3), round(float(r["sem"].max()), 3)] for r in rows],
        "dyn_cos_to_class": [round(r["dyn_cos_cls"], 4) for r in rows],
        "pca_variance_ratio": pca_ratio,
        "figures": [str(p1), str(p2)],
    }
    out = ROOT / "outputs" / "logs" / "semantic_inspect.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("[saved]", p1)
    print("[saved]", p2)


if __name__ == "__main__":
    main()

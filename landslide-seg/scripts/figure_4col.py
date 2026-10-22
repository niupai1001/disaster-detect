"""四列对照图：文本 | TerraMind 输出 | AnySat 输出 | 原始影像假彩色。

每行一个真实样本。四列分别是：
  ① 文本：11 条固定字符串（类别 + 10 属性），并给出它们与**该样本区域裁剪**的 CLIP 余弦
  ② TerraMind：4 层 token（L2/L5/L8/L11，14×14×384）的 PCA 伪彩色，横向拼接
  ③ AnySat：40 m 与 160 m 两档 dense（1536×128×128）的 PCA 伪彩色
  ④ 假彩色：B08(NIR)–B04(R)–B03(G) 合成，黄线为该样本的 GT 滑坡轮廓

用法: python scripts/figure_4col.py --n 6
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
from landslide_seg.models.build import DEFAULT_WEIGHTS, build_from_config  # noqa: E402
from landslide_seg.models.semantic_dynamic import ATTRIBUTE_VOCAB  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

CLASS_TEXT = "a landslide scar in a satellite image"


def rgb_of(x12, idx):
    rgb = np.stack([x12[idx[0]], x12[idx[1]], x12[idx[2]]], -1).astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def pca3(feat: torch.Tensor, q: int = 3) -> np.ndarray:
    """[C,H,W] -> [H,W]：PC1 的灰度图（比彩色更好读结构）。

    彩色 PCA 每个通道独立归一会让各层颜色含义不一致，看起来像噪声；
    PC1 灰度能直接看出"这一层的主要空间变化在哪"。
    """
    c, h, w = feat.shape
    x = feat.reshape(c, -1).T                     # [N,C]
    sub = x[::4]
    sub = sub - sub.mean(0, keepdim=True)
    u, s, v = torch.pca_lowrank(sub, q=q, niter=2)
    basis = v[:, :q]                              # [C,q]
    proj = (x - sub.mean(0, keepdim=True)) @ basis
    pc1 = proj[:, 0].reshape(h, w)
    lo, hi = torch.quantile(pc1.flatten(), torch.tensor([0.02, 0.98], device=pc1.device))
    return torch.clamp((pc1 - lo) / (hi - lo + 1e-9), 0, 1).cpu().numpy()


def bbox_of(mask: np.ndarray, pad: int = 3):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return x0, y0, x1, y1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    set_seed(1234)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import clip
    from PIL import Image

    clip_model, preprocess = clip.load(str(DEFAULT_WEIGHTS["remoteclip"]), device=device, jit=False)
    clip_model.eval()
    names = list(ATTRIBUTE_VOCAB)
    texts = [CLASS_TEXT] + [ATTRIBUTE_VOCAB[k] for k in names]
    with torch.no_grad():
        txt_f = clip_model.encode_text(clip.tokenize(texts).to(device)).float()
        txt_f /= txt_f.norm(dim=-1, keepdim=True)

    model = build_from_config(cfg, device=device)
    model.eval()

    # 跨事件取样，避免只看到一个场景
    all_recs = read_manifest(ROOT / "manifests" / "certified.jsonl")
    by_event = {}
    for r in all_recs:
        by_event.setdefault(r["event_id"], []).append(r)
    picked = []
    while len(picked) < args.n:
        for ev in sorted(by_event):
            if by_event[ev] and len(picked) < args.n:
                picked.append(by_event[ev].pop(0))

    rows = []
    for rec in picked:
        s = load_sample(rec, boa_offset=float(cfg["data"].get("boa_offset_dn", 0.0)))
        x12 = s["x12"]
        box = bbox_of(s["mask"])
        x12t = torch.from_numpy(x12)[None].to(device)
        x10 = torch.from_numpy(s["x10"])[None].to(device)
        dates = torch.tensor([[float(s["doy"])]], device=device)
        with torch.no_grad():
            tm = model.terramind(x12t)
            f40 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates, patch_size_m=40)
            f160 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates, patch_size_m=160)

        tm_imgs = []
        for tok in tm["tokens"]:
            f = tok[0].transpose(0, 1).reshape(-1, *tm["grid"])
            up = torch.nn.functional.interpolate(f[None], size=(128, 128), mode="bilinear")[0]
            tm_imgs.append(pca3(up))
        # dense = [上下文(块内恒定) | 局部(10 m)]，两段分开画
        as_imgs = [
            pca3(f40["split"]["context"][0]), pca3(f40["split"]["local"][0]),
            pca3(f160["split"]["context"][0]), pca3(f160["split"]["local"][0]),
        ]

        # CLIP：区域裁剪 vs 文本
        sims = np.zeros(len(texts))
        if box is not None:
            crop = rgb_of(x12, (3, 2, 1))[box[1]:box[3], box[0]:box[2]]
            if crop.shape[0] >= 8 and crop.shape[1] >= 8:
                pil = Image.fromarray((crop * 255).astype(np.uint8))
                with torch.no_grad():
                    im = clip_model.encode_image(preprocess(pil).unsqueeze(0).to(device)).float()
                    im /= im.norm(dim=-1, keepdim=True)
                    sims = (im @ txt_f.T)[0].cpu().numpy()
        rows.append({"rec": rec, "x12": x12, "mask": s["mask"], "box": box,
                     "tm": tm_imgs, "as": as_imgs, "sims": sims})

    n = len(rows)
    fig = plt.figure(figsize=(21, 2.75 * n))
    gs = fig.add_gridspec(n, 4, width_ratios=[1.30, 1.30, 1.50, 0.95],
                          hspace=0.28, wspace=0.06)
    for i, r in enumerate(rows):
        # ① 文本
        ax = fig.add_subplot(gs[i, 0]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.0, 1.0, f'{r["rec"]["sample_id"]}  ({r["rec"]["event_id"]})',
                fontsize=8, va="top", weight="bold")
        y = 0.90
        ax.text(0.0, y, f'"{CLASS_TEXT}"', fontsize=6.6, va="top", weight="bold")
        ax.text(0.98, y, f'{r["sims"][0]:+.3f}', fontsize=6.6, va="top", ha="right", family="monospace")
        y -= 0.082
        for k, nm in enumerate(names):
            ax.text(0.02, y, f'"{ATTRIBUTE_VOCAB[nm]}"', fontsize=6.0, va="top")
            ax.text(0.98, y, f'{r["sims"][k+1]:+.3f}', fontsize=6.0, va="top", ha="right",
                    family="monospace")
            y -= 0.075

        # ② TerraMind（四层拼接）
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(np.concatenate(r["tm"], axis=1), cmap="gray")
        for j, nm in enumerate(["L2", "L5", "L8", "L11"]):
            ax.text(64 + j * 128, 6, nm, fontsize=7, ha="center",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        ax.axis("off")

        # ③ AnySat（两尺度拼接）
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(np.concatenate(r["as"], axis=1), cmap="gray")
        for j, nm in enumerate(["40m ctx", "40m local", "160m ctx", "160m local"]):
            ax.text(64 + j * 128, 6, nm, fontsize=7, ha="center",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        ax.axis("off")

        # ④ 假彩色 NIR-R-G + GT 轮廓
        ax = fig.add_subplot(gs[i, 3])
        fc = rgb_of(r["x12"], (11, 8, 3))     # B12(SWIR2), B8A(NIR), B04(R)
        ax.imshow(fc)
        if r["mask"].sum() > 0:
            ax.contour((r["mask"] > 0).astype(float), levels=[0.5], colors="#00E5FF", linewidths=1.4)
        ax.axis("off")

    fig.suptitle("四列对照：① 文本字符串（右侧为与区域裁剪的 CLIP 余弦） ｜ ② TerraMind 四层 PCA ｜ "
                 "③ AnySat 两尺度 PCA ｜ ④ 假彩色 B12/B8A/B04（青线=GT）", fontsize=12)
    out = ROOT / "outputs" / "visualizations" / "four_column_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    js = ROOT / "outputs" / "logs" / "four_column_compare.json"
    js.write_text(json.dumps([{
        "sample_id": r["rec"]["sample_id"], "event": r["rec"]["event_id"],
        "class_text": CLASS_TEXT,
        "clip_region": [{"text": t, "clip": round(float(v), 4)} for t, v in zip(texts, r["sims"])],
    } for r in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    print("[saved]", out)
    print("[saved]", js)
    for r in rows:
        print(f'{r["rec"]["sample_id"]:24s} class_clip={r["sims"][0]:+.3f}  '
              f'top_attr={names[int(np.argmax(r["sims"][1:]))]}')


if __name__ == "__main__":
    main()

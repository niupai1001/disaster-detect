"""文本 ↔ 区域对照：把每条语义文本与它对应的影像区域摆在一起。

每行一个样本：
  ① 整幅 RGB，框出候选区域（GT 连通域）与模型预测区域
  ② 该区域的裁剪图（放大）
  ③ 文本清单：类别提示 + 10 条属性提示，各自给出两个分数
     - `CLIP`：用 RemoteCLIP **图像塔**对该裁剪做零样本相似度（独立参考，可肉眼核对）
     - `head`：当前模型的属性头输出（未训练，预期恒为 ~0.5）

为什么要两个分数：属性头现在是 bias 主导的常数，看不出对应关系；
用 CLIP 对同一块裁剪打分，才能判断"这段文本是否真的描述这块地"。

用法: python scripts/text_region_pairs.py --n 10
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
from landslide_seg.data.certified import filter_records, load_sample, read_manifest  # noqa: E402
from landslide_seg.models.build import DEFAULT_WEIGHTS, build_from_config  # noqa: E402
from landslide_seg.models.semantic_dynamic import ATTRIBUTE_VOCAB  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CLASS_TEXT = "a landslide scar in a satellite image"


def rgb_u8(x12: np.ndarray) -> np.ndarray:
    rgb = np.stack([x12[3], x12[2], x12[1]], -1).astype(np.float32)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def largest_region(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """返回最大连通域的 bbox（不引入 scipy，用简单的连通标记）。"""
    from collections import deque

    if mask.sum() == 0:
        return None
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    best = None
    best_area = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                q = deque([(y, x)])
                seen[y, x] = True
                ys, xs = [], []
                while q:
                    cy, cx = q.popleft()
                    ys.append(cy); xs.append(cx)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                if len(ys) > best_area:
                    best_area = len(ys)
                    best = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    set_seed(1234)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import clip

    clip_model, preprocess = clip.load(str(DEFAULT_WEIGHTS["remoteclip"]), device=device, jit=False)
    clip_model.eval()

    names = list(ATTRIBUTE_VOCAB)
    texts = [CLASS_TEXT] + [ATTRIBUTE_VOCAB[k] for k in names]
    model = build_from_config(cfg, device=device)
    model.eval()

    recs = filter_records(read_manifest(ROOT / "manifests" / "certified.jsonl"), split=args.split)[: args.n]
    records = []
    rows_img = []

    for rec in recs:
        s = load_sample(rec, boa_offset=float(cfg["data"].get("boa_offset_dn", 0.0)))
        rgb = rgb_u8(s["x12"])
        mask = s["mask"] > 0
        bbox = largest_region(mask)
        if bbox is None:
            continue
        bx0, by0, bx1, by1 = bbox
        pad = 2
        cx0, cy0 = max(bx0 - pad, 0), max(by0 - pad, 0)
        cx1, cy1 = min(bx1 + pad, 128), min(by1 + pad, 128)
        crop = rgb[cy0:cy1, cx0:cx1]

        # CLIP 零样本：区域裁剪 vs 每条文本
        pil = __import__("PIL.Image", fromlist=["Image"]).fromarray(crop)
        with torch.no_grad():
            img_t = preprocess(pil).unsqueeze(0).to(device)
            img_f = clip_model.encode_image(img_t).float()
            txt_f = clip_model.encode_text(clip.tokenize(texts).to(device)).float()
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            sims = (img_f @ txt_f.T)[0].cpu().numpy()

        # 模型当前的属性头输出（同一区域的池化）
        x12 = torch.from_numpy(s["x12"])[None].to(device)
        x10 = torch.from_numpy(s["x10"])[None].to(device)
        dates = torch.tensor([[float(s["doy"])]], device=device)
        valid = torch.from_numpy(s["valid"])[None].to(device)
        with torch.no_grad():
            out = model(x12, x10, dates, return_features=True)
            sem = model.semantic(
                out.features["pyramid"][0], torch.sigmoid(out.coarse_logits),
                valid.float().unsqueeze(1), spectral=x10,
            )
            head = torch.sigmoid(sem.attr_logits)[0].cpu().numpy()

        rec_row = {
            "sample_id": rec["sample_id"], "event": rec["event_id"],
            "bbox_xyxy": [cx0, cy0, cx1, cy1], "mask_px": int(mask.sum()),
            "class_text": CLASS_TEXT, "class_clip": round(float(sims[0]), 4),
            "attributes": [
                {"key": k, "text": ATTRIBUTE_VOCAB[k],
                 "clip": round(float(sims[i + 1]), 4), "head": round(float(head[i]), 4)}
                for i, k in enumerate(names)
            ],
        }
        records.append(rec_row)
        rows_img.append((rec_row, rgb, (bx0, by0, bx1, by1), crop))

    n = len(rows_img)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 2.6 * n),
                             gridspec_kw={"width_ratios": [1.0, 0.8, 1.9]})
    for i, (row, rgb, bbox, crop) in enumerate(rows_img):
        axes[i, 0].imshow(rgb)
        axes[i, 0].add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                       fill=False, edgecolor="#E24B4A", lw=1.2))
        axes[i, 0].set_title(f"{row['sample_id']}", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(crop)
        axes[i, 1].set_title(f"region crop {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}", fontsize=8)
        axes[i, 1].axis("off")

        ax = axes[i, 2]
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        y = 1.0
        ax.text(0.01, y, f'CLASS  "{row["class_text"]}"  CLIP={row["class_clip"]:.3f}',
                fontsize=7.5, va="top", weight="bold")
        y -= 0.10
        for a in row["attributes"]:
            mark = "●" if a["clip"] > 0.25 else "○"
            ax.text(0.02, y, f'{mark} {a["key"]:16s} "{a["text"]}"',
                    fontsize=6.8, va="top")
            ax.text(0.74, y, f'CLIP={a["clip"]:+.3f}  head={a["head"]:.3f}',
                    fontsize=6.8, va="top", family="monospace")
            y -= 0.088

    fig.suptitle("文本 ↔ 区域对照：CLIP=对区域裁剪的零样本相似度（可核对）；"
                 "head=当前模型属性头（未训练，预期恒定）", fontsize=11)
    fig.tight_layout()
    out = ROOT / "outputs" / "visualizations" / "text_region_pairs.png"
    fig.savefig(out, dpi=115)
    plt.close(fig)

    js = ROOT / "outputs" / "logs" / "text_region_pairs.json"
    js.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out}")
    print(f"[saved] {js}")
    for r in records:
        top = sorted(r["attributes"], key=lambda a: -a["clip"])[:3]
        print(f"{r['sample_id']:22s} class={r['class_clip']:+.3f} | "
              + ", ".join(f'{a["key"]}={a["clip"]:+.3f}' for a in top))


if __name__ == "__main__":
    main()

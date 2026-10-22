"""生成"数据集与模块输出"三组图。

图 1  rpt_dataset_gallery : 8 个样本 × （真彩色 / 假彩色 / GT / SCL）
图 2  rpt_dataset_stats   : 掩膜尺寸、云量、光谱曲线、时相时间轴
图 3  rpt_module_outputs  : 一个样本走完全流程的 10 个中间输出

用途：让人直观看懂"数据是什么样、每个模块吐出了什么、为什么是这样"。
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
from landslide_seg.data.sen12 import load_sen12  # noqa: E402
from landslide_seg.models.build import build_from_config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

SCL_NAMES = ["无数据", "饱和", "地形阴影", "云影", "植被", "裸地",
             "水体", "未分类", "中概率云", "高概率云", "薄云", "雪"]
SCL_COLORS = ["#000000", "#8c564b", "#7f7f7f", "#17becf", "#2ca02c", "#c5b0d5",
              "#1f77b4", "#ff7f0e", "#f7b6d2", "#e377c2", "#bcbd22", "#17becf"]


def rgb(x12, idx, bounds):
    """按给定的每通道 (lo, hi) 线性拉伸；bounds 由整组样本统一计算，跨样本可比。"""
    chans = []
    for b, (lo, hi) in zip(idx, bounds):
        c = x12[b].astype(np.float32)
        chans.append(np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1))
    return np.stack(chans, -1)


def group_bounds(loaded, chans, lo=2, hi=98):
    out = []
    for b in chans:
        vals = np.concatenate([sm["x12"][b].ravel()[::4] for sm in loaded])
        out.append((float(np.percentile(vals, lo)), float(np.percentile(vals, hi))))
    return out


def pca1(feat: torch.Tensor) -> np.ndarray:
    c, h, w = feat.shape
    x = feat.reshape(c, -1).T
    sub = x[::4]
    sub = sub - sub.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(sub, q=3, niter=2)
    proj = (x - sub.mean(0, keepdim=True)) @ v[:, :1]
    img = proj.reshape(h, w)
    lo, hi = torch.quantile(img.flatten(), torch.tensor([0.02, 0.98], device=img.device))
    return torch.clamp((img - lo) / (hi - lo + 1e-9), 0, 1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    set_seed(1234)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    off = float(cfg["data"].get("boa_offset_dn", 0.0))

    recs = read_manifest(ROOT / "manifests" / "certified.jsonl")
    # ---------- 图 1：样本画廊（按事件轮转取样） ----------
    by_ev = {}
    for r in recs:
        by_ev.setdefault(r["event_id"], []).append(r)
    picked = []
    while len(picked) < 8:
        for ev in sorted(by_ev):
            if by_ev[ev] and len(picked) < 8:
                picked.append(by_ev[ev].pop(0))

    loaded = [(rec, load_sample(rec, boa_offset=off), load_sen12(rec["source_uri"]))
              for rec in picked]
    bnd_tc = group_bounds([x[1] for x in loaded], (3, 2, 1))
    bnd_fc = group_bounds([x[1] for x in loaded], (11, 8, 3))

    fig, axes = plt.subplots(8, 4, figsize=(11.5, 20))
    for i, (rec, s, nc) in enumerate(loaded):
        idx = rec["time_index"]
        axes[i, 0].imshow(rgb(s["x12"], (3, 2, 1), bnd_tc))
        axes[i, 0].set_title(f'{rec["sample_id"]}  {rec["event_id"]}\n真彩色 B04/B03/B02', fontsize=8)
        axes[i, 1].imshow(rgb(s["x12"], (11, 8, 3), bnd_fc))
        axes[i, 1].set_title("假彩色 B12/B8A/B04", fontsize=8)
        axes[i, 2].imshow(s["mask"], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(f'GT 掩膜  {int(s["mask"].sum())} px ({s["mask"].mean()*100:.2f}%)', fontsize=8)
        scl = nc.scl[idx].astype(int)
        axes[i, 3].imshow(scl, cmap=ListedColormap(SCL_COLORS), vmin=0, vmax=11)
        axes[i, 3].set_title(f'SCL  {nc.date_at(idx)}  云{(scl==9).mean()*100:.0f}%', fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle("数据集样本长什么样（8 个样本，跨 4 个事件）", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "visualizations" / "rpt_dataset_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 2：数据集统计 ----------
    px, cloud = [], []
    spec_pos, spec_bg = [], []
    for rec in recs:
        s = load_sample(rec, boa_offset=off)
        nc = load_sen12(rec["source_uri"])
        idx = rec["time_index"]
        px.append(float((s["mask"] > 0).mean()))
        cloud.append(float(np.isin(nc.scl[idx], (2, 3, 8, 9, 10)).mean()))
        m = s["mask"] > 0
        if m.sum() > 0:
            spec_pos.append([float(s["x12"][b][m].mean()) for b in range(12)])
            spec_bg.append([float(s["x12"][b][~m].mean()) for b in range(12)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.hist(np.array(px) * 100, bins=20, color="#378ADD")
    ax.set_title("每个 128×128 图块里滑坡像素占比（%）", fontsize=10)
    ax.set_xlabel("占比 %"); ax.set_ylabel("样本数")

    ax = axes[0, 1]
    ax.hist(np.array(cloud) * 100, bins=20, color="#1D9E75")
    ax.set_title("所选时相的云+阴影占比（%）", fontsize=10)
    ax.set_xlabel("占比 %")

    ax = axes[1, 0]
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
             "B08", "B8A", "B09", "B11", "B12"]
    ax.plot(range(12), np.mean(spec_pos, 0), "-o", ms=4, label="滑坡区（GT=1）", color="#E24B4A")
    ax.plot(range(12), np.mean(spec_bg, 0), "-s", ms=4, label="背景（GT=0）", color="#185FA5")
    ax.set_xticks(range(12)); ax.set_xticklabels(bands, fontsize=8)
    ax.set_ylabel("DN"); ax.legend(fontsize=8)
    ax.set_title("12 波段平均光谱：滑坡区 vs 背景", fontsize=10)

    ax = axes[1, 1]
    s = load_sen12(recs[0]["source_uri"])
    dates = [s.date_at(t).isoformat()[5:] for t in range(s.num_steps)]
    cl = [float(np.isin(s.scl[t], (2, 3, 8, 9, 10)).mean() * 100) for t in range(s.num_steps)]
    ax.bar(range(s.num_steps), cl, color="#BA7517")
    ax.set_xticks(range(s.num_steps)); ax.set_xticklabels(dates, rotation=90, fontsize=7)
    ax.set_ylabel("云+阴影 %")
    ax.set_title(f"一个样本的 15 个时相（{recs[0]['sample_id']}）", fontsize=10)
    fig.suptitle("数据集统计特征", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "visualizations" / "rpt_dataset_stats.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 3：一个样本走完全流程 ----------
    model = build_from_config(cfg, device=device)
    model.eval()
    rec = picked[0]
    s = load_sample(rec, boa_offset=off)
    x12 = torch.from_numpy(s["x12"])[None].to(device)
    x10 = torch.from_numpy(s["x10"])[None].to(device)
    dates_t = torch.tensor([[float(s["doy"])]], device=device)
    valid = torch.from_numpy(s["valid"])[None].to(device)

    with torch.no_grad():
        tm = model.terramind(x12)
        f40 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates_t, patch_size_m=40)
        f160 = model.anysat(normalize_anysat(x10).unsqueeze(1), dates_t, patch_size_m=160)
        out = model(x12, x10, dates_t, region_mask=torch.from_numpy(s["mask"])[None].float().to(device),
                    valid=valid, return_features=True)
        sem = model.semantic(out.features["pyramid"][0], torch.sigmoid(out.coarse_logits),
                             valid.float().unsqueeze(1), spectral=x10)

    def tm_layer(li):
        tok = tm["tokens"][li][0].transpose(0, 1).reshape(-1, *tm["grid"])
        up = torch.nn.functional.interpolate(tok[None], size=(128, 128), mode="bilinear")[0]
        return pca1(up)

    panels = [
        (rgb(s["x12"], (11, 8, 3), group_bounds([s], (11, 8, 3))), "① 输入：假彩色 B12/B8A/B04"),
        (tm_layer(-1), "② TerraMind 末层 14×14→128（91 m）"),
        (pca1(f40["split"]["context"][0]), "③ AnySat 40m 上下文（块内恒定）"),
        (pca1(f40["split"]["local"][0]), "④ AnySat 40m 局部（10 m）"),
        (pca1(f160["split"]["context"][0]), "⑤ AnySat 160m 上下文"),
        (pca1(f160["split"]["local"][0]), "⑥ AnySat 160m 局部"),
        (pca1(out.features["pyramid"][0][0]), "⑦ 融合 P0（10 m）"),
        (pca1(out.features["pyramid"][3][0]), "⑧ 融合 P3（80 m）"),
        (sem.sem_map[0, 0].cpu().numpy(),
         f"⑨ 语义相似度图 [{float(sem.sem_map.min()):+.2f},{float(sem.sem_map.max()):+.2f}]"),
        (torch.sigmoid(out.coarse_logits)[0, 0].cpu().numpy(),
         f"⑩ 粗概率 [{float(torch.sigmoid(out.coarse_logits).min()):.3f},"
         f"{float(torch.sigmoid(out.coarse_logits).max()):.3f}]"),
        (torch.sigmoid(out.fine_logits)[0, 0].cpu().numpy(),
         f"⑪ 细概率 [{float(torch.sigmoid(out.fine_logits).min()):.3f},"
         f"{float(torch.sigmoid(out.fine_logits).max()):.3f}]"),
        (s["mask"], "⑫ GT 掩膜"),
    ]
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.6))
    for ax, (img, title) in zip(axes.ravel(), panels):
        if "语义" in title or "概率" in title:
            ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    fig.suptitle(f"一个样本的完整流程输出（{rec['sample_id']}，未训练权重）", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "visualizations" / "rpt_module_outputs.png", dpi=115, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_samples": len(recs),
        "mask_frac_mean_pct": round(float(np.mean(px) * 100), 3),
        "mask_frac_max_pct": round(float(np.max(px) * 100), 3),
        "cloud_mean_pct": round(float(np.mean(cloud) * 100), 2),
        "cloud_p90_pct": round(float(np.percentile(cloud, 90) * 100), 2),
        "spectrum_pos": [round(v, 1) for v in np.mean(spec_pos, 0)],
        "spectrum_bg": [round(v, 1) for v in np.mean(spec_bg, 0)],
    }
    (ROOT / "outputs" / "logs" / "rpt_dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print("[saved] rpt_dataset_gallery.png / rpt_dataset_stats.png / rpt_module_outputs.png")


if __name__ == "__main__":
    main()

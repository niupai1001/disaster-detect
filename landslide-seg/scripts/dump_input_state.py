"""打印"进入模型那一刻"的原始数据状态。

两个模态在入口处完全不对等：
- 文本：11 条**写死的英文字符串** → CLIP 分词（BPE，含 start/end、补齐到 77）
        → 文本塔 → [N,512] float32 单位向量，冻结成 buffer；
        之后再没有任何字符串参与计算。
- 影像：npz 里的原始 DN（int→float32，0–10000 量级）
        → 分别按两个骨干的官方统计做逐波段标准化
        → TerraMind 还会把 128×128 双线性放大到 224×224。

本脚本把这些中间量的真实数值 dump 出来（JSON + 图），便于逐项核对。
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
from landslide_seg.data.band_mapping import (  # noqa: E402
    ANY_SAT_S2_BANDS,
    ANY_SAT_S2_MEAN,
    ANY_SAT_S2_STD,
    S2_L2A_BANDS,
    normalize_anysat,
)
from landslide_seg.data.certified import filter_records, load_sample, read_manifest  # noqa: E402
from landslide_seg.encoders.terramind_tim import TERRAMIND_S2L2A_MEAN, TERRAMIND_S2L2A_STD, normalize_s2l2a_dn  # noqa: E402
from landslide_seg.models.build import DEFAULT_WEIGHTS  # noqa: E402
from landslide_seg.models.semantic_dynamic import ATTRIBUTE_VOCAB  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

CLASS_TEXT = "a landslide scar in a satellite image"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    set_seed(1234)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- 文本侧 ----------------
    import clip

    clip_model, _ = clip.load(str(DEFAULT_WEIGHTS["remoteclip"]), device=device, jit=False)
    clip_model.eval()
    names = list(ATTRIBUTE_VOCAB)
    texts = [CLASS_TEXT] + [ATTRIBUTE_VOCAB[k] for k in names]

    tok = clip.tokenize(texts).to(device)          # [N,77] int64
    eot = tok.argmax(dim=1)                        # 每个序列的 <end> 位置
    with torch.no_grad():
        emb = clip_model.encode_text(tok).float()
    emb_n = emb / emb.norm(dim=-1, keepdim=True)

    text_state = []
    for i, t in enumerate(texts):
        ids = tok[i].cpu().numpy()
        n = int(eot[i]) + 1
        text_state.append({
            "index": i,
            "role": "class" if i == 0 else f"attribute:{names[i-1]}",
            "text": t,
            "n_tokens_incl_sot_eot": n,
            "token_ids_first12": ids[:12].tolist(),
            "token_ids_active": ids[:n].tolist(),
            "padding_len": int(ids.shape[0]),
            "embedding_dtype": str(emb.dtype).replace("torch.", ""),
            "embedding_shape": list(emb[i].shape),
            "embedding_norm": round(float(emb[i].norm()), 4),
            "embedding_first8": [round(float(v), 5) for v in emb[i][:8]],
            "cos_to_class": round(float(emb_n[i] @ emb_n[0]), 4),
        })

    # ---------------- 影像侧 ----------------
    recs = filter_records(read_manifest(ROOT / "manifests" / "certified.jsonl"), split="test")
    rec = recs[args.sample]
    smp = load_sample(rec, boa_offset=float(cfg["data"].get("boa_offset_dn", 0.0)))
    x12, x10 = smp["x12"], smp["x10"]

    band_stats = []
    for i, b in enumerate(S2_L2A_BANDS):
        band_stats.append({
            "band": b, "dtype_in_npz": "float32",
            "min": float(x12[i].min()), "max": float(x12[i].max()),
            "mean": round(float(x12[i].mean()), 1),
            "tm_mean_ref": float(TERRAMIND_S2L2A_MEAN[i]), "tm_std_ref": float(TERRAMIND_S2L2A_STD[i]),
        })

    xn = normalize_s2l2a_dn(torch.from_numpy(x12)[None])[0]
    xa = normalize_anysat(torch.from_numpy(x10)[None])[0]

    patch_y, patch_x = 64, 64
    r = 2
    patch_raw = x12[:, patch_y - r:patch_y + r, patch_x - r:patch_x + r]
    patch_norm = xn[:, patch_y - r:patch_y + r, patch_x - r:patch_x + r]

    image_state = {
        "sample_id": rec["sample_id"],
        "source": rec["source_uri"],
        "cache": rec["cache"],
        "boa_offset_dn_applied": float(cfg["data"].get("boa_offset_dn", 0.0)),
        "x12": {"shape": list(x12.shape), "dtype": str(x12.dtype),
                "overall_min": float(x12.min()), "overall_max": float(x12.max()),
                "bands": band_stats},
        "x10": {"bands": list(ANY_SAT_S2_BANDS),
                "shape": list(x10.shape), "dtype": str(x10.dtype),
                "anysat_mean_ref": [float(v) for v in ANY_SAT_S2_MEAN],
                "anysat_std_ref": [float(v) for v in ANY_SAT_S2_STD]},
        "after_terramind_norm": {
            "shape": list(xn.shape), "dtype": "float32",
            "per_band_mean": [round(float(v), 3) for v in xn.mean(dim=(1, 2))],
            "per_band_std": [round(float(v), 3) for v in xn.std(dim=(1, 2))],
            "note": "再被双线性放大到 224x224 才进 TerraMind",
        },
        "after_anysat_norm": {
            "shape": list(xa.shape), "dtype": "float32",
            "per_band_mean": [round(float(v), 3) for v in xa.mean(dim=(1, 2))],
            "per_band_std": [round(float(v), 3) for v in xa.std(dim=(1, 2))],
            "note": "再 unsqueeze 成 [1,1,10,128,128] 才进 AnySat",
        },
        "patch_4x4_at_64_64": {
            "raw_dn_band_B04": np.round(patch_raw[3], 1).tolist(),
            "raw_dn_band_B08": np.round(patch_raw[7], 1).tolist(),
            "normalized_band_B04": np.round(patch_norm[3].numpy(), 3).tolist(),
            "normalized_band_B08": np.round(patch_norm[7].numpy(), 3).tolist(),
        },
    }

    # ---------------- 图 ----------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.32, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_title("① 文本进入时的原始状态：字符串 → token id → 512 维向量（此后文本不再参与计算）", fontsize=11, loc="left")
    lines = []
    for t in text_state:
        ids = ",".join(str(v) for v in t["token_ids_active"][:8])
        lines.append(f'{t["index"]:>2}  {t["role"]:<28} "{t["text"]}"\n'
                     f'      len={t["n_tokens_incl_sot_eot"]:>2}  ids=[{ids}...]  '
                     f'emb[:4]=[{", ".join(f"{v:+.3f}" for v in t["embedding_first8"][:4])}]  '
                     f'cos(cls)={t["cos_to_class"]:+.3f}')
    ax.text(0.005, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=7.4)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(emb_n.cpu().numpy()[:40], aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    ax.set_title("前 40 维的文本向量（已归一化）", fontsize=9)
    ax.set_yticks(range(len(texts)))
    ax.set_yticklabels([t["role"].split(":")[-1] for t in text_state], fontsize=6)

    ax = fig.add_subplot(gs[1, 1])
    show = [0, 3, 7, 9]                                  # B01 B04 B08 B09
    ax.imshow(np.concatenate([x12[i] for i in show], axis=1), cmap="gray",
              vmin=0, vmax=float(np.percentile(x12, 99)))
    ax.set_title("原始 DN（B01|B04|B08|B09 拼接，0–p99）", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.concatenate([xn[i].numpy() for i in show], axis=1), cmap="gray")
    ax.set_title("TerraMind 标准化后（同四波段）", fontsize=9)
    ax.axis("off")

    fig.suptitle(f"输入原始状态 — 文本固定 11 条；影像 {rec['sample_id']}（{rec['event_id']}）", fontsize=12)
    out = ROOT / "outputs" / "visualizations" / "input_raw_state.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)

    js = ROOT / "outputs" / "logs" / "input_raw_state.json"
    js.write_text(json.dumps({"text": text_state, "image": image_state}, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    print(json.dumps({"text": text_state[:2], "image_bands": band_stats[:3]}, indent=1, ensure_ascii=False))
    print(f"[saved] {out}")
    print(f"[saved] {js}")


if __name__ == "__main__":
    main()

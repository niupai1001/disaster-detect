"""官方 AnySat / DOFA 原样前向：把真实 Sentinel-2 十波段图块喂进去，记录实际输出。

设计约束（与项目内 use_next_v2 的读出不同）：
- 只从 **官方仓库代码 + 官方发布权重** 构造模型，不经过 ``landslide_seg`` 的封装、
  不加 1x1 投影、不加分类头；因此输出就是官方 forward 的返回值本身。
- 只读：不训练、不改权重、不写预测；用于回答"官方系统对一块 S2 到底输出什么"。
- 输入预处理沿用项目已核过的官方契约（AnySat 官方通道统计、DOFA 波长条件），
  预处理函数与 anchors.py 保持同一实现，避免出现第二套口径。

用法::

    python -X utf8 scripts/native_official_probe.py --tiles hiroshima_s2_1000
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.data.band_mapping import ANY_SAT_S2_BANDS, normalize_anysat  # noqa: E402

CACHE_DIR = ROOT / "data" / "s2_10band_v4_p20"
MANIFEST = ROOT / "manifests" / "v4r1" / "pool_100.jsonl"
DOFA_CODE = ROOT / "vendor" / "dofa-official-0cfb7e1"
DOFA_CKPT = ROOT / "weights" / "dofa" / "XShadow_DOFA_base" / "DOFA_ViT_base_e100.pth"
ANYSAT_CODE = ROOT / "vendor" / "AnySat-main"
ANYSAT_CKPT = ROOT / "weights" / "AnySat_base.pth"

#: DOFA 官方十波段名义中心波长（微米），与 configs/segformer/v4r3 的 dofa 配置一致
DOFA_WAVELENGTHS_S2 = (0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.832, 0.864, 1.61, 2.2)
#: 同一权重吃 RGB 时只用三个可见光波长（DOFA 论文的波长条件能力）
DOFA_WAVELENGTHS_RGB = (0.665, 0.560, 0.490)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# 官方模型
# ---------------------------------------------------------------------------
def build_dofa(device: torch.device):
    """官方 downstream_tasks/geobench_segmentation 的 dense 编码器，224 原生输入。"""
    if str(DOFA_CODE) not in sys.path:
        sys.path.insert(0, str(DOFA_CODE))
    official = load_module(
        "native_probe_dofa_dense",
        DOFA_CODE / "downstream_tasks" / "geobench_segmentation" / "model.py",
    )
    model = official.vit_base_patch16(img_size=224, drop_path_rate=0.0)
    payload = torch.load(DOFA_CKPT, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    loaded = model.load_state_dict(state, strict=False)
    allowed = {"mask_token", "norm.weight", "norm.bias", "projector.weight", "projector.bias"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError(f"DOFA 权重与官方 dense 编码器不匹配: {loaded}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    info = {
        "backend": "vendor/dofa-official-0cfb7e1/downstream_tasks/geobench_segmentation/model.py:vit_base_patch16",
        "checkpoint": str(DOFA_CKPT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(DOFA_CKPT),
        "checkpoint_keys": len(state),
        "checkpoint_has_decoder": any("decoder" in k for k in state),
        "params_total": sum(p.numel() for p in model.parameters()),
        "out_indices": model.out_indices,
        "patch_size": model.patch_embed.kernel_size,
        "dense_stride_px": model.patch_embed.kernel_size,
        "img_size": model.img_size,
        "unexpected_keys": list(loaded.unexpected_keys),
        "missing_keys": list(loaded.missing_keys),
    }
    return model, info


def build_anysat(device: torch.device):
    """官方 hubconf.AnySat(base)，官方权重严格加载。"""
    if str(ANYSAT_CODE / "src") not in sys.path:
        sys.path.append(str(ANYSAT_CODE / "src"))
    if str(ANYSAT_CODE) not in sys.path:
        sys.path.insert(0, str(ANYSAT_CODE))
    hub = load_module("native_probe_anysat_hubconf", ANYSAT_CODE / "hubconf.py")
    model = hub.AnySat(model_size="base", flash_attn=False)
    payload = torch.load(ANYSAT_CKPT, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    loaded = model.model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    info = {
        "backend": "vendor/AnySat-main/hubconf.py:AnySat(model_size='base', flash_attn=False)",
        "checkpoint": str(ANYSAT_CKPT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(ANYSAT_CKPT),
        "params_total": sum(p.numel() for p in model.parameters()),
        "embed_dim": 768,
        "depth": len(model.model.blocks),
        "modalities": list(model.config.get("modalities", {}).get("all", [])),
        "loaded_missing": len(loaded.missing_keys),
        "loaded_unexpected": len(loaded.unexpected_keys),
    }
    return model, info


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------
def read_meta(sample_ids: list[str]) -> dict:
    meta = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["sample_id"] in sample_ids:
                meta[rec["sample_id"]] = rec
    missing = set(sample_ids) - set(meta)
    if missing:
        raise KeyError(f"清单里找不到: {sorted(missing)}")
    return meta


def load_tile(sample_id: str) -> dict:
    path = CACHE_DIR / f"{sample_id}.npz"
    data = np.load(path)
    x = data["x10"].astype(np.float32)          # (10,128,128) 反射率 x10000
    valid = data["valid"].astype(bool)
    label = data["mask"].astype(np.uint8)
    return {"image_dn": x, "valid": valid, "label": label, "cache": path}


def dofa_input(tile: dict, bands_idx: tuple[int, ...], size: int) -> np.ndarray:
    """官方 dense 前向要求固定 224 网格：同范围双线性重采样，无效像元置 0。

    DOFA 期望 **0–1 物理反射率**（与项目 readout 一致）；缓存里是 DN＝反射率×10000，
    因此这里除以 10000。喂 DN 不会报错，但会把四级特征标准差从 0.38/0.55/0.96/5.50
    推到约 1.4e3，注意力分布随之改变（实测见报告"输入量纲"一节）。
    """
    x = np.where(tile["valid"][None], tile["image_dn"], 0.0)[list(bands_idx)] / 10000.0
    t = torch.from_numpy(x.astype(np.float32))[None]
    if size != x.shape[-1]:
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


def anysat_input(tile: dict, day: int) -> dict:
    bands = list(ANY_SAT_S2_BANDS)
    x = normalize_anysat(torch.from_numpy(tile["image_dn"]) * 10000.0)
    x = torch.where(torch.from_numpy(tile["valid"])[None], x, torch.zeros_like(x))   # 无效像元在中性处（与 anchors.py 一致）
    return {"s2": x[None, None], "s2_dates": torch.tensor([[day]], dtype=torch.long), "bands": bands}


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
def feature_stats(f: torch.Tensor) -> dict:
    f = f.float()
    return {
        "shape": list(f.shape),
        "dtype": str(f.dtype).replace("torch.", ""),
        "min": float(f.min()),
        "max": float(f.max()),
        "mean": float(f.mean()),
        "std": float(f.std()),
        "finite": bool(torch.isfinite(f).all()),
        "l2_norm_mean": float(f.flatten(0, -2).norm(dim=0).mean()) if f.ndim >= 3 else float(f.norm()),
    }


def effective_rank(feats: np.ndarray) -> dict:
    """参与率（participation ratio）：协方差特征值分布的有效维数。"""
    x = feats - feats.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    lam = (s ** 2) / max(len(x) - 1, 1)
    lam = np.clip(lam, 1e-12, None)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    cum = np.cumsum(lam) / lam.sum()
    return {
        "participation_ratio": pr,
        "dims_for_90pct_var": int(np.searchsorted(cum, 0.90) + 1),
        "top1_var_share": float(lam[0] / lam.sum()),
        "input_dim": int(feats.shape[1]),
    }


def pca_rgb(feats: np.ndarray) -> np.ndarray:
    """把 (N,D) 特征压成三主成分伪彩，返回 (N,3)，按 2-98 百分位拉伸。"""
    x = feats - feats.mean(0, keepdims=True)
    x = x / (x.std(0, keepdims=True) + 1e-6)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pcs = u[:, :3] * s[:3]
    for i in range(1, 3):
        if np.corrcoef(pcs[:, i], pcs[:, 0])[0, 1] < 0:
            pcs[:, i] *= -1
    rgb = np.zeros((pcs.shape[0], 3), np.float32)
    for i in range(3):
        lo, hi = np.percentile(pcs[:, i], [2, 98])
        rgb[:, i] = np.clip((pcs[:, i] - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


def anisotropy(tokens: np.ndarray, max_tokens: int = 2000, seed: int = 1234) -> dict:
    """token 场的各向异性：原始余弦、去均值后的余弦、公共分量占比。

    单个 ViT 的 token 常共享一个很大的公共向量，直接算余弦会得到接近 1 的假"相似"，
    因此同时给出"减去场平均后再算"的那一版。
    """
    t = tokens.reshape(-1, tokens.shape[-1]).astype(np.float64)
    if len(t) > max_tokens:
        idx = np.random.default_rng(seed).choice(len(t), max_tokens, replace=False)
        t = t[idx]
    nrm = np.linalg.norm(t, axis=1)
    mu = t.mean(0)
    tn = t / (nrm[:, None] + 1e-12)
    s_raw = (tn @ tn.T)[np.triu_indices(len(t), 1)]
    c = t - mu
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    s_cen = (cn @ cn.T)[np.triu_indices(len(c), 1)]
    return {
        "n_tokens_sampled": int(len(t)),
        "norm_mean": round(float(nrm.mean()), 4),
        "mean_pair_cos_raw": round(float(s_raw.mean()), 5),
        "mean_pair_cos_centred": round(float(s_cen.mean()), 5),
        "cos_to_mean": round(float((tn @ (mu / (np.linalg.norm(mu) + 1e-12))).mean()), 5),
        "spatial_std_over_norm": round(float(t.std(0).mean() / (nrm.mean() + 1e-12)), 6),
    }


def lag1_autocorr(grid: np.ndarray) -> dict:
    """第一主成分图的一阶空间自相关（右邻 + 下邻），衡量输出里还剩多少空间结构。"""
    h, w, d = grid.shape
    x = grid.reshape(-1, d).astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    pc = (u[:, 0] * s[0]).reshape(h, w)
    pc = (pc - pc.mean()) / (pc.std() + 1e-12)
    hz = float(np.corrcoef(pc[:, :-1].ravel(), pc[:, 1:].ravel())[0, 1])
    vt = float(np.corrcoef(pc[:-1, :].ravel(), pc[1:, :].ravel())[0, 1])
    return {"horizontal": round(hz, 4), "vertical": round(vt, 4), "mean": round((hz + vt) / 2, 4),
            "grid": [h, w], "pc1_var_share": round(float((s[0] ** 2) / (s ** 2).sum()), 4)}


def offset_profile(grid: np.ndarray, offsets=(1, 2, 3, 4, 8)) -> dict:
    """第一主成分沿 x / y 不同偏移的相关，用来判断块内子 patch 的排列是否与像元顺序一致。"""
    h, w, d = grid.shape
    x = grid.reshape(-1, d).astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pc = (x @ vt[0]).reshape(h, w)
    pc = (pc - pc.mean()) / (pc.std() + 1e-12)

    def corr(a, b):
        return round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 4)

    return {"dx": {str(k): corr(pc[:, :-k], pc[:, k:]) for k in offsets},
            "dy": {str(k): corr(pc[:-k, :], pc[k:, :]) for k in offsets}}


def ridge_r2(feats: np.ndarray, target: np.ndarray, seed: int = 1234, alpha: float = 1.0) -> dict:
    """留出 20% 像元的岭回归：特征里线性可还原多少原始反射率（R²，逐波段）。"""
    n = len(feats)
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * 0.8)
    tr, te = order[:cut], order[cut:]
    xm, xs = feats[tr].mean(0), feats[tr].std(0) + 1e-6
    X = (feats - xm) / xs
    ym, ys = target[tr].mean(0), target[tr].std(0) + 1e-6
    Y = (target - ym) / ys
    d = X.shape[1]
    A = X[tr].T @ X[tr] + alpha * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, X[tr].T @ Y[tr])
    pred = X[te] @ W
    sse = ((Y[te] - pred) ** 2).sum(0)
    sst = ((Y[te] - Y[te].mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1 - sse / sst
    return {"r2_per_band": [round(float(v), 4) for v in r2], "r2_mean": round(float(np.mean(r2)), 4)}


def reconstruct_rgb(feats: np.ndarray, target: np.ndarray, hw: int) -> np.ndarray:
    """用同一岭回归把三可见光波段还原出来，返回 (N,3)，仅用于看图。"""
    xm, xs = feats.mean(0), feats.std(0) + 1e-6
    X = (feats - xm) / xs
    ym, ys = target.mean(0), target.std(0) + 1e-6
    Y = (target - ym) / ys
    W = np.linalg.solve(X.T @ X + np.eye(X.shape[1], dtype=np.float32), X.T @ Y)
    pred = (X @ W) * ys + ym
    return pred


def false_color(image_dn: np.ndarray, valid: np.ndarray, bands=(2, 1, 0)) -> np.ndarray:
    img = image_dn[list(bands)] / 10000.0
    out = np.zeros((128, 128, 3), np.float32)
    for i in range(3):
        b = img[i][valid]
        lo, hi = np.percentile(b, [2, 98])
        out[..., i] = np.clip((img[i] - lo) / max(hi - lo, 1e-6), 0, 1)
    out[~valid] = 0.35
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_tile(sample_id, meta, dofa, dofa_info, anysat, patch_sizes, device, do_fig):
    tile = load_tile(sample_id)
    valid, label = tile["valid"], tile["label"]
    date = datetime.fromisoformat(meta["date"])
    day = date.timetuple().tm_yday - 1
    result = {
        "sample_id": sample_id,
        "event_id": meta["event_id"],
        "date": meta["date"],
        "protocol_role": meta.get("protocol_role"),
        "label_px": int(label.sum()),
        "valid_ratio": round(float(valid.mean()), 4),
        "bbox": meta.get("bbox"),
        "day_of_year": day,
        "outputs": {},
    }

    # --- DOFA：十波段（官方波长） ---
    x10 = dofa_input(tile, tuple(range(10)), dofa_info["img_size"]).to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        levels = dofa(x10, list(DOFA_WAVELENGTHS_S2))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    result["outputs"]["dofa_s2"] = {
        "input": {"shape": list(x10.shape), "wavelengths_um": list(DOFA_WAVELENGTHS_S2),
                  "resample": f"{tile['image_dn'].shape[-1]}px/10m -> {dofa_info['img_size']}px"},
        "return_type": "list[Tensor]（4 个中间 block 的密集特征）",
        "latency_s": round(dt, 4),
        "levels": [feature_stats(f) for f in levels],
    }
    dofa_level4 = levels[-1][0].float().cpu().numpy()          # (768,14,14)
    dofa_cat = torch.cat([torch.nn.functional.interpolate(f, size=(128, 128), mode="bilinear", align_corners=False)
                          for f in levels], 1)[0].float().cpu().numpy()   # (3072,128,128)

    # --- DOFA：RGB 三波段（同一权重，不同波长） ---
    bands_rgb = (2, 1, 0)   # B04,B03,B02
    xrgb = dofa_input(tile, bands_rgb, dofa_info["img_size"]).to(device)
    with torch.no_grad():
        levels_rgb = dofa(xrgb, list(DOFA_WAVELENGTHS_RGB))
    result["outputs"]["dofa_rgb"] = {
        "input": {"shape": list(xrgb.shape), "wavelengths_um": list(DOFA_WAVELENGTHS_RGB),
                  "bands": ["B04", "B03", "B02"]},
        "return_type": "list[Tensor]",
        "levels": [feature_stats(f) for f in levels_rgb],
        "note": "同一发布权重，仅把波长列表换成三个可见光中心波长；卷积核由波长现场生成",
    }
    _a = dofa_level4.reshape(768, -1)
    _b = levels_rgb[-1][0].float().cpu().numpy().reshape(768, -1)
    _a = _a / (np.linalg.norm(_a, axis=0, keepdims=True) + 1e-12)
    _b = _b / (np.linalg.norm(_b, axis=0, keepdims=True) + 1e-12)
    result["outputs"]["dofa_rgb"]["token_cos_vs_s2_mean"] = round(float((_a * _b).sum(0).mean()), 4)

    # --- AnySat：多物理 patch 尺寸 ---
    feed = anysat_input(tile, day)
    anysat_out = {}
    ctx = loc = anysat_patch40 = anysat_tile_vec = None
    for p in patch_sizes:
        inp = {"s2": feed["s2"].to(device), "s2_dates": feed["s2_dates"].to(device)}
        timings, outs = {}, {}
        for mode in ("patch", "tile", "dense"):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                if mode == "dense":
                    outs[mode] = anysat(inp, patch_size=p, output="dense", output_modality="s2")
                else:
                    outs[mode] = anysat(inp, patch_size=p, output=mode)
            torch.cuda.synchronize()
            timings[mode] = round(time.perf_counter() - t0, 4)
        patch, tile_vec, dense = outs["patch"], outs["tile"], outs["dense"]
        anysat_out[f"patch_{p}m"] = {
            "patch_size_m": p,
            "input": {"s2_shape": list(feed["s2"].shape), "s2_dates": [[day]],
                      "normalization": "官方 ANY_SAT_S2_MEAN/STD，输入 DN=反射率x10000"},
            "outputs": {
                "patch": feature_stats(patch),
                "tile": feature_stats(tile_vec),
                "dense": feature_stats(dense),
            },
            "patch_grid": list(patch.shape[1:3]),
            "dense_layout": "最后一维 1536 = 前 768 上下文段（每 patch 恒定）+ 后 768 局部段（10m）",
            "latency_s": timings,
        }
        if p == 40:
            anysat_patch40 = patch[0].float().cpu().numpy()      # (32,32,768)
            ctx = dense[..., :768][0].float().cpu().numpy()      # (128,128,768)
            loc = dense[..., 768:][0].float().cpu().numpy()      # (128,128,768)
            # 上下文段在 40m patch 内是否恒定：取 4x4 块的通道标准差
            blk = ctx.reshape(32, 4, 32, 4, 768).transpose(0, 2, 1, 3, 4).reshape(32, 32, 16, 768)
            anysat_out["patch_40m"]["context_within_patch_std"] = {
                "max": round(float(blk.std(axis=2).max()), 8),
                "mean": round(float(blk.std(axis=2).mean()), 8),
            }
            local_blk = loc.reshape(32, 4, 32, 4, 768).transpose(0, 2, 1, 3, 4).reshape(32, 32, 16, 768)
            anysat_out["patch_40m"]["local_within_patch_std"] = {
                "mean": round(float(local_blk.std(axis=2).mean()), 4),
            }
            anysat_tile_vec = tile_vec[0].float().cpu().numpy()
    result["outputs"]["anysat"] = anysat_out
    if anysat_tile_vec is not None:
        result["outputs"]["anysat"]["tile_vector_40m_head8"] = [round(float(v), 5) for v in anysat_tile_vec[:8]]
        result["outputs"]["anysat"]["tile_vector_note"] = "完整 768 维不落盘，这里只留前 8 维便于核对同一性"

    # --- 特征诊断（只在有效像元上） ---
    vmask = valid
    target = (tile["image_dn"] / 10000.0).transpose(1, 2, 0)[vmask]        # (N,10)
    diag = {
        "note": "全部统计只在有效像元上计算，无效像元不参与",
        "n_valid_px": int(vmask.sum()),
        "dofa_level4_tokens": effective_rank(dofa_level4.reshape(768, -1).T),
        "dofa_concat_10m": effective_rank(dofa_cat.transpose(1, 2, 0)[vmask]),
    }
    if ctx is not None:
        diag.update({
            "anysat_context_10m": effective_rank(ctx[vmask]),
            "anysat_local_10m": effective_rank(loc[vmask]),
            "anysat_patch40_tokens": effective_rank(anysat_patch40.reshape(-1, 768)),
        })
    diag["linear_recoverability"] = {
        "dofa_concat_10m": ridge_r2(dofa_cat.transpose(1, 2, 0)[vmask], target),
    }
    if ctx is not None:
        diag["linear_recoverability"].update({
            "anysat_dense_10m": ridge_r2(np.concatenate([ctx, loc], -1)[vmask], target),
            "anysat_local_only": ridge_r2(loc[vmask], target),
        })

    # --- token 各向异性：公共分量有多大，空间差异占多少 ---
    ctx_cells = ctx.reshape(32, 4, 32, 4, 768)[:, 0, :, 0, :] if ctx is not None else None
    diag["token_anisotropy"] = {
        "dofa_level4": anisotropy(dofa_level4.reshape(768, -1).T),
    }
    if ctx is not None:
        diag["token_anisotropy"].update({
            "anysat_patch40": anisotropy(anysat_patch40),
            "anysat_context_cells": anisotropy(ctx_cells),
            "anysat_local_10m": anisotropy(loc),
        })
    diag["spatial_autocorrelation_lag1"] = {
        "input_B04": lag1_autocorr((tile["image_dn"][2] / 10000.0)[..., None]),
        "dofa_level4": lag1_autocorr(dofa_level4.transpose(1, 2, 0)),
    }
    if ctx is not None:
        diag["spatial_autocorrelation_lag1"].update({
            "anysat_context_cells": lag1_autocorr(ctx_cells),
            "anysat_local_10m": lag1_autocorr(loc),
        })
        # 块内顺序检查：局部段沿 x / y 的偏移相关剖面（dx=1 应远高于 dy=1，若布局是行优先）
        diag["local_offset_profile"] = offset_profile(loc)
    result["diagnostics"] = diag

    if do_fig and ctx is not None:
        make_figure(result, tile, dofa_level4, dofa_cat, ctx, loc, vmask, target)
        result["figure"] = str((OUT / "figures" / f"{sample_id}.png").relative_to(ROOT))
    return result


def make_figure(result, tile, dofa_level4, dofa_cat, ctx, loc, vmask, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    valid = tile["valid"]
    rgb = false_color(tile["image_dn"], valid)

    def label_map():
        m = np.zeros((128, 128, 3), np.float32)
        m[valid] = 0.32                       # 有效但非滑坡：中灰
        m[tile["label"].astype(bool) & valid] = (0.95, 0.75, 0.1)   # 滑坡：黄
        return m

    dofa_l4 = pca_rgb(dofa_level4.reshape(768, -1).T).reshape(14, 14, 3)
    dofa_l4 = np.kron(dofa_l4, np.ones((9, 9, 1), np.float32))[:128, :128]
    dofa_l4 = np.pad(dofa_l4, ((0, 128 - dofa_l4.shape[0]), (0, 128 - dofa_l4.shape[1]), (0, 0)),
                     mode="edge")

    def scatter(colors):
        m = np.zeros((128, 128, 3), np.float32)
        m[vmask] = colors
        m[~vmask] = 0.35
        return m

    dofa_pca_map = scatter(pca_rgb(dofa_cat.transpose(1, 2, 0)[vmask]))
    ctx_map = scatter(pca_rgb(ctx[vmask]))

    def pc1_gray(feats):
        x = feats - feats.mean(0, keepdims=True)
        x = x / (x.std(0, keepdims=True) + 1e-6)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        pc = u[:, 0] * s[0]
        lo, hi = np.percentile(pc, [2, 98])
        g = np.clip((pc - lo) / max(hi - lo, 1e-6), 0, 1)
        m = np.zeros((128, 128), np.float32)
        m[vmask] = g
        m[~vmask] = 0.35
        return m

    loc_map = pc1_gray(loc[vmask])

    rec_dofa = reconstruct_rgb(dofa_cat.transpose(1, 2, 0)[vmask], target, 128)
    rec_any = reconstruct_rgb(np.concatenate([ctx, loc], -1)[vmask], target, 128)

    def rec_map(rec):
        m = np.zeros((128, 128, 3), np.float32)
        for i, b in enumerate([2, 1, 0]):
            lo, hi = np.percentile(rec[:, b], [2, 98])
            flat = np.zeros(128 * 128, np.float32)
            flat[vmask.ravel()] = np.clip((rec[:, b] - lo) / max(hi - lo, 1e-6), 0, 1)
            m[..., i] = flat.reshape(128, 128)
        m[~vmask] = 0.35
        return m

    rec_dofa_map, rec_any_map = rec_map(rec_dofa), rec_map(rec_any)

    panels = [
        (rgb, f"输入 {result['sample_id']}\nB04/B03/B02，2–98% 拉伸；灰色=无效像元"),
        (label_map(), f"滑坡标注\n黄色=滑坡（{result['label_px']} 像元），深灰=有效无滑坡"),
        (dofa_l4, "DOFA 第 4 级 token 三主成分\n14×14 格（≈91m）放大，格边界可见"),
        (dofa_pca_map, "DOFA 四级特征拼 3072 维三主成分\n项目解码实际可用的那一份"),
        (ctx_map, "AnySat 上下文段 768 维三主成分\n40m 方格内恒定"),
        (loc_map, "AnySat 局部段第一主成分（灰度）\n10m 子 patch，逐像元，行相关见结论四"),
        (rec_dofa_map, "DOFA 特征线性还原可见光\n"
                       f"留出像元 R²={result['diagnostics']['linear_recoverability']['dofa_concat_10m']['r2_mean']}"),
        (rec_any_map, "AnySat 特征线性还原可见光\n"
                      f"留出像元 R²={result['diagnostics']['linear_recoverability']['anysat_dense_10m']['r2_mean']}"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 10.4))
    for ax, (img, title) in zip(axes.ravel(), panels):
        gray = img.ndim == 2
        ax.imshow(img, interpolation="nearest", cmap="gray" if gray else None, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10.5, linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"官方 AnySat / DOFA 对同一块 Sentinel-2 的原生输出"
        f"（{result['event_id']} {result['date']}，只读前向，formal_updates=0）",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.08, hspace=0.18)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{result['sample_id']}.png", dpi=150)
    plt.close(fig)


OUT = ROOT / "outputs" / "native_probe"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=["hiroshima_s2_1000", "thrissur_s2_1001",
                                                   "chimanimani_s2_1001"])
    ap.add_argument("--patch-sizes", type=int, nargs="+", default=[40, 160])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[env] device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'}")

    dofa, dofa_info = build_dofa(device)
    anysat, anysat_info = build_anysat(device)
    print(f"[model] DOFA params={dofa_info['params_total'] / 1e6:.1f}M "
          f"stride={dofa_info['dense_stride_px']}px decoder_in_ckpt={dofa_info['checkpoint_has_decoder']}")
    print(f"[model] AnySat params={anysat_info['params_total'] / 1e6:.1f}M depth={anysat_info['depth']}")

    meta = read_meta(args.tiles)
    results = []
    for sid in args.tiles:
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        r = run_tile(sid, meta[sid], dofa, dofa_info, anysat, tuple(args.patch_sizes), device,
                     not args.no_figures)
        r["peak_gpu_mem_mb"] = (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                if device.type == "cuda" else None)
        results.append(r)
        (OUT / f"{sid}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[tile] {sid}: dofa {r['outputs']['dofa_s2']['levels'][-1]['shape']} "
              f"{r['outputs']['dofa_s2']['latency_s']}s | anysat dense "
              f"{r['outputs']['anysat']['patch_40m']['outputs']['dense']['shape']} "
              f"{r['outputs']['anysat']['patch_40m']['latency_s']}")

    summary = {
        "purpose": "官方 AnySat / DOFA 原样前向：S2 十波段图块进去，记录实际输出形态与诊断",
        "formal_updates": 0,
        "env": {"torch": torch.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "dofa": dofa_info,
        "anysat": anysat_info,
        "inputs": {"cache_dir": str(CACHE_DIR.relative_to(ROOT)), "manifest": str(MANIFEST.relative_to(ROOT))},
        "tiles": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()

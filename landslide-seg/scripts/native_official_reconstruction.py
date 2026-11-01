"""能否拿到官方「最终产物」：DOFA 按波长出图 / AnySat 的对应物。

官方双路里，只有 DOFA 带像素解码器（``pretraining/models_base_ofa_mae.py`` 的
``Dynamic_MLP_Decoder``，按请求波长现场生成反卷积核）。AnySat 是 JEPA，解码器输出的是
被遮挡 patch 的**隐表示**（``predicted_tokens``），没有像素头。

本脚本只做一件事：把真实 S2 图块送进官方 MAE 的 encoder→decoder，看能不能得到重建影像，
并把"权重缺失"与"接口不通"区分开：

- 官方发布权重 ``DOFA_ViT_base_e100.pth`` 只有编码器；解码器参数全部缺失，随机初始化。
- 因此形状能跑通、能出图，但图与输入无关（用留出像元 R² 量化）。

用法::

    python -X utf8 scripts/native_official_reconstruction.py --tiles hiroshima_s2_1000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOFA_CODE = ROOT / "vendor" / "dofa-official-0cfb7e1"
DOFA_CKPT = ROOT / "weights" / "dofa" / "XShadow_DOFA_base" / "DOFA_ViT_base_e100.pth"
CACHE_DIR = ROOT / "data" / "s2_10band_v4_p20"
MANIFEST = ROOT / "manifests" / "v4r1" / "pool_100.jsonl"
OUT = ROOT / "outputs" / "native_reconstruction"

WAVES_S2 = (0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.832, 0.864, 1.61, 2.2)
BANDS_S2 = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
#: 训练波段里没有、但 DOFA 声称可以按波长解码出来的两个波段（海岸气溶胶 443nm、水汽 945nm）
WAVES_UNSEEN = (0.443, 0.945)
UNSEEN_NAMES = ("B01_443nm", "B09_945nm")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build_mae(device: torch.device):
    """官方 MAE；``pretraining/`` 必须在 sys.path 前面，否则 wave_dynamic_layer 取到缺类的根副本。"""
    # 顺序要紧：官方仓库根目录下的 wave_dynamic_layer 缺 GaussianFourierFeatureTransform，
    # 只有 pretraining/ 里那一份是完整的；它必须排在 sys.path 更前面。
    for p in (DOFA_CODE, DOFA_CODE / "pretraining"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import timm

    # 官方 __init__ 会 timm.create_model(..., pretrained=True) 拉 ImageNet 权重当蒸馏教师；
    # 重建链路不使用 teacher 分支，这里改为不下载。仅影响 teacher 参数，不碰 encoder/decoder。
    _orig_create = timm.create_model

    def _create_no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return _orig_create(*args, **kwargs)

    timm.create_model = _create_no_pretrained
    try:
        mae_module = load_module("native_recon_dofa_mae", DOFA_CODE / "pretraining" / "models_base_ofa_mae.py")

        # 官方 initialize_weights 末尾要加载两个**未随仓库/权重发布**的资产：
        #   1) timm 的 ImageNet teacher（联网下载）；
        #   2) weight_generator_1000_0.01_er50k.pt（波长权重生成器的预训练初值，zip 里没有）。
        # 两者都只影响 teacher 与 decoder 的初值；decoder 权重本来就缺失。这里按官方同样的
        # 正则初始化重写一遍，只去掉这两次外部加载，不改任何结构。
        def _init_weights_no_external_assets(self):
            pos_embed = mae_module.get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            dec_pos = mae_module.get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos).float().unsqueeze(0))
            torch.nn.init.normal_(self.cls_token, std=.02)
            torch.nn.init.normal_(self.mask_token, std=.02)
            self.apply(self._init_weights)

        mae_module.MaskedAutoencoderViT.initialize_weights = _init_weights_no_external_assets
        model = mae_module.mae_vit_base_patch16_dec512d8b(img_size=224)
    finally:
        timm.create_model = _orig_create

    payload = torch.load(DOFA_CKPT, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    loaded = model.load_state_dict(state, strict=False)
    encoder_missing = [k for k in loaded.missing_keys if not k.startswith(("decoder", "teacher"))]
    decoder_missing = [k for k in loaded.missing_keys if k.startswith("decoder")]
    teacher_missing = [k for k in loaded.missing_keys if k.startswith("teacher")]
    if encoder_missing:
        raise RuntimeError(f"编码器权重不完整: {encoder_missing[:5]}")
    model.eval().to(device)
    info = {
        "backend": "vendor/dofa-official-0cfb7e1/pretraining/models_base_ofa_mae.py:mae_vit_base_patch16_dec512d8b",
        "checkpoint": str(DOFA_CKPT.relative_to(ROOT)),
        "checkpoint_keys": len(state),
        "params_total_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "params_decoder_M": round(sum(p.numel() for n, p in model.named_parameters()
                                      if n.startswith("decoder")) / 1e6, 1),
        "missing_decoder_keys": len(decoder_missing),
        "missing_decoder_tensors": decoder_missing,
        "missing_teacher_keys": len(teacher_missing),
        "missing_encoder_keys": encoder_missing,
        "unexpected_keys": list(loaded.unexpected_keys),
        "note": "解码器缺失＝官方未发布；teacher 仅用于预训练蒸馏损失，重建链路不经过它，故未下载其 ImageNet 权重",
    }
    return model, info


def load_tile(sample_id: str, size: int) -> tuple[torch.Tensor, np.ndarray, dict]:
    data = np.load(CACHE_DIR / f"{sample_id}.npz")
    dn = data["x10"].astype(np.float32)
    valid = data["valid"].astype(bool)
    refl = np.where(valid[None], dn, 0.0) / 10000.0
    t = torch.from_numpy(refl)[None]
    if size != refl.shape[-1]:
        t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    meta = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["sample_id"] == sample_id:
                meta = rec
                break
    return t, valid, meta


def held_out_r2(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, seed: int = 1234) -> dict:
    """逐波段：预测值对真值的留出 R²（不用任何拟合，纯比较）。"""
    m = valid
    y, p = target[:, m], pred[:, m]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(y.shape[1])
    te = idx[: max(len(idx) // 5, 1)]
    sse = ((y[:, te] - p[:, te]) ** 2).sum(1)
    sst = ((y[:, te] - y[:, te].mean(1, keepdims=True)) ** 2).sum(1) + 1e-12
    r2 = 1 - sse / sst
    corr = [float(np.corrcoef(y[i, te], p[i, te])[0, 1]) if p[i, te].std() > 1e-8 else 0.0
            for i in range(y.shape[0])]
    return {"r2_per_band": [round(float(v), 4) for v in r2],
            "corr_per_band": [round(v, 4) for v in corr],
            "r2_mean": round(float(np.mean(r2)), 4)}


def stretch(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        lo, hi = np.percentile(img[i][valid], [2, 98])
        out[i] = np.clip((img[i] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def make_figure(sample_id: str, inp: np.ndarray, pred: np.ndarray, unseen: np.ndarray,
                valid: np.ndarray, r2: dict, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ci = stretch(inp, valid)
    cp = stretch(pred, valid)
    rgb_i = np.stack([ci[2], ci[1], ci[0]], -1)
    rgb_p = np.stack([cp[2], cp[1], cp[0]], -1)
    rgb_i[~valid] = 0.35
    rgb_p[~valid] = 0.35
    gray_i = ci[6].copy(); gray_i[~valid] = 0.35
    gray_p = cp[6].copy(); gray_p[~valid] = 0.35
    un = stretch(unseen, valid)

    fig, axes = plt.subplots(2, 4, figsize=(17, 9.6))
    panels = [
        (rgb_i, f"输入 {sample_id}\nB04/B03/B02，反射率 0–1", None),
        (rgb_p, "官方 MAE 解码器输出（同一组 10 个波长）\n随机解码器 → 不是重建", None),
        (gray_i, "输入 B08（842nm）", "gray"),
        (gray_p, "解码器输出 B08", "gray"),
        (un[0], f"解码器在未见波长 {UNSEEN_NAMES[0]} 上的输出\n形状通路可用，内容无意义", "gray"),
        (un[1], f"解码器在未见波长 {UNSEEN_NAMES[1]} 上的输出", "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel()[:6], panels):
        ax.imshow(img, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10.5, linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])

    # 散点：预测 B04 对 输入 B04
    ax = axes.ravel()[6]
    ax.scatter(inp[2][valid], pred[2][valid], s=2, alpha=0.35, color="#006335")
    ax.set_xlabel("输入 B04 反射率"); ax.set_ylabel("解码器输出 B04")
    ax.set_title(f"输入 vs 输出（B04）\n留出 R²={r2['r2_per_band'][2]:+.3f}", fontsize=10.5)
    ax.grid(alpha=0.25)

    ax = axes.ravel()[7]
    ax.axis("off")
    txt = ("读法\n"
           "· 第 1、2 格本应一致：一致才叫重建。\n"
           "  现在不一致，因为解码器 119 个\n"
           "  张量的权重未发布。\n"
           "· 第 5、6 格说明“按波长出图”的\n"
           "  接口是通的：换波长只改通道数。\n"
           f"· 十波段留出 R² 均值 {r2['r2_mean']:+.2f}，\n"
           f"  最好的单波段也仅 {max(r2['r2_per_band']):+.2f}。\n"
           "· 结论：形状可行、数值无意义，\n"
           "  要拿到官方声称的产物必须自训解码器。")
    ax.text(0.0, 0.95, txt, va="top", ha="left", fontsize=10, linespacing=1.8)

    fig.suptitle("DOFA 官方 MAE 解码器对一块 Sentinel-2 的“重建”尝试（只读前向，formal_updates=0）",
                 fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, wspace=0.08, hspace=0.20)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def unpatchify(x: torch.Tensor, p: int = 16) -> torch.Tensor:
    """官方 ``unpatchify`` 的等价实现。

    官方函数用 ``in_chan = x.shape[2] / (p**2)`` 得到浮点数再 reshape，在 torch 2.11 上直接抛
    ``TypeError``（shape 里出现 float）。这里只把该值取整，其余 reshape/einsum 顺序与官方一致。
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]
    c = x.shape[2] // (p * p)
    x = x.reshape(x.shape[0], h, w, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], c, h * p, w * p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=["hiroshima_s2_1000"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, info = build_mae(device)
    print(f"[model] params={info['params_total_M']}M decoder={info['params_decoder_M']}M "
          f"missing_decoder={info['missing_decoder_keys']}")

    results = []
    for sid in args.tiles:
        x, valid, meta = load_tile(sid, args.size)
        x = x.to(device)
        waves = torch.tensor(WAVES_S2, dtype=torch.float32)
        with torch.no_grad():
            latent, mask, ids_restore = model.forward_encoder(x, 0.0, list(WAVES_S2))
            pred_patches = model.forward_decoder(latent, ids_restore, x.shape[1])
            pred = unpatchify(pred_patches)[0].float().cpu().numpy()
            # 换一组训练里从未出现的波长，走同一条解码通路。
            # 解码器要的是编码器里那套"波长→128 维"投影后的表示（Dynamic_MLP_OFA 内部做法），
            # 直接喂原始微米值会因维度不对报错，所以这里复用同一个 fclayer。
            from util.pos_embed import get_1d_sincos_pos_embed_from_grid_torch
            pe = model.patch_embed
            wvs = torch.tensor(WAVES_UNSEEN, dtype=torch.float32, device=device) * 1000
            model.waves = pe.fclayer(get_1d_sincos_pos_embed_from_grid_torch(pe.wv_planes, wvs))
            unseen_patches = model.forward_decoder(latent, ids_restore, len(WAVES_UNSEEN))
            unseen = unpatchify(unseen_patches)[0].float().cpu().numpy()

        inp_np = x[0].float().cpu().numpy()
        # 有效掩膜跟着输入一起放大到模型网格（最近邻，避免把边界糊进来）
        valid_m = torch.nn.functional.interpolate(
            torch.from_numpy(valid)[None, None].float(), size=pred.shape[-2:], mode="nearest"
        )[0, 0].numpy().astype(bool)
        r2 = held_out_r2(pred, inp_np, valid_m)
        rec = {
            "sample_id": sid,
            "event_id": meta.get("event_id"),
            "date": meta.get("date"),
            "protocol_role": meta.get("protocol_role"),
            "label_px": int(np.load(CACHE_DIR / f"{sid}.npz")["mask"].sum()),
            "valid_ratio": round(float(valid.mean()), 4),
            "input": {"shape": list(inp_np.shape), "wavelengths_um": list(WAVES_S2)},
            "decoder_output_same_wavelengths": {
                "shape": list(pred.shape),
                "min": round(float(pred.min()), 4), "max": round(float(pred.max()), 4),
                "mean": round(float(pred.mean()), 4), "std": round(float(pred.std()), 4),
            },
            "vs_input": r2,
            "decoder_output_unseen_wavelengths": {
                "shape": list(unseen.shape), "wavelengths_um": list(WAVES_UNSEEN),
                "min": round(float(unseen.min()), 4), "max": round(float(unseen.max()), 4),
            },
        }
        fig_path = OUT / "figures" / f"{sid}.png"
        make_figure(sid, inp_np, pred, unseen, valid_m, r2, fig_path)
        rec["figure"] = str(fig_path.relative_to(ROOT))
        results.append(rec)
        (OUT / f"{sid}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[tile] {sid}: pred {tuple(pred.shape)} R2_mean={r2['r2_mean']:+.3f} "
              f"corr={r2['corr_per_band'][:3]}")

    summary = {
        "purpose": "官方 MAE 解码器能否产出重建影像／任意波段影像（只读前向）",
        "formal_updates": 0,
        "env": {"torch": torch.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "dofa": info,
        "anysat": {
            "has_pixel_decoder": False,
            "evidence": "src/models/networks/JEPAAny.py:forward_decoder 只写 out['predicted_tokens']（768 维隐表示），无像素投影；"
                        "hubconf 权重严格加载 Any_module 成功（missing=0），说明解码头/predictor 未随发布权重提供。",
        },
        "tiles": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()

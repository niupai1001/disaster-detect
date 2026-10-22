"""真正冻结的表征 probe：编码器不动，只训四个极小的 1×1 头。

对应执行计划的 B 组。设计要点：
- 编码器整体 ``eval()`` 且 ``requires_grad_(False)``，含 DOFA 的波长条件输入层；
- 移除预训练分割 decoder / Stem / Feature2Pyramid / UPerNet，只用编码器特征；
- 每层特征先做**无可训练仿射参数**的通道 LayerNorm，再接独立的 ``1×1→1`` 线性头；
- 四个头的 logits 双线性插值到 128×128 后**等权平均**；
- 线性头零初始化（首步输出恒为 0，与"零初始化的恒等"一致）。

编码器种类：
- ``segformer_b0`` / ``segformer_b2``：原生 3 通道 RGB 输入，ADE 预训练的 RGB 均值/标准差；
- ``dofa_rgb``：DOFA ViT，3 通道 + 3 个波长；
- ``dofa_10``：DOFA ViT，10 通道 + 10 个波长。

DOFA 官方 ``out_indices=[3,5,7,11]`` 即第 4/6/8/12 个 block，输出统一 stride 16（128 px → 8×8）、768 通道。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..config import resolve_path
from ..data.segformer_dataset import S2_10_BANDS

#: 10 波段的中心波长（微米）；与 DOFA 官方下游方案一致
S2_WAVELENGTHS_10 = (0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 1.610, 2.190)
#: RGB = B04/B03/B02 在十波段里的下标
RGB_CH = (2, 1, 0)
RGB_WAVELENGTHS = (0.665, 0.560, 0.490)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChannelNormNoAffine(nn.Module):
    """在通道维做 LayerNorm，且不带任何可训练仿射参数。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,H,W]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FrozenProbe(nn.Module):
    """``[B,10,H,W]``（已按本折统计 z-score）→ ``{"logits","layer_logits","features"}``。"""

    def __init__(self, encoder: str, pretrained_dir: str | Path, stats: dict, image_size: int = 128,
                 wavelength_version: str = 'legacy_v5') -> None:
        super().__init__()
        if encoder not in ("segformer_b0", "segformer_b2", "dofa_rgb", "dofa_10"):
            raise ValueError(f"未知 probe 编码器 {encoder!r}")
        self.encoder_kind = encoder
        if wavelength_version not in ('legacy_v5', 'named_bands_v6'):
            raise ValueError(wavelength_version)
        from .spectral_contract import wavelengths_for
        self.wavelength_version = wavelength_version
        self.wavelengths_10 = (wavelengths_for(S2_10_BANDS) if wavelength_version == 'named_bands_v6'
                               else S2_WAVELENGTHS_10)
        self.image_size = int(image_size)
        self.stats = {"mean": list(stats["mean"]), "std": list(stats["std"])}
        if len(self.stats["mean"]) != len(S2_10_BANDS):
            raise ValueError("probe 需要一个十波段的统计文件")

        if encoder.startswith("segformer"):
            from transformers import SegformerForSemanticSegmentation
            full = SegformerForSemanticSegmentation.from_pretrained(str(resolve_path(pretrained_dir)))
            self.net = full.segformer
            del full
            want = 256 if encoder == "segformer_b0" else 768
            got = int(self.net.config.decoder_hidden_size) if hasattr(self.net.config, "decoder_hidden_size") else None
            # 编码器通道数才是关心项；用配置里的 hidden_sizes 记录真实通道
            self.encoder_channels = list(self.net.config.hidden_sizes)
            self.encoder_strides = [4, 8, 16, 32]
            self.weight_identity = {"dir": str(resolve_path(pretrained_dir)).replace("\\", "/"),
                                    "hidden_sizes": self.encoder_channels, "decoder_hidden": got}
        else:
            src = resolve_path("vendor/dofa-official-0cfb7e1")
            for p in (src, resolve_path("vendor/dofa_runtime"), src / "pretraining", src / "pretraining/util"):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            from downstream_tasks.geobench_segmentation.model import vit_base_patch16
            self.net = vit_base_patch16(img_size=self.image_size, drop_path_rate=0.)
            ckpt = resolve_path("weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth")
            if not ckpt.is_file():
                cands = sorted((resolve_path("weights/dofa")).rglob("*.pth"))
                if not cands:
                    raise FileNotFoundError("找不到 DOFA 预训练权重")
                ckpt = cands[0]
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            state = state.get("model", state)
            pos = state["pos_embed"]
            if pos.shape[1] - 1 != self.net.pos_embed.shape[1] - 1:
                # 128 px → 64 patch；官方 ckpt 为 224 px → 196，按双三次插值对齐
                n_old = int(round((pos.shape[1] - 1) ** 0.5))
                n_new = int(round((self.net.pos_embed.shape[1] - 1) ** 0.5))
                cls, grid = pos[:, :1], pos[:, 1:].reshape(1, n_old, n_old, -1).permute(0, 3, 1, 2)
                grid = nn.functional.interpolate(grid, size=(n_new, n_new), mode="bicubic", align_corners=False)
                pos = torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, n_new * n_new, -1)], dim=1)
            state["pos_embed"] = pos
            info = self.net.load_state_dict(state, strict=False)
            allowed = {"mask_token", "norm.weight", "norm.bias", "projector.weight", "projector.bias"}
            if info.missing_keys or set(info.unexpected_keys) - allowed:
                raise ValueError(f"DOFA 权重不完整: {info}")
            self.encoder_channels = [768] * 4
            self.encoder_strides = [16] * 4
            self.weight_identity = {"checkpoint": str(ckpt).replace("\\", "/"),
                                    "out_indices": list(self.net.out_indices),
                                    "unexpected": sorted(info.unexpected_keys)}

        self.net.requires_grad_(False)
        self.net.eval()

        self.norms = nn.ModuleList([ChannelNormNoAffine(c) for c in self.encoder_channels])
        self.heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in self.encoder_channels])
        for h in self.heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.head_params = sum(p.numel() for p in list(self.norms.parameters()) + list(self.heads.parameters()))
        # 训练器/checkpoint 会直接读取这些身份字段
        self.adapter_type = f"frozen_{encoder}"
        self.backbone = encoder
        self.use_stem = False
        self.refine = "none"
        self.fusion_mode = None
        self.exchange_mode = None
        self.load_info = dict(self.weight_identity)
        self.adapter_report = {
            "type": "frozen_probe",
            "encoder": encoder,
            "encoder_channels": list(self.encoder_channels),
            "encoder_strides": list(self.encoder_strides),
            "head": "per-layer ChannelNorm(no affine) + 1x1x1 conv, zero-init; logits averaged after bilinear upsample",
            "trainable_params": self.head_params,
            "encoder_frozen": True,
            "wavelength_version": self.wavelength_version,
            "band_order": list(S2_10_BANDS),
            "wavelengths_10": list(self.wavelengths_10),
        }
        self._enc_digest0 = self._digest()

    # ---------- 预处理 ----------
    def _prep(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[float, ...] | None]:
        """输入是已按十波段统计 z-score 的张量；这里换成各编码器要求的输入。"""
        mean = torch.tensor(self.stats["mean"], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.stats["std"], device=x.device, dtype=x.dtype)
        dn = x * std[None, :, None, None] + mean[None, :, None, None]  # 还原到 DN

        if self.encoder_kind.startswith("segformer"):
            rgb = torch.stack([dn[:, i] for i in RGB_CH], dim=1) / 10000.0   # 反射率
            m = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype)[None, :, None, None]
            s = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype)[None, :, None, None]
            return (rgb - m) / s, None
        if self.encoder_kind == "dofa_rgb":
            # 十波段 DOFA 用的是逐波段 z-score；RGB 版取对应三通道，保持共同波段预处理完全一致
            return torch.stack([x[:, i] for i in RGB_CH], dim=1), RGB_WAVELENGTHS
        return x, self.wavelengths_10

    # ---------- 前向 ----------
    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        inp, wl = self._prep(x)
        if self.encoder_kind.startswith("segformer"):
            out = self.net(pixel_values=inp, output_hidden_states=True)
            hs = out.hidden_states
            feats = []
            for i, h in enumerate(hs):
                if h.ndim == 3:  # [B, N, C] → [B, C, h, w]
                    hw = int(round(h.shape[1] ** 0.5))
                    h = h.reshape(h.shape[0], hw, hw, h.shape[2]).permute(0, 3, 1, 2).contiguous()
                feats.append(h)
            return feats
        return self.net(inp, wl)   # 官方接口已返回 4 个 (B,768,8,8)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self._features(x)
        logits = []
        for f, n, h in zip(feats, self.norms, self.heads):
            z = h(n(f))
            logits.append(nn.functional.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False))
        merged = torch.stack(logits, dim=0).mean(dim=0)
        return {"logits": merged, "layer_logits": logits, "features": feats}

    # ---------- 训练器需要的接口 ----------
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"encoder": [], "decoder": [p for p in self.parameters() if p.requires_grad], "new": []}

    def count_params(self) -> dict:
        enc = sum(p.numel() for p in self.net.parameters())
        head = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": enc + head, "trainable": head, "encoder": enc, "decoder": head, "new": 0}

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()          # 编码器永远 eval（BN/LN 统计不更新）
        self.net.requires_grad_(False)
        return self

    def probe_integrity(self) -> dict:
        """训练器每 100 更新调用；编码器必须逐位未变、无梯度、处于 eval。"""
        grad = any(p.requires_grad or p.grad is not None for p in self.net.parameters())
        return {"encoder_unchanged": self._digest() == self._enc_digest0,
                "encoder_has_grad": grad,
                "encoder_training": bool(self.net.training)}

    def _digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for k, v in sorted(self.net.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

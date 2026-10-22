"""单头语义—细节注意力融合（N1 SUM / N2 AFF）。

执行计划：``docs/single_head_attention_execution_plan_2026-10-19.md``。

架构（无粗输出、无 C+Δ 残差）：

    D = 官方 MiT-B0 + 官方 SegFormer 解码器特征 (B,256,h,w)
    S = 现有 SpectralStem 细节分支          (B,32,H,W)
    U = up(GN8(Conv1x1(D,256→64)))
    V = GN8(Conv1x1(S,32→64))
    Z = U + V              (N1)
      = AFF(U,V)           (N2)
    F = Conv3x3(64,64,pad1) → GN8 → GELU → Conv1x1(64,1)

AFF 参考：Dai et al., "Attentional Feature Fusion", WACV 2021；
参考实现 https://github.com/YimianDai/open-aff （``aff_pytorch/aff_net/fusion.py``）。
本项目适配：单次 AFF、channels=64、r=4；两条注意力支路末端 BN 采用零缩放/零偏置
（原实现为常规初始化），使初始 W≡0.5、Z 严格退化为 U+V，便于 N1/N2 共享初始化对照。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer import (
    BACKBONE_DECODER_HIDDEN,
    SpectralProjection,
    _load_pretrained,
    adapt_first_conv_direct10,
)
from .spectral_stem import SpectralStem

AFF_SOURCE = {
    "paper": "Attentional Feature Fusion, WACV 2021",
    "repo": "https://github.com/YimianDai/open-aff",
    "file": "aff_pytorch/aff_net/fusion.py",
    "adaptation": "single AFF, channels=64, r=4, last BN of both branches zero-init (W=0.5)",
}


class AFF(nn.Module):
    """单次 AFF：``Z = 2W⊙U + 2(1-W)⊙V``，W 由局部 + 全局分支生成。"""

    def __init__(self, channels: int = 64, r: int = 4) -> None:
        super().__init__()
        d = max(channels // r, 8)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        for m in (self.local_att[-1], self.global_att[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = u + v
        w = torch.sigmoid(self.local_att(s) + self.global_att(s))
        return 2.0 * w * u + 2.0 * (1.0 - w) * v, w


class SingleHeadFusion(nn.Module):
    """对齐投影 + SUM/AFF 融合 + 单个最终预测头。"""

    def __init__(
        self,
        decoder_ch: int = 256,
        stem_ch: int = 32,
        width: int = 64,
        fusion: str = "sum",
        groups: int = 8,
    ) -> None:
        super().__init__()
        if fusion not in ("sum", "aff"):
            raise ValueError(f"fusion 必须是 sum/aff，得到 {fusion!r}")
        self.fusion = fusion
        self.width = width
        self.proj_d = nn.Sequential(
            nn.Conv2d(decoder_ch, width, 1, bias=False), nn.GroupNorm(groups, width)
        )
        self.proj_s = nn.Sequential(
            nn.Conv2d(stem_ch, width, 1, bias=False), nn.GroupNorm(groups, width)
        )
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.head[-1].bias)  # 标准小随机权重，bias=0
        # AFF 最后创建：N1/N2 的共享张量在相同 RNG 序列下完全一致
        self.aff = AFF(width, 4) if fusion == "aff" else None

    def forward(
        self, decoder_feat: torch.Tensor, stem_feat: torch.Tensor, size,
        w_override: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        u = F.interpolate(self.proj_d(decoder_feat), size=size, mode="bilinear", align_corners=False)
        v = self.proj_s(stem_feat)
        if self.aff is not None:
            if w_override is None:
                z, w = self.aff(u, v)
            else:
                w = torch.as_tensor(w_override, dtype=u.dtype, device=u.device)
                z = 2.0 * w * u + 2.0 * (1.0 - w) * v
        else:
            z, w = u + v, None
        return self.head(z), w


class SingleHeadSeg(nn.Module):
    """单头模型：官方编码器/解码器特征 + 细节分支 + 单头融合；无粗输出。"""

    def __init__(
        self,
        *,
        adapter: str = "direct10",
        backbone: str = "b0",
        stem: bool = True,
        fusion: str = "sum",
        width: int = 64,
        pretrained_dir: str = "weights/segformer/segformer-b0-finetuned-ade-512-512",
        in_chans: int = 10,
        band_order: list[str] | None = None,
        stem_width: int = 32,
    ) -> None:
        super().__init__()
        from landslide_seg.data.segformer_dataset import S2_10_BANDS

        self.band_order = list(band_order or S2_10_BANDS)
        assert len(self.band_order) == in_chans, self.band_order
        if adapter not in ("direct10", "project10to3"):
            raise ValueError(f"adapter 必须是 direct10/project10to3，得到 {adapter!r}")
        if backbone not in BACKBONE_DECODER_HIDDEN:
            raise ValueError(f"backbone 必须是 {sorted(BACKBONE_DECODER_HIDDEN)}，得到 {backbone!r}")
        if not stem:
            raise ValueError("单头模型需要细节分支（stem=true）")
        if fusion not in ("sum", "aff"):
            raise ValueError(f"fusion 必须是 sum/aff，得到 {fusion!r}")

        base, load_info = _load_pretrained(pretrained_dir)
        expected_hidden = BACKBONE_DECODER_HIDDEN[backbone]
        got_hidden = int(base.config.decoder_hidden_size)
        if got_hidden != expected_hidden:
            raise ValueError(
                f"backbone={backbone!r} 期望解码器 {expected_hidden} 维，但权重是 {got_hidden} 维"
            )
        self.backbone = backbone
        self.in_chans = in_chans
        self.adapter_type = adapter
        self.use_stem = True
        self.refine = "single"
        self.fusion_mode = fusion
        self.width = width
        decoder_hidden = got_hidden

        # 最终分类移到融合后：移除官方二分类头，避免未参与计算的参数
        base.decode_head.classifier = nn.Identity()
        self.cls_report = {
            "replaced": f"Conv2d({decoder_hidden},{base.config.num_labels},1) -> Identity",
            "reason": "单头模型把最终预测移到融合之后",
        }
        self.model = base

        self.projection = None
        if adapter == "direct10":
            self.adapter_report = adapt_first_conv_direct10(base, in_chans, self.band_order)
        else:
            self.projection = SpectralProjection(in_chans, self.band_order)
            self.adapter_report = self.projection.describe()

        self.stem = SpectralStem(in_chans, stem_width)
        self.fusion_head = SingleHeadFusion(decoder_hidden, stem_width, width, fusion)
        self.load_info = {
            "missing_keys": list(load_info.get("missing_keys", [])),
            "unexpected_keys": list(load_info.get("unexpected_keys", [])),
            "mismatched_keys": [str(k) for k in load_info.get("mismatched_keys", [])],
            "error_msgs": list(load_info.get("error_msgs", [])),
        }

    def forward(
        self, x: torch.Tensor, *, return_attn: bool = False,
        attn_override: torch.Tensor | float | None = None,
    ) -> dict:
        if x.ndim != 4 or x.shape[1] != self.in_chans:
            raise ValueError(f"期望 [B,{self.in_chans},H,W]，得到 {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if self.projection is not None:
            x = self.projection(x)
        enc = self.model.segformer(pixel_values=x, output_hidden_states=True)
        d = self.model.decode_head(enc.hidden_states)
        s = self.stem(x)
        logits, attn = self.fusion_head(d, s, (h, w), w_override=attn_override)
        res = {"logits": logits, "base_logits": None}
        if return_attn:
            res["attn"] = attn
        return res

    # ------------------------------------------------------------------
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        enc, dec, new = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("model.segformer."):
                enc.append(p)
            elif name.startswith("model.decode_head."):
                dec.append(p)
            else:
                new.append(p)
        return {"encoder": enc, "decoder": dec, "new": new}

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        groups = self.param_groups()
        return {
            "total": total,
            "trainable": trainable,
            "encoder": sum(p.numel() for p in groups["encoder"]),
            "decoder": sum(p.numel() for p in groups["decoder"]),
            "new": sum(p.numel() for p in groups["new"]),
        }

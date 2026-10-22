"""原分辨率光谱分支（Spectral Stem）。

设计来自执行计划 §4：在**共同 10m 网格**上做 stride=1 的轻量卷积，保留 SegFormer
解码器 1/4 网格早就不具备的细节，再与上采样的解码特征拼接，走一个残差头输出
``ΔL``。最终 ``L = L_base + ΔL``。

三个刻意的约束：
1. 全部 stride=1（1×1 → 3×3，padding=1），无池化、无下采样——"原网格"不是修辞；
2. 残差头末层零初始化 → 初始输出 ``ΔL ≡ 0``，与关闭 Stem 的模型严格等价；
3. 零初始化会让末层上一层在**首次 backward 前**没有梯度，属于预期行为；
   训练若干步后仍无更新才说明断路（脚本里单独检查）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralStem(nn.Module):
    """``[B,10,H,W] → [B,32,H,W]``，stride 全为 1。"""

    def __init__(self, in_chans: int = 10, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualFusionHead(nn.Module):
    """``concat(up(D), stem) → ΔL``；末层零初始化。"""

    def __init__(self, decoder_ch: int = 256, stem_ch: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(decoder_ch + stem_ch, hidden, 1, bias=True),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, decoder_feat: torch.Tensor, stem_feat: torch.Tensor, size) -> torch.Tensor:
        d = F.interpolate(decoder_feat, size=size, mode="bilinear", align_corners=False)
        h = torch.cat([d, stem_feat], dim=1)
        return self.out(self.body(h))

class CoarseGuidedRefiner(nn.Module):
    """粗定位引导的精分头：``concat(up(D), S, P)`` → ΔL，末层零初始化。

    ``P = sigmoid(up(C))`` 是粗阶段概率；``zero_cond`` 变体把 P 固定为 0，
    用于区分"显式定位条件"与"新增精分头容量"的贡献（CF0/CF1 消融）。
    """

    def __init__(self, decoder_ch: int = 256, stem_ch: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(decoder_ch + stem_ch + 1, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, decoder_feat: torch.Tensor, stem_feat: torch.Tensor, cond: torch.Tensor, size):
        d = F.interpolate(decoder_feat, size=size, mode="bilinear", align_corners=False)
        h = torch.cat([d, stem_feat, cond], dim=1)
        return self.out(self.body(h))

"""阶段 2：精细分割解码器（U-Net 式，轻量）。

输入：
- 融合金字塔 ``P0..P3``（10/20/40/80 m）；
- **粗阶段预测**的 soft 概率图（不是 GT，训练时也只用预测，§4.5）；
- 语义模块给出的 FiLM 参数 ``gamma/beta``。

输出：``[B,1,H,W]`` 细分割 logits。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out, eps=1e-5),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out, eps=1e-5),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FineRefiner(nn.Module):
    """由粗到细的解码器。

    Args:
        channels: 融合特征通道 C。
        hidden: 解码器内部通道数。
        n_levels: 金字塔层数（与融合器一致）。
    """

    def __init__(self, channels: int = 128, hidden: int = 96, n_levels: int = 4):
        super().__init__()
        self.n_levels = n_levels
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-5),
            nn.GELU(),
        )
        self.up_blocks = nn.ModuleList(
            [_Block(hidden + channels, hidden) for _ in range(n_levels - 1)]
        )
        # 最细层额外接收粗概率图 + 语义相似度
        self.fuse_coarse = _Block(hidden + channels + 2, hidden)
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(
        self,
        pyramid: Sequence[torch.Tensor],
        coarse_prob: torch.Tensor,
        *,
        sem_map: torch.Tensor | None = None,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """
        Args:
            pyramid: ``[P0..Pn]``，``P0`` 最细。
            coarse_prob: ``[B,1,h0,w0]`` 粗阶段 sigmoid 概率。
            sem_map: 可选，``[B,1,H,W]`` 语义相似度，作为额外空间先验。
            film: 可选 ``(gamma, beta)``，形状 ``[1,C]``，对特征做通道调制。
        """
        n = len(pyramid)
        x = self.stem(pyramid[-1])
        for i in range(n - 2, -1, -1):
            x = F.interpolate(
                x, size=pyramid[i].shape[-2:], mode="bilinear", align_corners=False
            )
            x = torch.cat([x, pyramid[i]], dim=1)
            x = self.up_blocks[i](x)

        extra = [coarse_prob]
        if sem_map is None:
            sem_map = torch.zeros_like(coarse_prob)
        if sem_map.shape[-2:] != coarse_prob.shape[-2:]:
            sem_map = F.interpolate(
                sem_map, size=coarse_prob.shape[-2:], mode="bilinear", align_corners=False
            )
        extra.append(sem_map)

        # 语义 FiLM 调制最细层融合特征（通道数 = C，与 gamma 对齐）
        p0 = pyramid[0]
        if film is not None:
            gamma, beta = film
            if gamma.shape[-1] == p0.shape[1]:
                # 注意：gamma 形状是 [B,C]，不能用 view(1,-1,1,1)（会把 batch 折进通道）
                p0 = p0 * (1 + gamma[..., None, None]) + beta[..., None, None]

        x = torch.cat([x, p0, *extra], dim=1)
        x = self.fuse_coarse(x)
        return {"fine_logits": self.head(x), "fine_feature": x}

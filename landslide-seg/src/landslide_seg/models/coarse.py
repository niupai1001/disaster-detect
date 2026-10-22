"""阶段 1：高召回粗定位 + 图块级存在性。

输出：
- ``coarse_logits`` ``[B,1,H,W]``：在融合金字塔最细层分辨率上的二值 logits；
- ``presence_logits`` ``[B,1]``：图块内是否存在滑坡。

语义相似度热图在此处以**可学习标量**耦合进粗 logits（§4.4 用途 2），
初始权重很小，避免训练早期被未对齐的语义压过视觉证据。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseLocator(nn.Module):
    """轻量 FPN 头部：自顶向下融合 + 粗 mask + presence。

    Args:
        channels: 融合金字塔通道数 C。
        hidden: 头部内部通道数。
        n_levels: 金字塔层数（与 ``FeatureFusion.level_strides`` 一致）。
    """

    def __init__(self, channels: int = 128, hidden: int = 64, n_levels: int = 4):
        super().__init__()
        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden, eps=1e-5),
                    nn.GELU(),
                )
                for _ in range(n_levels)
            ]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden, eps=1e-5),
                    nn.GELU(),
                )
                for _ in range(n_levels)
            ]
        )
        self.out = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-5),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.presence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, max(hidden // 2, 8)),
            nn.GELU(),
            nn.Linear(max(hidden // 2, 8), 1),
        )
        # 语义相似度的耦合强度，初始 0.1，训练可调
        self.semantic_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        pyramid: Sequence[torch.Tensor],
        sem_map: torch.Tensor | None = None,
    ) -> dict:
        """Args:
            pyramid: ``[P0..Pn]``，``P0`` 为最细层。
            sem_map: ``[B,1,H,W]`` 语义相似度热图（logit 尺度），
                按 ``semantic_scale`` 叠加到粗 logits。
        """
        n = len(pyramid)
        feats = [self.lateral[i](p) for i, p in enumerate(pyramid)]
        for i in range(n - 2, -1, -1):
            up = F.interpolate(
                feats[i + 1], size=feats[i].shape[-2:], mode="bilinear", align_corners=False
            )
            feats[i] = feats[i] + up
        feats = [self.smooth[i](f) for i, f in enumerate(feats)]

        logits = self.out(feats[0])
        if sem_map is not None:
            if sem_map.shape[-2:] != logits.shape[-2:]:
                sem_map = F.interpolate(
                    sem_map, size=logits.shape[-2:], mode="bilinear", align_corners=False
                )
            logits = logits + self.semantic_scale * sem_map
        return {
            "coarse_logits": logits,
            "presence_logits": self.presence(feats[-1]),
            "coarse_feature": feats[0],
            "fpn_features": feats,
        }

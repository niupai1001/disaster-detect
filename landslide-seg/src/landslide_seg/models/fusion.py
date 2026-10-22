"""双骨干特征融合。

输入是两支路各自输出的"特征谱"（可能尺度、通道数都不同），输出是统一的金字塔
``P_l``，每个 level 对应一个明确的物理 stride（米），方便做几何一致性测试。

关键点（Landslide-Seg V1 §4.3）：
- 每个来源先做 1x1 投影到公共通道 C；
- 每个来源按自身 physical stride 重采样到目标 level 网格；
- 来源之间的权重是可学习标量（softmax 归一化），但**不是**"置信度"，不对外解释；
- 训练时支持 branch dropout（随机屏蔽某一路），降低单路主导风险。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeatureMap:
    """带来源标注的特征图，禁止用模型名猜分辨率。"""

    tensor: torch.Tensor      # [B, C, H, W]
    source: str               # 例: "terramind.S2L2A.tim", "anysat.s2@40m"
    stride_m: float           # 该特征图 1 像素对应的地面米数
    frozen: bool = True

    @property
    def channels(self) -> int:
        return int(self.tensor.shape[1])

    def describe(self) -> dict:
        b, c, h, w = self.tensor.shape
        return {
            "source": self.source,
            "channels": c,
            "height": h,
            "width": w,
            "physical_stride_m": self.stride_m,
            "dtype": str(self.tensor.dtype),
            "frozen": self.frozen,
            "tokens": h * w,
        }


class _Project(nn.Module):
    """通道投影 + 归一化，保持空间尺寸。"""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.norm = nn.BatchNorm2d(c_out, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class FeatureFusion(nn.Module):
    """把多来源、多尺度特征融合为固定通道数的金字塔。

    Args:
        channels: 公共通道数 C。
        level_strides: 目标 level 的物理 stride（米），**必须严格递增**。
        sources: 允许出现的来源名（用于固定参数表，保证 checkpoint 可复现）。
        branch_dropout: 训练时随机屏蔽一路的概率（0 表示关闭）。
    """

    def __init__(
        self,
        channels: int = 128,
        level_strides: Sequence[float] = (10, 20, 40, 80),
        sources: Sequence[str] = (
            "terramind.S2L2A",
            "anysat.s2@40m",
            "anysat.s2@80m",
            "anysat.s2@160m",
        ),
        branch_dropout: float = 0.1,
    ):
        super().__init__()
        if list(level_strides) != sorted(level_strides):
            raise ValueError("level_strides 必须严格递增")
        self.channels = channels
        self.level_strides = tuple(float(s) for s in level_strides)
        self.sources = tuple(sources)
        self.branch_dropout = branch_dropout

        # 每个来源一个投影器；输入通道在第一次 forward 时由 build() 确定。
        # ModuleDict 的 key 不能含 "." / "@"，这里做一次稳定映射
        self.key_of = {name: self._safe_key(name) for name in self.sources}
        self.proj = nn.ModuleDict()
        # 每个 (level, source) 一个 logit，softmax 后作为融合权重
        self.gate = nn.Parameter(
            torch.zeros(len(self.level_strides), len(self.sources))
        )
        # 每个 level 的融合后细化
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels, eps=1e-5),
                    nn.GELU(),
                )
                for _ in self.level_strides
            ]
        )
        self._built = False

    @staticmethod
    def _safe_key(name: str) -> str:
        return name.replace(".", "_").replace("@", "_")

    # ------------------------------------------------------------------
    def build(self, channel_map: Dict[str, int]) -> None:
        """根据实际输入通道数建立投影层（必须在第一次 forward 前调用一次）。"""
        for name in self.sources:
            if name not in channel_map:
                raise KeyError(f"融合器缺少来源 {name} 的通道数，现有 {list(channel_map)}")
            self.proj[self.key_of[name]] = _Project(int(channel_map[name]), self.channels)
        self._built = True

    def _resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        down = size[0] < x.shape[-2]
        return F.interpolate(
            x, size=size, mode="area" if down else "bilinear", align_corners=None if down else False
        )

    def forward(
        self,
        features: Sequence[FeatureMap],
        target_size: tuple[int, int],
        *,
        training_branch_dropout: bool | None = None,
    ) -> List[torch.Tensor]:
        """Args:
            features: 各来源特征（可缺项；缺项不参与该轮融合）。
            target_size: 最细 level 的 (H, W)，用于反推每个 level 的网格尺寸。

        Returns:
            list of ``[B, C, h_l, w_l]``，``h_l = target_size[0] * stride0 / stride_l``。
        """
        if not self._built:
            raise RuntimeError("FeatureFusion.build() 未调用")

        if training_branch_dropout is None:
            drop = self.branch_dropout if self.training else 0.0
        else:
            drop = float(training_branch_dropout)

        by_source: Dict[str, torch.Tensor] = {}
        for fm in features:
            if fm.source not in self.key_of:
                continue
            if drop > 0 and torch.rand(()) < drop:
                continue
            by_source[fm.source] = self.proj[self.key_of[fm.source]](fm.tensor)

        if not by_source:
            # 极小概率所有分支同时被丢弃；保底保留 TerraMind（若存在）
            fallback = [
                fm for fm in features
                if fm.source in self.key_of and fm.source.startswith("terramind")
            ] or [fm for fm in features if fm.source in self.key_of]
            if not fallback:
                raise RuntimeError("没有可用分支，无法融合")
            fm = fallback[0]
            by_source[fm.source] = self.proj[self.key_of[fm.source]](fm.tensor)

        base_stride = self.level_strides[0]
        outputs: List[torch.Tensor] = []
        present_idx = [self.sources.index(n) for n in by_source]
        for li, stride in enumerate(self.level_strides):
            h = max(1, int(round(target_size[0] * base_stride / stride)))
            w = max(1, int(round(target_size[1] * base_stride / stride)))
            # 仅在"本轮实际参与的来源"之间做 softmax，保证权重和为 1
            w_soft = torch.softmax(self.gate[li, present_idx], dim=0)
            acc = None
            for wi, name in zip(w_soft, by_source):
                resized = self._resize(by_source[name], (h, w))
                acc = wi * resized if acc is None else acc + wi * resized
            outputs.append(self.refine[li](acc))
        return outputs

    def gate_report(self) -> List[dict]:
        """导出当前门控权重（仅用于日志/消融，不解释为置信度）。"""
        with torch.no_grad():
            w = torch.softmax(self.gate, dim=1)
        return [
            {
                "stride_m": self.level_strides[i],
                "weights": {s: float(w[i, j]) for j, s in enumerate(self.sources)},
            }
            for i in range(len(self.level_strides))
        ]

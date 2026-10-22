"""AnySat Base 适配器（复用官方 repo，冻结权重）。

AnySat 官方接口：
- 输入 ``{"s2": [B,T,10,H,W], "s2_dates": [B,T]}``，``T=1`` 表示单期；
- ``patch_size`` 单位是**米**，必须是 10 的整数倍；
- 分割必须用 ``output="dense"``，对 s2 的 dense 输出为原生 10m 网格、``2*D`` 通道。

本适配器只做封装与 provenance 记录，模型本身完全来自官方代码与官方权重。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class AnySatRunInfo:
    patch_size_m: int
    out_shape: tuple[int, ...]
    latency_s: float
    num_patches: int


def _load_hubconf(repo_dir: Path):
    """以独立模块名加载官方 ``hubconf.py``，避免包名污染。"""
    path = repo_dir / "hubconf.py"
    if not path.is_file():
        raise FileNotFoundError(f"AnySat 仓库缺少 hubconf.py: {path}")
    if str(repo_dir / "src") not in sys.path:
        sys.path.append(str(repo_dir / "src"))
    # 官方 hubconf 内部使用 ``from src.models...``，因此仓库根目录也必须在 sys.path 上
    # （torch.hub.load 会自动加，本地按文件加载时需要我们自己加）
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    spec = importlib.util.spec_from_file_location("anysat_hubconf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # type: ignore[union-attr]
    return module


class AnySatScaleAdapter(nn.Module):
    """多物理尺度的 AnySat dense 特征提取器（冻结）。

    Args:
        repo_dir: AnySat 官方仓库根目录。
        weight_path: ``AnySat.pth`` 本地路径（官方 ``g-astruc/AnySat``）。
        patch_sizes_m: 物理 patch 尺寸（米），如 ``(40, 160)``。
        freeze: 冻结参数（默认 True）。
    """

    def __init__(
        self,
        repo_dir: str | os.PathLike,
        weight_path: str | os.PathLike,
        patch_sizes_m: Sequence[int] = (40, 160),
        freeze: bool = True,
    ):
        super().__init__()
        self.repo_dir = Path(repo_dir)
        self.weight_path = Path(weight_path)
        if not self.weight_path.is_file():
            raise FileNotFoundError(f"AnySat 权重不存在: {self.weight_path}")
        for p in patch_sizes_m:
            if p % 10 != 0:
                raise ValueError(f"AnySat patch_size 必须是 10 的倍数，得到 {p}")
        self.patch_sizes_m = tuple(int(p) for p in patch_sizes_m)
        self.freeze = freeze

        hub = _load_hubconf(self.repo_dir)
        self.model = hub.AnySat(model_size="base", flash_attn=False)
        payload = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        missing, unexpected = self.model.model.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(
                f"AnySat 权重与官方结构不匹配：missing={len(missing)} unexpected={len(unexpected)}；"
                f"前若干项 missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        self.model.eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.dim = 768

    @property
    def frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.parameters())

    @torch.no_grad()
    def forward(
        self,
        s2: torch.Tensor,
        s2_dates: torch.Tensor,
        patch_size_m: int,
    ) -> dict:
        """Args:
            s2: ``[B,1,10,H,W]`` 已按官方统计标准化。
            s2_dates: ``[B,1]`` 一年内日序（0=1/1）。
            patch_size_m: 单次前向使用的物理 patch 尺寸（米）。

        Returns:
            dict with ``feature`` ``[B,2D,h,w]``, ``stride_m``, ``info``.
        """
        import time

        t0 = time.perf_counter()
        feats = self.model(
            {"s2": s2, "s2_dates": s2_dates},
            patch_size=patch_size_m,
            output="dense",
            output_modality="s2",
        )
        dt = time.perf_counter() - t0
        # 官方 dense 输出是 NHWC（见 Any_multi.forward_release 最后的 permute），
        # 统一转成 NCHW 供后续卷积使用；out_shape 记录的是官方原始形状。
        official_shape = tuple(feats.shape)
        # dense 输出是两段拼接：[B,H,W, D(上下文, 每个 patch 恒定) | D(局部, 10 m)]
        # 上下文段若当成 10 m 特征用，必然在 patch 边界形成台阶（实测块内 std = 0）。
        split = None
        if feats.shape[-1] == 2 * int(self.dim):
            feats_nhwc = feats
            split = {
                "context": feats_nhwc[..., : int(self.dim)].permute(0, 3, 1, 2).contiguous(),
                "local": feats_nhwc[..., int(self.dim):].permute(0, 3, 1, 2).contiguous(),
            }
        if feats.ndim == 4 and feats.shape[-1] > feats.shape[1]:
            feats = feats.permute(0, 3, 1, 2).contiguous()
        # patch_size 单位是米，输入 10m/px：每边 patch 数 = 图块边长(米) / patch_size(米)
        tile_side_m = s2.shape[-1] * 10.0
        per_side = int(round(tile_side_m / patch_size_m))
        out = {
            "feature": feats,
            "stride_m": 10.0,   # AnySat 对 s2 的 dense 输出固定在原生 10m 网格
            "info": AnySatRunInfo(
                patch_size_m=patch_size_m,
                out_shape=official_shape,
                latency_s=dt,
                num_patches=per_side * per_side,
            ),
        }
        if split is not None:
            out["split"] = split
        return out

    def describe(self) -> dict:
        return {
            "name": "AnySat-base",
            "backend": "gastruc/AnySat hubconf (official)",
            "dim": self.dim,
            "dense_channels": 2 * self.dim,
            "patch_sizes_m": self.patch_sizes_m,
            "frozen": self.frozen,
            "params": sum(p.numel() for p in self.model.parameters()),
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

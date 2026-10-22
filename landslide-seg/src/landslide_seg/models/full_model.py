"""完整模型：TerraMind(TiM) + AnySat(多尺度) + 语义三阶段 + 粗细两阶段分割。

第一版就是完整架构（Landslide-Seg V1 §2），不允许为了跑通而删除任一支路。
两支路默认冻结，可训练参数集中在融合、语义适配、粗定位与精细化四个模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.band_mapping import normalize_anysat
from ..encoders.anysat_scale import AnySatScaleAdapter
from ..encoders.terramind_tim import TerraMindTiMAdapter, normalize_s2l2a_dn
from .coarse import CoarseLocator
from .fusion import FeatureFusion, FeatureMap
from .refiner import FineRefiner
from .semantic import SemanticAdapter


@dataclass
class ForwardOutput:
    fine_logits: torch.Tensor
    coarse_logits: torch.Tensor
    presence_logits: torch.Tensor
    sem_map: torch.Tensor
    region_embedding: torch.Tensor | None = None
    region_exists: torch.Tensor | None = None
    features: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return getattr(self, key)


class LandslideSegV1(nn.Module):
    """完整 V1 模型。

    Args:
        terramind: 已构建的 TerraMindTiM 适配器（可为 None 以做 −TiM 消融）
        anysat: 已构建的 AnySat 适配器（可为 None 以做 −AnySat 消融）
        semantic: 语义适配器（冻结文本塔 + 可训练投影）
        channels: 融合通道数
        level_strides: 金字塔目标 stride（米）
        anysat_scales_m: 参与融合的 AnySat 物理尺度
    """

    def __init__(
        self,
        terramind: TerraMindTiMAdapter,
        anysat: AnySatScaleAdapter,
        semantic: SemanticAdapter,
        channels: int = 128,
        level_strides: Sequence[float] = (10, 20, 40, 80),
        anysat_scales_m: Sequence[int] = (40, 160),
        branch_dropout: float = 0.1,
    ):
        super().__init__()
        self.terramind = terramind
        self.anysat = anysat
        self.semantic = semantic
        self.anysat_scales_m = tuple(anysat_scales_m)
        self.level_strides = tuple(float(s) for s in level_strides)

        sources = []
        if terramind is not None:
            sources.append("terramind.S2L2A")
        if anysat is not None:
            sources.extend([f"anysat.s2@{p}m" for p in self.anysat_scales_m])

        self.fusion = FeatureFusion(
            channels=channels,
            level_strides=self.level_strides,
            sources=sources,
            branch_dropout=branch_dropout,
        )
        self.coarse = CoarseLocator(channels=channels, hidden=64, n_levels=len(self.level_strides))
        self.refiner = FineRefiner(channels=channels, hidden=96, n_levels=len(self.level_strides))

        # 预建融合投影：通道数由各骨干官方输出维度决定，避免 forward 时才建层
        # 导致新建的层留在 CPU（`.to(device)` 之后再建会漏掉设备迁移）。
        expected: dict[str, int] = {}
        if terramind is not None:
            expected["terramind.S2L2A"] = int(terramind.dim)
        if anysat is not None:
            split = bool(getattr(anysat, "split_heads", False))
            for p in self.anysat_scales_m:
                if split:
                    expected[f"anysat.s2@{p}m.context"] = int(anysat.dim)
                    expected[f"anysat.s2@{p}m.local"] = int(anysat.dim)
                else:
                    expected[f"anysat.s2@{p}m"] = int(anysat.dim) * 2
        self.fusion.build(expected)
        self._expected_channels = expected

    # ------------------------------------------------------------------
    def _build_fusion(self, channel_map: dict[str, int]) -> None:
        """校验实际通道数与预建结构一致（不一致说明骨干实现变了）。"""
        for k, v in channel_map.items():
            if k in self._expected_channels and self._expected_channels[k] != v:
                raise RuntimeError(
                    f"融合投影通道不匹配: {k} 预建 {self._expected_channels[k]} 实际 {v}"
                )
        self._fusion_built = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        groups = {}
        for name, mod in (
            ("terramind", self.terramind),
            ("anysat", self.anysat),
            ("fusion", self.fusion),
            ("semantic", self.semantic),
            ("coarse", self.coarse),
            ("refiner", self.refiner),
        ):
            if mod is None:
                continue
            n = sum(p.numel() for p in mod.parameters())
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            groups[name] = {"params": n, "trainable": t}
        return {"total": total, "trainable": train, "groups": groups}

    # ------------------------------------------------------------------
    def forward(
        self,
        x12: torch.Tensor,
        x10: torch.Tensor,
        s2_dates: torch.Tensor,
        *,
        tim: bool | None = None,
        region_mask: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        return_features: bool = True,
    ) -> ForwardOutput:
        """Args:
            x12: ``[B,12,H,W]`` **DN 尺度**的完整 S2L2A（未标准化）。
            x10: ``[B,10,H,W]`` **DN 尺度**的 AnySat 波段（未标准化）。
            s2_dates: ``[B,1]`` 一年内日序。
            tim: 覆盖 TiM 开关，None 表示用实例默认。
            region_mask: ``[B,H,W]`` 训练时传入真值区域，用于区域-文本监督；
                **推理时不得传入**（本模型也不从它派生任何提示）。
        """
        b, _, h, w = x12.shape

        features: list[FeatureMap] = []
        channel_map: dict[str, int] = {}
        extras: dict = {}

        if self.terramind is not None:
            tm_out = self.terramind(normalize_s2l2a_dn(x12), tim=tim)
            tokens = tm_out["tokens"][-1]
            tm_map = self.terramind.tokens_to_map(tokens)
            grid_h, grid_w = tm_out["grid"]
            stride_m = h * 10.0 / grid_h
            features.append(
                FeatureMap(tm_map, "terramind.S2L2A", stride_m=stride_m, frozen=True)
            )
            channel_map["terramind.S2L2A"] = tm_map.shape[1]
            extras["terramind_info"] = tm_out["info"]

        if self.anysat is not None:
            s2_norm = normalize_anysat(x10)
            for p in self.anysat_scales_m:
                out = self.anysat(s2_norm.unsqueeze(1), s2_dates, patch_size_m=p)
                if "split" in out:
                    # 上下文段：每个 patch 一个向量，按 patch 的物理尺寸放（stride = p 米）
                    fc = out["split"]["context"]
                    features.append(FeatureMap(fc, f"anysat.s2@{p}m.context",
                                               stride_m=float(p), frozen=True))
                    channel_map[f"anysat.s2@{p}m.context"] = fc.shape[1]
                    # 局部段：真正的 10 m 特征
                    fl = out["split"]["local"]
                    features.append(FeatureMap(fl, f"anysat.s2@{p}m.local",
                                               stride_m=10.0, frozen=True))
                    channel_map[f"anysat.s2@{p}m.local"] = fl.shape[1]
                else:
                    f = out["feature"]
                    features.append(
                        FeatureMap(f, f"anysat.s2@{p}m", stride_m=out["stride_m"], frozen=True)
                    )
                    channel_map[f"anysat.s2@{p}m"] = f.shape[1]
                extras[f"anysat_info@{p}m"] = out["info"]

        if not getattr(self, "_fusion_built", False):
            self._build_fusion(channel_map)

        pyramid = self.fusion(features, target_size=(h, w))

        # 语义是"两遍"的：先用 FPN 自己的粗 logits 得到**预测**候选区域（不碰 GT），
        # 再据此算区域-背景关系，组合出该样本独有的语义向量，最后回补到粗 logits。
        raw = self.coarse(pyramid, None)
        region_soft = torch.sigmoid(raw["coarse_logits"])
        v = valid.float() if valid is not None else torch.ones_like(region_soft)
        if v.ndim == 3:
            v = v.unsqueeze(1)
        sem = self.semantic(pyramid[0], region_soft, v, spectral=x10)
        sem_map = sem.sem_map
        coarse_logits = raw["coarse_logits"]
        if self.sem_use_loc:
            coarse_logits = coarse_logits + self.coarse.semantic_scale * sem_map
        coarse = {**raw, "coarse_logits": coarse_logits}
        coarse_prob = torch.sigmoid(coarse_logits)
        fine = self.refiner(
            pyramid, coarse_prob,
            sem_map=sem_map if self.sem_use_ref else None,
            film=sem.film if self.sem_use_ref else None,
        )

        region_emb = None
        region_exists = None
        if region_mask is not None and self.sem_use_align:
            # 前景/背景在**同一张图**内对比，避免"同类别即负例"的错误
            rm = (region_mask > 0).float()
            if valid is not None:
                v = valid.float()
                if v.ndim == 3:          # [B,H,W] -> [B,1,H,W]，否则会广播成 [B,B,H,W]
                    v = v.unsqueeze(1)
                fg_mask = rm * v
                bg_mask = (1.0 - rm) * v
            else:
                fg_mask = rm
                bg_mask = 1.0 - rm
            fg = self.semantic.region_embedding(pyramid[0], fg_mask)
            bg = self.semantic.region_embedding(pyramid[0], bg_mask)
            region_emb = torch.stack([fg, bg], dim=1)                  # [B,2,D]
            region_exists = torch.stack(
                [
                    fg_mask.flatten(1).sum(1) > 0,
                    bg_mask.flatten(1).sum(1) > 0,
                ],
                dim=1,
            )                                                          # [B,2] bool

        return ForwardOutput(
            fine_logits=fine["fine_logits"],
            coarse_logits=coarse["coarse_logits"],
            presence_logits=coarse["presence_logits"],
            sem_map=sem_map,
            region_embedding=region_emb,
            region_exists=region_exists,
            features={
                "pyramid": [p.detach() if not p.requires_grad else p for p in pyramid],
                "channels": channel_map,
                "extras": extras,
            } if return_features else {},
        )

    @torch.no_grad()
    def predict(
        self,
        x12: torch.Tensor,
        x10: torch.Tensor,
        s2_dates: torch.Tensor,
        threshold: float = 0.5,
        empty_threshold: float = 0.5,
        use_presence_gate: bool = True,
        valid: torch.Tensor | None = None,
    ) -> dict:
        """推理入口：只吃单期观测，**不使用任何 GT 信息**。

        Args:
            use_presence_gate: 是否用 presence 分数做"空图拒检"。
                若 presence 头没有可用负样本训练过（本数据集即如此），
                这个门控是没有依据的，应显式关掉，否则会用随机分数去清零预测。
        """
        self.eval()
        out = self.forward(x12, x10, s2_dates, return_features=False)
        prob = torch.sigmoid(out.fine_logits)
        presence = torch.sigmoid(out.presence_logits)
        if valid is not None:
            v = valid.float().view(-1, 1, 1, 1)
            prob = prob * v           # 无效像素不输出概率
        mask = (prob > threshold).float()
        if use_presence_gate:
            mask = mask * (presence > empty_threshold).float().view(-1, 1, 1, 1)
        return {
            "prob": prob,
            "mask": mask,
            "presence": presence,
            "coarse_prob": torch.sigmoid(out.coarse_logits),
        }

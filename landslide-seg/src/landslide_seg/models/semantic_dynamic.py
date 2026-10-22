"""动态语义：共享类别概念 + 影像条件属性 + 区域-背景关系。

与旧实现的区别（旧版是错的）：
- 旧版：一条固定英文短语 → 一个固定 512 维原型 → 全图一个相似度图。
  它对每张图完全相同，不含任何影像信息，本质是"文本锚点"，不是语义。
- 新版：
  1. **类别语义**：稳定的类别查询（文本编码，冻结），作为注意力的 query 去聚合视觉特征；
  2. **属性语义**：由一个属性头从视觉特征预测 K 个属性的响应（每张图不同），
     再用这些响应把属性词向量加权组合进语义向量——这是"影像条件化"的关键；
  3. **关系语义**：候选区域（用**预测**的粗概率，不用 GT）与周边背景在光谱、纹理、
     空间形态上的对比，投影后并入语义向量。

属性监督：
- 标注值 ∈ {0, 1}；**缺失 = 未知**，用 -1 表示，不参与损失（绝不是"否定"）。
- 本数据集没有任何属性标注，因此训练时 `L_attribute` 不生效（applied=False），
  但推理时属性响应仍然由影像预测产生——这正是设计要求。

10 m 分辨率无法辨认的细节（如岩性、滑动面深度、建筑层数）不进入属性词表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: 10 m Sentinel-2 上可观测的属性词表（只放"看得见"的）
ATTRIBUTE_VOCAB: dict[str, str] = {
    "exposed_soil": "exposed soil on a slope",
    "rocky_scar": "rocky landslide scar",
    "veg_disturbance": "disturbed or broken vegetation",
    "debris_texture": "debris flow texture with mixed deposits",
    "water_wet_ground": "standing water or wet ground",
    "dense_forest": "dense forest canopy",
    "sparse_vegetation": "sparse vegetation or shrubland",
    "bare_rock": "bare rock or cliff",
    "cropland_flat": "cropland or flat field",
    "shadow_or_cloud": "terrain shadow or cloud",
}

#: 推理时不得使用的属性（10 m 分辨不出来），显式记录以便审阅
NON_OBSERVABLE = ("lithology", "slip_surface_depth", "building_floors", "soil_moisture_pct")


@dataclass
class SemanticOutput:
    dyn: torch.Tensor            # [B,D] 当前样本的动态语义向量
    sem_map: torch.Tensor        # [B,1,H,W] 相似度图（logit 尺度）
    attr_logits: torch.Tensor    # [B,K] 属性响应（未过 sigmoid）
    relation: torch.Tensor       # [B,R] 关系特征（光谱/纹理/形态对比）
    film: tuple[torch.Tensor, torch.Tensor]


class DynamicSemantic(nn.Module):
    """把"类别查询 + 属性响应 + 关系对比"组合成每样本独有的语义向量。"""

    def __init__(
        self,
        channels: int,
        embed_dim: int = 512,
        class_prototypes: torch.Tensor | None = None,
        attribute_embeddings: torch.Tensor | None = None,
        attribute_names: Sequence[str] | None = None,
        mode: str = "full",
        use_attributes: bool = True,
        use_relation: bool = True,
    ):
        super().__init__()
        assert mode in ("class_only", "full")
        self.mode = mode
        self.use_attributes = use_attributes and mode == "full"
        self.use_relation = use_relation and mode == "full"
        self.channels = channels
        self.embed_dim = embed_dim
        names = list(attribute_names or ATTRIBUTE_VOCAB)
        self.attribute_names = names
        k = len(names)

        # 类别查询（冻结文本编码，[1,D]）
        if class_prototypes is None:
            class_prototypes = torch.zeros(1, embed_dim)
        self.register_buffer("class_query", F.normalize(class_prototypes.float(), dim=-1))
        # 属性词向量（冻结文本编码，[K,D]）
        if attribute_embeddings is None:
            attribute_embeddings = torch.zeros(k, embed_dim)
        self.register_buffer("attr_emb", F.normalize(attribute_embeddings.float(), dim=-1))

        # 视觉侧：1×1 投影到 embed_dim
        self.vis_proj = nn.Sequential(
            nn.Conv2d(channels, embed_dim, 1, bias=False),
            nn.GroupNorm(32, embed_dim, eps=1e-5),
        )
        # 属性头：从整图池化特征预测 K 个属性响应（影像条件化）
        self.attr_head = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, k),
        )
        # 关系头： FG-BG 对比特征 -> 语义空间
        # 输入 = 视觉对比(C) + 光谱对比(10) + 纹理对比(1) + 形态(2)
        self.rel_dim = channels + 10 + 3
        self.rel_proj = nn.Sequential(
            nn.Linear(self.rel_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # FiLM：从动态语义生成调制参数
        self.film = nn.Sequential(nn.Linear(embed_dim, 2 * channels), nn.GELU())
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _masked_mean(feat: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """feat[B,C,H,W], w[B,1,H,W] -> [B,C]；权重和为 0 时返回 0。"""
        num = (feat * w).sum(dim=(2, 3))
        den = w.sum(dim=(2, 3)).clamp_min(1e-6)
        return num / den

    def forward(
        self,
        p0: torch.Tensor,               # [B,C,H,W] 融合金字塔最细层
        region_soft: torch.Tensor,      # [B,1,H,W] **预测**的前景概率（不是 GT）
        valid: torch.Tensor,            # [B,1,H,W] 有效像素
        spectral: torch.Tensor,         # [B,10,H,W] 原始/标准化后的波段（用于光谱对比）
    ) -> SemanticOutput:
        b, c, h, w = p0.shape
        vis = self.vis_proj(p0)                       # [B,D,H,W]

        # ---- 属性响应：**按候选区域池化**，不是整图 ----
        # 属性（裸土/植被扰动…）描述的是滑坡区域本身；整图池化会被背景压倒，
        # 导致所有样本都预测成"密林/稀疏植被"。前景为空时退回整图。
        vv = valid.float()
        fg_w0 = region_soft * vv
        area_fg = fg_w0.flatten(1).sum(1, keepdim=True)
        pooled_fg = self._masked_mean(p0, fg_w0)
        pooled_all = self._masked_mean(p0, vv)
        use_fg = (area_fg > 8.0).float()               # 至少 8 个像素才认为有候选区
        pooled = (pooled_fg * use_fg + pooled_all * (1 - use_fg)).view(b, c)
        attr_logits = self.attr_head(pooled)          # [B,K]
        attr_alpha = torch.sigmoid(attr_logits)       # [B,K]

        if self.mode == "class_only":
            dyn = self.class_query.expand(b, -1)
            relation = torch.zeros(b, self.rel_dim, device=p0.device)
        else:
            # ---- 动态语义 = 类别查询 + Σ α_k·属性词向量 ----
            if self.use_attributes:
                combo = (attr_alpha.unsqueeze(-1) * self.attr_emb.unsqueeze(0)).sum(1)  # [B,D]
                dyn = self.class_query + combo
            else:
                dyn = self.class_query.expand(b, -1)

            # ---- 关系： FG vs BG（用预测前景，不用 GT） ----
            fg_w = region_soft * valid
            bg_w = (1.0 - region_soft) * valid
            # 视觉对比用 C 维的 P0（不是 512 维投影后特征），保证 rel_dim 与定义一致
            fg_vis = self._masked_mean(p0, fg_w)
            bg_vis = self._masked_mean(p0, bg_w)
            vis_diff = fg_vis - bg_vis

            fg_spec = self._masked_mean(spectral, fg_w)
            bg_spec = self._masked_mean(spectral, bg_w)
            spec_diff = fg_spec - bg_spec

            # 纹理对比：梯度幅值在 FG 与 BG 上的差
            p0g = p0.abs().mean(1, keepdim=True)
            g = p0g.diff(dim=3, prepend=p0g[..., :1])
            tex_fg = self._masked_mean(g, fg_w).mean(-1, keepdim=True)
            tex_bg = self._masked_mean(g, bg_w).mean(-1, keepdim=True)
            # 形态：面积占比 + 前景概率的离散度
            area = fg_w.mean(dim=(2, 3))
            spread = region_soft.flatten(1).std(1, keepdim=True)

            # 各块的量级差了几个数量级（视觉 ~0.1、光谱 DN ~10^3、形态 ~10^-2），
            # 直接 concat 会让光谱项独占关系向量。这里按块做 L2 归一化，使各块可比。
            def _l2n(x: torch.Tensor) -> torch.Tensor:
                return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

            relation = torch.cat(
                [
                    _l2n(vis_diff),
                    _l2n(spec_diff),
                    _l2n(tex_fg - tex_bg),
                    _l2n(area),
                    _l2n(spread),
                ],
                dim=1,
            )
            if self.use_relation:
                dyn = dyn + self.rel_proj(relation)

        dyn = F.normalize(dyn, dim=-1)
        # 真正的余弦相似度：视觉侧也要归一化，否则 sem_map 的量级会随通道方差漂移
        sem_map = (
            F.normalize(vis, dim=1) * dyn.view(b, -1, 1, 1)
        ).sum(1, keepdim=True) * self.logit_scale
        gamma, beta = self.film(dyn).chunk(2, dim=1)
        gamma, beta = gamma.contiguous(), beta.contiguous()
        return SemanticOutput(dyn=dyn, sem_map=sem_map, attr_logits=attr_logits,
                              relation=relation, film=(gamma, beta))

    # ------------------------------------------------------------------
    @property
    def class_prototype(self) -> torch.Tensor:
        """兼容旧接口：区域-文本损失仍以类别查询为正例锚点。"""
        return self.class_query

    @property
    def attributes_active(self) -> bool:
        """属性**监督**是否生效（由是否有标注决定，与是否预测无关）。"""
        return False

    def region_embedding(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """区域池化到语义空间：[B,C,H,W] + [B,1,H,W] -> [B,D]（已归一化）。"""
        vis = self.vis_proj(feature)
        pooled = self._masked_mean(vis, mask.float())
        return F.normalize(pooled, dim=-1)

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "embed_dim": self.embed_dim,
            "attributes": list(self.attribute_names),
            "n_attributes": len(self.attribute_names),
            "non_observable_excluded": list(NON_OBSERVABLE),
            "class_query_shape": tuple(self.class_query.shape),
            "attr_emb_shape": tuple(self.attr_emb.shape),
            "attributes_supervised": False,
        }

    def attribute_loss(
        self, attr_logits: torch.Tensor, attr_target: torch.Tensor | None
    ) -> tuple[torch.Tensor, bool]:
        """属性监督：target ∈ {0,1,-1}，-1 表示未知，**不参与损失**。"""
        if attr_target is None:
            return torch.zeros((), device=attr_logits.device), False
        t = attr_target.float()
        known = t >= 0
        if known.sum() == 0:
            return torch.zeros((), device=attr_logits.device), False
        loss = F.binary_cross_entropy_with_logits(attr_logits[known], t[known])
        return loss, True

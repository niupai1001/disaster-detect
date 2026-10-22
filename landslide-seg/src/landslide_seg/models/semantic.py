"""语义模块：冻结文本编码器 + 类别/属性原型 + 三种用途的接口。

语义在本模型中**真实参与三处**（Landslide-Seg V1 §4.4）：
1. 表征监督：``region_embedding()`` 提供区域特征，用于和文本原型做对比损失；
2. 粗定位调制：``similarity_map()`` 产出相似度热图，进入 ``CoarseLocator``；
3. 精细分割调制：``film_params()`` 产出 FiLM 参数，调制 ``FineRefiner`` 的特征。

推理时**不依赖 GT Mask**：类别原型是固定的文本查询，相似度图由视觉特征自行计算。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CLASS_PROMPTS = (
    "a landslide scar in a satellite image",
)

DEFAULT_ATTRIBUTE_PROMPTS = (
    "exposed soil on a slope",
    "rocky landslide scar",
    "vegetation disturbance",
    "forest background",
    "surrounding vegetation contrast",
)


class RemoteCLIPTextEncoder(nn.Module):
    """冻结的 RemoteCLIP 文本塔。

    使用官方 ``RemoteCLIP-ViT-B-32.pt``（openai CLIP 格式），
    通过 ``clip`` 包加载；权重缺失或加载失败一律抛错，**不允许随机初始化冒充**。
    """

    def __init__(self, ckpt_path: str | os.PathLike):
        super().__init__()
        ckpt = Path(ckpt_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"RemoteCLIP 权重不存在: {ckpt}。"
                "请从 https://huggingface.co/chendelong/RemoteCLIP 下载 RemoteCLIP-ViT-B-32.pt"
            )
        try:
            import clip  # openai CLIP（可通过 clip-anytorch 安装）
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 openai CLIP 来加载 RemoteCLIP 权重：pip install clip-anytorch"
            ) from exc

        model, _ = clip.load(str(ckpt), device="cpu", jit=False)
        # 注意：CLIP.dtype 是读取 self.visual.dtype 的属性，不能把 visual 置空，
        # 因此这里保留完整 checkpoint（视觉塔冻结但不用）。
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = int(self.model.text_projection.shape[1])
        self._clip = clip

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        """返回 ``[K, D]`` 的 L2 归一化文本嵌入。"""
        tokens = self._clip.tokenize(list(prompts), truncate=True)
        emb = self.model.encode_text(tokens)
        return F.normalize(emb.float(), dim=-1)


class SemanticAdapter(nn.Module):
    """把冻结文本原型接到视觉特征上的可训练适配器。

    Args:
        channels: 视觉特征通道 C。
        embed_dim: 文本嵌入维度（RemoteCLIP ViT-B-32 为 512）。
        text_encoder: 已加载的冻结文本编码器。
        class_prompts / attribute_prompts: 文本提示。
        use_attributes: 是否启用属性原型（无属性标注时仍可计算，但训练信号为 0）。
    """

    def __init__(
        self,
        text_encoder: RemoteCLIPTextEncoder,
        channels: int = 128,
        class_prompts: Sequence[str] = DEFAULT_CLASS_PROMPTS,
        attribute_prompts: Sequence[str] = DEFAULT_ATTRIBUTE_PROMPTS,
        use_attributes: bool = True,
    ):
        super().__init__()
        self.text = text_encoder
        self.channels = channels
        self.embed_dim = text_encoder.embed_dim
        self.use_attributes = use_attributes

        cls_emb = text_encoder.encode(class_prompts)          # [1, D]
        self.register_buffer("class_prototype", cls_emb.mean(0, keepdim=True))
        if use_attributes:
            attr_emb = text_encoder.encode(attribute_prompts)  # [K, D]
            self.register_buffer("attribute_prototypes", attr_emb)
        else:
            self.register_buffer("attribute_prototypes", torch.zeros(0, self.embed_dim))
        # 属性原型目前**只计算、不参与任何前向或损失**：
        # 没有经过核实的属性标注，按规范不能凭空构造属性监督（§4.4）。
        # 这三条路径中真正生效的只有 class_prototype。
        self.attributes_active = False

        # 视觉侧投影：C -> D，用于相似度与区域-文本对齐。
        # 用 GroupNorm 而不是 BatchNorm：区域池化后是 [B,D,1,1]，
        # 单样本（batch=1，1280m 图块的常见情形）时 BN 会直接报错。
        self.vis_proj = nn.Sequential(
            nn.Conv2d(channels, self.embed_dim, 1, bias=False),
            nn.GroupNorm(32, self.embed_dim, eps=1e-5),
        )
        # 区域池化后的投影（与 vis_proj 共享语义空间）
        self.region_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        # 文本原型 -> FiLM(gamma,beta)
        self.film = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, 2 * channels),
        )
        # 相似度 logit 温度（可学习，初始 1/0.07 的倒数形式）
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    # ------------------------------------------------------------------
    def similarity_map(self, finest: torch.Tensor) -> torch.Tensor:
        """``[B,C,H,W]`` -> ``[B,1,H,W]`` 类别相似度（logit 尺度）。"""
        v = F.normalize(self.vis_proj(finest), dim=1)
        p = F.normalize(self.class_prototype, dim=-1).view(1, -1, 1, 1)
        sim = (v * p).sum(dim=1, keepdim=True)
        return sim * self.logit_scale

    def film_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """由类别原型生成 FiLM 参数 ``gamma/beta``，形状 ``[1,2C]`` 各一半。"""
        p = F.normalize(self.class_prototype, dim=-1)
        out = self.film(p).view(1, -1)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta

    def region_embedding(
        self,
        feature: torch.Tensor,
        region_mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """按区域掩膜池化视觉特征并投影到文本空间。

        Args:
            feature: ``[B,C,H,W]`` 融合特征。
            region_mask: ``[B,1,H,W]`` 或 ``[B,H,W]``，取值 {0,1}。

        Returns:
            ``[B,D]`` L2 归一化的区域嵌入；空区域返回零向量。
        """
        if region_mask.ndim == 3:
            region_mask = region_mask.unsqueeze(1)
        if region_mask.shape[-2:] != feature.shape[-2:]:
            region_mask = F.interpolate(
                region_mask.float(), size=feature.shape[-2:], mode="nearest"
            )
        m = region_mask.float()
        area = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        pooled = (feature * m).sum(dim=(2, 3), keepdim=True) / area   # [B,C,1,1]
        v = self.vis_proj(pooled).flatten(1)                          # [B,D]
        v = self.region_proj(v)
        norm = v.norm(dim=-1, keepdim=True)
        return torch.where(norm > eps, v / norm.clamp_min(eps), torch.zeros_like(v))

    def describe(self) -> dict:
        return {
            "text_backbone": "RemoteCLIP ViT-B-32 (frozen)",
            "embed_dim": self.embed_dim,
            "class_prompt": DEFAULT_CLASS_PROMPTS[0],
            "class_prototype": tuple(self.class_prototype.shape),
            "attribute_prototypes": tuple(self.attribute_prototypes.shape),
            "uses_attributes": self.use_attributes,
            "attributes_active": self.attributes_active,
            "note": "文本来自代码内固定提示词，非用户输入；属性分支未激活，"
                    "只有类别原型真正参与前向与损失",
        }

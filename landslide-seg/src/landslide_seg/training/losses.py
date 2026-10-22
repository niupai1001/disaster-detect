"""统一损失。

``L = L_fine + λc·L_coarse + λp·L_presence + λs·L_region_text + λa·L_attribute``

设计约束（Landslide-Seg V1 §6）：
- 分割项用 ``valid`` 掩膜排除云遮挡/无效像素；
- 区域-文本监督**不能**把"另一处滑坡"当作负例，因此采用"区域 vs 类别原型"的
  二分类匹配目标，而不是 in-batch 对比；
- 属性损失只在有属性标注时非零，缺失时返回 0 并标记 ``applied=False``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


def _reduce(t: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return t
    v = valid.float()
    return (t * v).sum() / v.sum().clamp_min(1.0)


def bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """``pos_weight`` 与官方一致：正类项的权重。None/1.0 等价于普通 BCE。

    ``F.binary_cross_entropy_with_logits`` 的 ``pos_weight`` 只作用于正类项
    （``-[p·y·logσ(z) + (1-y)·log(1-σ(z))]``），因此 p>1 会把模型推向"多报"。
    """
    pw = None
    if pos_weight is not None and float(pos_weight) != 1.0:
        pw = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none", pos_weight=pw
    )
    return _reduce(loss, valid)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    if valid is not None:
        v = valid.float()
        prob = prob * v
        target = target.float() * v
    else:
        target = target.float()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


@dataclass
class LossOutput:
    total: torch.Tensor
    parts: Dict[str, torch.Tensor]
    applied: Dict[str, bool]

    def as_dict(self) -> dict:
        d = {k: float(v.detach()) for k, v in self.parts.items()}
        d["total"] = float(self.total.detach())
        d.update({f"applied_{k}": v for k, v in self.applied.items()})
        return d


class CombinedLoss(torch.nn.Module):
    """组合损失，权重全部显式可配。"""

    def __init__(
        self,
        lambda_coarse: float = 0.5,
        lambda_presence: float = 0.2,
        lambda_region_text: float = 0.1,
        lambda_attribute: float = 0.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.lambda_coarse = lambda_coarse
        self.lambda_presence = lambda_presence
        self.lambda_region_text = lambda_region_text
        self.lambda_attribute = lambda_attribute
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def seg_loss(self, logits, target, valid):
        return self.bce_weight * bce_loss(logits, target, valid) + self.dice_weight * dice_loss(
            logits, target, valid
        )

    def forward(
        self,
        *,
        fine_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None,
        region_embedding: torch.Tensor | None = None,
        class_prototype: torch.Tensor | None = None,
        region_label: torch.Tensor | None = None,
        region_exists: torch.Tensor | None = None,
        attribute_logits: torch.Tensor | None = None,
        attribute_target: torch.Tensor | None = None,
    ) -> LossOutput:
        parts: Dict[str, torch.Tensor] = {}
        applied: Dict[str, bool] = {}

        parts["fine"] = self.seg_loss(fine_logits, target, valid)
        applied["fine"] = True

        parts["coarse"] = self.seg_loss(coarse_logits, target, valid)
        applied["coarse"] = True

        # presence：图块内是否存在滑坡（target 形状 [B,1,H,W]，只在 valid 内判断）
        dims = tuple(range(1, target.ndim))
        tgt = target.float()
        if valid is not None:
            tgt = tgt * valid.float()
        has_ls = (tgt.sum(dim=dims) > 0).float().view(-1, 1)
        parts["presence"] = F.binary_cross_entropy_with_logits(presence_logits, has_ls)
        applied["presence"] = True

        # 区域-文本：前景 vs 背景的**同图对比**
        # region_embedding: [B,R,D]，R=2（0=前景, 1=背景）；region_exists: [B,R] bool
        if (
            region_embedding is not None
            and class_prototype is not None
            and region_label is not None
            and region_embedding.numel() > 0
        ):
            proto = F.normalize(class_prototype.float(), dim=-1)
            emb = F.normalize(region_embedding.float(), dim=-1)   # [B,R,D]
            sim = (emb * proto).sum(dim=-1) * 10.0                 # [B,R]
            labels = region_label.float().view(1, -1).expand_as(sim)
            if region_exists is not None:
                keep = region_exists.bool()
                sim = sim[keep]
                labels = labels[keep]
            if sim.numel() > 0:
                parts["region_text"] = F.binary_cross_entropy_with_logits(sim, labels)
                applied["region_text"] = True
            else:
                parts["region_text"] = torch.zeros((), device=fine_logits.device)
                applied["region_text"] = False
        else:
            parts["region_text"] = torch.zeros((), device=fine_logits.device)
            applied["region_text"] = False

        # 属性语义：仅在有可核实标注时启用
        if attribute_logits is not None and attribute_target is not None:
            parts["attribute"] = F.binary_cross_entropy_with_logits(
                attribute_logits, attribute_target.float()
            )
            applied["attribute"] = True
        else:
            parts["attribute"] = torch.zeros((), device=fine_logits.device)
            applied["attribute"] = False

        total = (
            parts["fine"]
            + self.lambda_coarse * parts["coarse"]
            + self.lambda_presence * parts["presence"]
            + self.lambda_region_text * parts["region_text"]
            + self.lambda_attribute * parts["attribute"]
        )
        return LossOutput(total=total, parts=parts, applied=applied)

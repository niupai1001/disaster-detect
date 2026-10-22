"""按位置建模波段关系：把 ``BandRelationBlock`` 接到十波段入口之前。

数据流（与既有的光谱关系实现一致）：

    x10（z-score）
      └─► BandRelationBlock：4×4 小块内 10 个波段 token 互相注意；
            关系偏置由该小块的波段响应差与波长差算出（因此随位置变化）；
            读出后残差加回输入
      └─► SegFormer B0（主路）与细节分支（两路都读这份修正后的输入）

这里的"位置"指 4×4 小块的位置：同一波段对在不同小块的关系权重可以不同。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .spectral_temporal import BandRelationBlock


class SpectralRelationSeg(nn.Module):
    def __init__(self, visual: nn.Module, stats: dict, *, patch: int = 4, width: int = 64,
                 heads: int = 4, layers: int = 2, relation_mode: str = "measured",
                 identity: str = "wavelength") -> None:
        super().__init__()
        self.visual = visual
        self.relation = BandRelationBlock(patch=patch, width=width, heads=heads, layers=layers,
                                          stats=stats, relation_mode=relation_mode,
                                          identity=identity)
        self.patch, self.width, self.heads, self.layers = int(patch), int(width), int(heads), int(layers)
        self.relation_mode, self.identity = relation_mode, identity
        for attr in ("adapter_type", "backbone", "use_stem", "refine", "fusion_mode",
                     "exchange_mode", "adapter_report", "load_info"):
            if hasattr(visual, attr):
                setattr(self, attr, getattr(visual, attr))

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        if x.ndim != 4 or x.shape[1] != 10:
            raise ValueError(f"期望 [B,10,H,W]，得到 {tuple(x.shape)}")
        return self.visual(self.relation(x), **kwargs)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        enc, dec, new = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if ".segformer." in name:
                enc.append(p)
            elif ".decode_head." in name:
                dec.append(p)
            else:
                new.append(p)
        return {"encoder": enc, "decoder": dec, "new": new}

    def count_params(self) -> dict:
        g = self.param_groups()
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "encoder": sum(p.numel() for p in g["encoder"]),
                "decoder": sum(p.numel() for p in g["decoder"]),
                "new": sum(p.numel() for p in g["new"]),
                "relation_block": sum(p.numel() for p in self.relation.parameters())}

    def spectral_report(self) -> dict:
        return {"module": "BandRelationBlock", "patch": self.patch, "width": self.width,
                "heads": self.heads, "layers": self.layers,
                "relation_mode": self.relation_mode, "identity": self.identity,
                "location": "十波段入口之前，残差加回后同时进入主路与细节分支"}

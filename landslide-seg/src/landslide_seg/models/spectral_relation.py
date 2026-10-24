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


class BandRelationFeature(nn.Module):
    """把同一关系编码器的 token 表示输出成 ``F_spec``（不再重构十波段）。

    输入 ``[B,10,H,W]``，4×4 小块 → 每块 10 个 band token → 自注意力 →
    10 个 token 取均值 → ``[B,width,H/p,W/p]`` → 1×1 投影到解码特征宽度。
    投影正常初始化；S4 模型的 ``alpha`` 初值为 0，因此融合支路初始零贡献。
    不能同时把投影与 alpha 都置零：两者都为 0 时任务梯度到不了任何一侧。
    """

    def __init__(self, *, out_ch: int, patch: int = 4, width: int = 64, heads: int = 4,
                 layers: int = 2, stats: dict, relation_mode: str = "measured",
                 identity: str = "wavelength") -> None:
        super().__init__()
        self.block = BandRelationBlock(patch=patch, width=width, heads=heads, layers=layers,
                                       stats=stats, relation_mode=relation_mode,
                                       identity=identity)
        self.proj = nn.Conv2d(int(width), int(out_ch), 1)
        self.out_ch = int(out_ch)
        self.width = int(width)
        self.relation_mode, self.identity = relation_mode, identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, gh, gw = self.block.tokens(x)                 # [B,P,10,W]
        feat = z.mean(dim=2).transpose(1, 2).reshape(z.shape[0], self.width, gh, gw)
        return self.proj(feat)

    def describe(self) -> dict:
        return {"module": "BandRelationFeature", "out_ch": self.out_ch, "width": self.width,
                "patch": self.block.patch, "heads": self.block.heads,
                "layers": len(self.block.layers), "relation_mode": self.relation_mode,
                "identity": self.identity,
                "readout": "band tokens mean -> zero-init 1x1 conv to decoder width",
                "spatial": f"decoder grid H/{self.block.patch} x W/{self.block.patch}"}



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
        unused = (sum(p.numel() for p in self.relation.relation.parameters())
                  if self.relation.relation_mode == "none" else 0)
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "encoder": sum(p.numel() for p in g["encoder"]),
                "decoder": sum(p.numel() for p in g["decoder"]),
                "new": sum(p.numel() for p in g["new"]),
                # relation_mode=none 时关系偏置 MLP 仍被构造但不在前向图中，不能算有效容量
                "unused_relation_params": unused,
                "effective_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad) - unused,
                "relation_block": sum(p.numel() for p in self.relation.parameters())}

    def spectral_report(self) -> dict:
        return {"module": "BandRelationBlock", "patch": self.patch, "width": self.width,
                "heads": self.heads, "layers": self.layers,
                "relation_mode": self.relation_mode, "identity": self.identity,
                "location": "十波段入口之前，残差加回后同时进入主路与细节分支"}

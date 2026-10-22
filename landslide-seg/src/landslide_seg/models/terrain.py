"""地形（DEM＋坡度）支路与软评分融合。

设计约束来自计划 §3：

- 只复用 Torchvision ResNet18 的 stem 与第一残差阶段（1/4 分辨率、64 通道），
  不新增完整视觉骨干；输出经统一读出头得到一张评分图；
- 融合固定为 ``最终 logits = 原视觉 logits + α · tanh(地形评分)``；
- 读出头零初始化 ⇒ 初始预测与视觉参照逐位相同；
- α 可学习（默认 0.01 初始化）：若 α 也被置零，分支梯度恒为 0，读出头永远不学；
  保持 α≈0.01 时梯度 ∝ α，分支从第一步起就能收到梯度。固定 α 的对照通过
  ``alpha_learnable=False`` 把 α 注册为 buffer（不进优化器）。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TerrainEncoder(nn.Module):
    """ResNet18 stem + layer1：``[B,2,H,W] -> [B,64,H/4,W/4]``。"""

    def __init__(self, in_chans: int = 2, pretrained: str | bool = "imagenet") -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
        base = resnet18(weights=weights)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if weights is not None and in_chans != 3:
            # RGB 预训练权重对 DEM/坡度没有物理含义；均值投影只用作尺度初始化，
            # 不声称"迁移了视觉特征"。
            with torch.no_grad():
                self.conv1.weight.copy_(base.conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1))
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.out_channels = 64
        self.init_source = "imagenet" if weights is not None else "random"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return self.layer1(x)


class TerrainScoreHead(nn.Module):
    """统一读出头：64 → 32 → 1，最后一层零初始化。"""

    def __init__(self, in_chans: int = 64, width: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_chans, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = self.body(feat)
        x = self.out(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class TerrainFusionSeg(nn.Module):
    """视觉分支 + 地形软评分分支。

    输入是 12 通道（十波段 + DEM + SLOPE）：前 10 通道进视觉分支，后 2 通道进
    地形分支。数据集侧的归一化统计与通道顺序不变。
    """

    def __init__(
        self,
        visual: nn.Module,
        *,
        n_visual: int = 10,
        terrain_chans: int = 2,
        terrain_pretrained: str | bool = "imagenet",
        terrain_weights: str | Path | None = None,
        load_head: bool = False,
        head_weights: str | Path | None = None,
        alpha: float = 0.01,
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__()
        self.visual = visual
        self.n_visual = int(n_visual)
        self.terrain_channels = int(terrain_chans)
        self.terrain = TerrainEncoder(self.terrain_channels, pretrained=terrain_pretrained)
        self.head = TerrainScoreHead(self.terrain.out_channels)
        self.alpha_learnable = bool(alpha_learnable)
        if self.alpha_learnable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.terrain_init = {"kind": str(terrain_pretrained), "weights": None, "head_weights": None}
        if terrain_weights:
            self._load(terrain_weights, target="terrain")
            self.terrain_init["weights"] = str(terrain_weights).replace("\\", "/")
        if load_head and head_weights:
            self._load(head_weights, target="head")
            self.terrain_init["head_weights"] = str(head_weights).replace("\\", "/")
        # 便于 model_identity / 日志读到的公共属性
        for attr in ("adapter_type", "backbone", "use_stem", "refine", "fusion_mode",
                     "exchange_mode", "adapter_report", "load_info"):
            if hasattr(visual, attr):
                setattr(self, attr, getattr(visual, attr))

    # ------------------------------------------------------------------
    def _load(self, path: str | Path, target: str) -> None:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        state = ckpt.get("encoder" if target == "terrain" else "head", ckpt)
        module = self.terrain if target == "terrain" else self.head
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"地形权重缺少参数: {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"地形权重有多余参数: {unexpected[:5]}")

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        if x.ndim != 4 or x.shape[1] != self.n_visual + self.terrain_channels:
            raise ValueError(f"期望 [B,{self.n_visual + self.terrain_channels},H,W]，得到 {tuple(x.shape)}")
        xv = x[:, : self.n_visual]
        xt = x[:, self.n_visual:]
        out = self.visual(xv, **kwargs)
        size = out["logits"].shape[-2:]
        score = self.head(self.terrain(xt), size)
        out["terrain_score"] = score
        out["prototype_logits"] = out["logits"]
        out["logits"] = out["logits"] + self.alpha * torch.tanh(score)
        out["alpha"] = self.alpha.detach()
        return out

    # ------------------------------------------------------------------
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
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "encoder": sum(p.numel() for p in g["encoder"]),
            "decoder": sum(p.numel() for p in g["decoder"]),
            "new": sum(p.numel() for p in g["new"]),
            "terrain_encoder": sum(p.numel() for p in self.terrain.parameters()),
            "terrain_head": sum(p.numel() for p in self.head.parameters()),
        }

    def terrain_report(self) -> dict:
        return {
            "alpha": float(self.alpha.detach().cpu()),
            "alpha_learnable": self.alpha_learnable,
            "init": self.terrain_init,
            "terrain_channels": self.terrain_channels,
            "fusion": "logits_visual + alpha * tanh(terrain_score)",
            "encoder_init": getattr(self.terrain, "init_source", None),
        }

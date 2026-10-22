"""光谱旁路：在官方解码特征之后、分类之前注入一次。

三种光谱输入：

- ``indices``：四个确定性指数（NDVI、NDMI、NBR、红边归一化差），在反归一化后的
  反射率上计算，ε=1e-4；
- ``identity``：同数量、同结构的四个原始波段通道（B04/B08/B8A/B11 的 z-score），
  不含任何物理公式，用作"容量控制"；
- ``wavelength``：十个波段经一个 1×1 线性映射，权重按中心波长的高斯核初始化，
  即"波长感知的轻量嵌入"。

三种融合方式：

- ``gated``：``D' = D + α · sigmoid(G([D, S])) · Proj(S)``；
- ``additive``：``D' = D + α · Proj(S)``，同参数量的加性对照；
- ``lowrank``：``Proj(S) = U(V(S))``，rank=8 的乘性对照；
- ``fixed_gate``：门控固定为 0.5，不学习。

α 是可学习标量，初值 0；读出头/投影正常初始化，因此第一步只有 α 与投影之外的
参数不动，α 从第一步起被 ``∂L/∂α`` 推动，第二步起旁路拿到梯度。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.segformer_dataset import S2_10_BANDS

#: 各波段中心波长（微米）与原生空间分辨率（米），来自 Sentinel-2 官方规格
WAVELENGTH_UM = {
    "B02": 0.4924, "B03": 0.5598, "B04": 0.6646, "B05": 0.7039,
    "B06": 0.7402, "B07": 0.7825, "B08": 0.8328, "B8A": 0.8647,
    "B11": 1.6104, "B12": 2.1857,
}
NATIVE_GSD = {"B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
              "B07": 20, "B08": 10, "B8A": 20, "B11": 20, "B12": 20}

FEATURES = ("indices", "identity", "wavelength")
FUSIONS = ("gated", "additive", "lowrank", "fixed_gate")


class SpectralBranch(nn.Module):
    def __init__(self, feature: str, fusion: str, dec_ch: int, rank: int = 8,
                 n_out: int = 4, eps: float = 1e-4) -> None:
        super().__init__()
        if feature not in FEATURES:
            raise ValueError(f"feature 必须是 {FEATURES}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion 必须是 {FUSIONS}")
        self.feature, self.fusion, self.eps = feature, fusion, float(eps)
        self.n_out = int(n_out)
        if feature == "wavelength":
            self.embed = nn.Conv2d(len(S2_10_BANDS), self.n_out, 1, bias=False)
            self._init_wavelength_embed()
        self.proj = nn.Conv2d(self.n_out, dec_ch, 1)
        self.low_rank = None
        if fusion == "lowrank":
            self.proj = nn.Conv2d(self.n_out, rank, 1)
            self.low_rank = nn.Conv2d(rank, dec_ch, 1)
        self.gate = None
        if fusion in ("gated", "lowrank", "fixed_gate"):
            self.gate = nn.Conv2d(dec_ch + self.n_out, 1, 3, padding=1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 0.0) if fusion != "fixed_gate" else None
        self.alpha = nn.Parameter(torch.zeros(1))

    def _init_wavelength_embed(self) -> None:
        """按波长相似度初始化 1×1 权重：每个输出通道对应一个波长中心。"""
        with torch.no_grad():
            lams = torch.tensor([WAVELENGTH_UM[b] for b in S2_10_BANDS])
            centers = torch.linspace(lams.min(), lams.max(), self.n_out)
            w = torch.exp(-((lams[None, :] - centers[:, None]) / 0.25) ** 2)
            self.embed.weight.copy_(w[:, :, None, None])

    def spectral_maps(self, x10: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        dn = x10 * std + mean                     # 反归一化到 DN
        b = {name: dn[:, i:i + 1] for i, name in enumerate(S2_10_BANDS)}
        if self.feature == "indices":
            ndvi = (b["B8A"] - b["B04"]) / (b["B8A"] + b["B04"] + self.eps)
            ndmi = (b["B8A"] - b["B11"]) / (b["B8A"] + b["B11"] + self.eps)
            nbr = (b["B8A"] - b["B12"]) / (b["B8A"] + b["B12"] + self.eps)
            rend = (b["B06"] - b["B05"]) / (b["B06"] + b["B05"] + self.eps)
            return torch.cat([ndvi, ndmi, nbr, rend], dim=1)
        if self.feature == "identity":
            return torch.cat([x10[:, S2_10_BANDS.index(k):S2_10_BANDS.index(k) + 1]
                              for k in ("B04", "B08", "B8A", "B11")], dim=1)
        return self.embed(x10)

    def forward(self, D: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, dict]:
        Sd = F.interpolate(S, size=D.shape[-2:], mode="bilinear", align_corners=False)
        if self.fusion == "additive":
            delta = self.proj(Sd)
            gate_mean = None
        else:
            msg = self.proj(Sd)
            if self.low_rank is not None:
                msg = self.low_rank(msg)
            g = torch.sigmoid(self.gate(torch.cat([D, Sd], dim=1)))
            if self.fusion == "fixed_gate":
                g = torch.full_like(g, 0.5)
            delta = g * msg
            gate_mean = float(g.detach().mean())
        return delta * self.alpha, {"gate_mean": gate_mean, "alpha": float(self.alpha.detach())}


class SpectralFusionSeg(nn.Module):
    """把 ``SpectralBranch`` 挂到视觉模型的分类器之前。"""

    def __init__(self, visual: nn.Module, mean: list[float], std: list[float],
                 feature: str = "indices", fusion: str = "gated", rank: int = 8) -> None:
        super().__init__()
        self.visual = visual
        dec_ch = int(visual.model.config.decoder_hidden_size)
        self.branch = SpectralBranch(feature, fusion, dec_ch, rank=rank)
        self.register_buffer("band_mean", torch.tensor(mean, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("band_std", torch.tensor(std, dtype=torch.float32)[None, :, None, None])
        self._S: torch.Tensor | None = None
        self._last: dict = {}
        self.feature, self.fusion = feature, fusion
        # 视觉模型已在同一模块上注册了捕获钩子；本钩子在其后运行并替换分类器输入。
        visual.model.decode_head.classifier.register_forward_pre_hook(self._inject)
        for attr in ("adapter_type", "backbone", "use_stem", "refine", "fusion_mode",
                     "exchange_mode", "adapter_report", "load_info"):
            if hasattr(visual, attr):
                setattr(self, attr, getattr(visual, attr))

    def _inject(self, module, args):  # noqa: ANN001
        D = args[0]
        delta, info = self.branch(D, self._S)
        self._last = info
        return (D + delta,)

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        if x.ndim != 4 or x.shape[1] != len(S2_10_BANDS):
            raise ValueError(f"期望 [B,10,H,W]，得到 {tuple(x.shape)}")
        mean = self.band_mean.to(x.dtype)
        std = self.band_std.to(x.dtype)
        self._S = self.branch.spectral_maps(x, mean, std)
        out = self.visual(x, **kwargs)
        out["spectral_info"] = dict(self._last)
        return out

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
                "spectral_branch": sum(p.numel() for p in self.branch.parameters())}

    def spectral_report(self) -> dict:
        return {"feature": self.feature, "fusion": self.fusion,
                "alpha": float(self.branch.alpha.detach()),
                "branch_params": sum(p.numel() for p in self.branch.parameters()),
                "note": "注入位置：官方解码特征之后、分类器之前"}

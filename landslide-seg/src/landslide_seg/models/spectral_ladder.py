"""光谱三级：S1 显式输入 → S2 波段身份 → S3 关系建模。

每一级都是"十波段入口之前做一件事"，并且各自带一个同容量对照：

S1 显式物理关系（输入侧）
    处理组：在十波段后追加四个指数（NDVI、NDMI、NBR、红边归一化差），
            在反归一化的存储DN上计算，ε=1e-4，共14通道；物理反射率来源未核实；
    对照：  同样追加四个通道，但取原始波段（B04/B08/B8A/B11 的 z-score），
            不含任何物理公式。两者通道数、首层参数量完全相同。

S2 波段身份（输入侧，10 通道不变）
    处理组：每个通道加一个由中心波长生成的嵌入 ``MLP(fourier(λ))``；
    对照：  每个通道加一个可学习的标量偏置，不接触波长。
    两者新增参数量同阶（处理组是一个小 MLP，对照组是 10 个标量）。

S3 波段关系（输入侧，10 通道不变）
    处理组：低秩通道交互 ``x + B(A(x))``，A: 10→r，B: r→10，r=4；
    对照：  用满秩10×10的1×1卷积；两层之间没有激活，不属于非线性交互。
    两者都只做逐像元的跨波段线性组合。

所有变体都作用在归一化后的十波段输入上，修正后的输入同时进入 SegFormer 主路
与原始网格细节分支；不改变解码器，也不在输出端加融合项。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.segformer_dataset import S2_10_BANDS
from .spectral_contract import RADIOMETRY_MODES, harmonized_reflectance, spectral_indices

RANK_DEFAULT = 4
EPS = 1e-4

#: 各波段中心波长（微米），来自 Sentinel-2 官方规格
WAVELENGTH_UM = {
    "B02": 0.4924, "B03": 0.5598, "B04": 0.6646, "B05": 0.7039,
    "B06": 0.7402, "B07": 0.7825, "B08": 0.8328, "B8A": 0.8647,
    "B11": 1.6104, "B12": 2.1857,
}

LEVELS = ("S1", "S2", "S3")


def wavelength_features(wl: torch.Tensor) -> torch.Tensor:
    freq = 2.0 ** torch.arange(4, device=wl.device, dtype=wl.dtype) * torch.pi
    a = wl[:, None] * freq[None, :]
    return torch.cat([wl[:, None], a.sin(), a.cos()], dim=-1)


class SpectralLadder(nn.Module):
    """在十波段入口前施加 S1/S2/S3 之一；输出通道数由 ``out_chans`` 决定。"""

    def __init__(self, level: str, variant: str, stats: dict, rank: int = RANK_DEFAULT,
                 radiometry: str = "legacy_dn", extra_stats: dict | None = None) -> None:
        super().__init__()
        if level not in LEVELS:
            raise ValueError(f"level 必须是 {LEVELS}")
        if variant not in ("treatment", "control", "conditional", "free"):
            raise ValueError("variant 必须是 treatment/control/conditional/free")
        self.level, self.variant = level, variant
        if radiometry not in RADIOMETRY_MODES:
            raise ValueError(f"radiometry 必须为 {RADIOMETRY_MODES}")
        self.radiometry = radiometry
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32).flatten()
        std = torch.as_tensor(stats["std"], dtype=torch.float32).flatten()
        if mean.numel() != 10:
            raise ValueError("光谱模块需要十波段统计")
        self.register_buffer("dn_mean", mean.clone())
        self.register_buffer("dn_std", std.clamp_min(1e-6).clone())
        self.register_buffer("waves", torch.tensor([WAVELENGTH_UM[b] for b in S2_10_BANDS]))
        self.out_chans = 10
        # S1 新增四个指数通道的 TRAIN 统计；仅在 treatment 使用。
        # 非持久 buffer：历史 checkpoint 没有这些键，严格加载不受影响。
        self.register_buffer("extra_mean", torch.zeros(4), persistent=False)
        self.register_buffer("extra_std", torch.ones(4), persistent=False)
        self.extra_normalized = False
        self.extra_stats_source: str | None = None
        if extra_stats is not None:
            ex_mean = torch.as_tensor(extra_stats["mean"], dtype=torch.float32).flatten()
            ex_std = torch.as_tensor(extra_stats["std"], dtype=torch.float32).flatten()
            if ex_mean.numel() != 4 or ex_std.numel() != 4:
                raise ValueError("S1 新增通道统计必须是四个指数通道")
            self.extra_mean.copy_(ex_mean)
            self.extra_std.copy_(ex_std.clamp_min(1e-6))
            self.extra_normalized = True
            self.extra_stats_source = str(extra_stats.get("source", "external"))
        # 历史 S1 配置没有额外统计；保留其未归一化行为以兼容旧 checkpoint，
        # 新正式配置由 check_research_protocol 强制要求 extra_stats。

        if level == "S1":
            self.out_chans = 14
        elif level == "S2":
            if variant == "treatment":
                self.band_bias = nn.Sequential(nn.Linear(9, 16), nn.GELU(), nn.Linear(16, 1))
            elif variant == "control":
                self.band_bias = None
                self.channel_bias = nn.Parameter(torch.zeros(10))
            else:
                # 同参数量对照：10×10 入口混合 + 偏置；conditional 的矩阵由各波段波长生成
                self.mix_bias = nn.Parameter(torch.zeros(10))
                if variant == "conditional":
                    self.wave_to_row = nn.Linear(9, 10, bias=True)
                    nn.init.zeros_(self.wave_to_row.weight)
                    nn.init.zeros_(self.wave_to_row.bias)
                else:
                    self.free_mix = nn.Parameter(torch.zeros(10, 10))
        else:  # S3
            r = rank if variant == "treatment" else 10
            self.down = nn.Conv2d(10, r, 1, bias=False)
            self.up = nn.Conv2d(r, 10, 1, bias=False)
            nn.init.zeros_(self.up.weight)

    # ------------------------------------------------------------------
    def _indices(self, x: torch.Tensor) -> torch.Tensor:
        if self.radiometry == "harmonized_dn_v1":
            return spectral_indices(harmonized_reflectance(x, self.dn_mean, self.dn_std), EPS)
        dn = x * self.dn_std[None, :, None, None] + self.dn_mean[None, :, None, None]
        b = {n: dn[:, i:i + 1] for i, n in enumerate(S2_10_BANDS)}
        ndvi = (b["B8A"] - b["B04"]) / (b["B8A"] + b["B04"] + EPS)
        ndmi = (b["B8A"] - b["B11"]) / (b["B8A"] + b["B11"] + EPS)
        nbr = (b["B8A"] - b["B12"]) / (b["B8A"] + b["B12"] + EPS)
        rend = (b["B06"] - b["B05"]) / (b["B06"] + b["B05"] + EPS)
        return torch.cat([ndvi, ndmi, nbr, rend], dim=1)

    def _identity_channels(self, x: torch.Tensor) -> torch.Tensor:
        idx = [S2_10_BANDS.index(k) for k in ("B04", "B08", "B8A", "B11")]
        return torch.cat([x[:, i:i + 1] for i in idx], dim=1)

    def _normalized_indices(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._indices(x)
        if not self.extra_normalized:
            return idx
        return (idx - self.extra_mean[None, :, None, None]) / self.extra_std[None, :, None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.level == "S1":
            if self.variant == "treatment":
                extra = self._normalized_indices(x)
            else:
                # 对照的四个原始通道已由 dataset 用 TRAIN 十波段统计标准化。
                extra = self._identity_channels(x)
            return torch.cat([x, extra], dim=1)
        if self.level == "S2":
            if self.variant == "treatment":
                bias = self.band_bias(wavelength_features(self.waves))[:, 0]     # [10]
                return x + bias[None, :, None, None]
            if self.variant == "control":
                return x + self.channel_bias[None, :, None, None]
            weight = (self.wave_to_row(wavelength_features(self.waves))
                      if self.variant == "conditional" else self.free_mix)
            return (x + torch.einsum("bchw,dc->bdhw", x, weight)
                    + self.mix_bias[None, :, None, None])
        # S3
        return x + self.up(self.down(x))

    def report(self) -> dict:
        mechanism = {
            "S1": ("harmonized_indices_normalized" if self.extra_normalized
                   else "stored_dn_indices_unnormalized_historical") if self.variant == "treatment"
            else "extra_raw_channels_dataset_normalized",
            "S2": {"treatment": "static_wavelength_bias", "control": "free_channel_bias",
                   "conditional": "wavelength_conditional_input_mix",
                   "free": "free_channel_input_mix"}[self.variant],
            "S3": "lowrank_linear_mix" if self.variant == "treatment" else "fullrank_linear_mix",
        }
        return {"level": self.level, "variant": self.variant, "out_chans": self.out_chans,
                "radiometry": self.radiometry,
                "mechanism": mechanism[self.level],
                "extra_stats": (None if not self.extra_normalized else {
                    "mean": self.extra_mean.tolist(), "std": self.extra_std.tolist(),
                    "source": self.extra_stats_source,
                    "note": "四个指数通道用 TRAIN 统计独立标准化；十波段仍用 data.stats"}),
                "note": "入口侧变换；修正后的输入同时进入主路与细节分支"}


class SpectralLadderSeg(nn.Module):
    def __init__(self, visual: nn.Module, stats: dict, level: str, variant: str,
                 rank: int = RANK_DEFAULT, radiometry: str = "legacy_dn",
                 extra_stats: dict | None = None) -> None:
        super().__init__()
        self.visual = visual
        self.ladder = SpectralLadder(level, variant, stats, rank=rank, radiometry=radiometry,
                                     extra_stats=extra_stats)
        self.level, self.variant = level, variant
        for attr in ("adapter_type", "backbone", "use_stem", "refine", "fusion_mode",
                     "exchange_mode", "adapter_report", "load_info"):
            if hasattr(visual, attr):
                setattr(self, attr, getattr(visual, attr))

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        if x.ndim != 4 or x.shape[1] != 10:
            raise ValueError(f"期望 [B,10,H,W]，得到 {tuple(x.shape)}")
        return self.visual(self.ladder(x), **kwargs)

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
                "ladder": sum(p.numel() for p in self.ladder.parameters())}

    def spectral_report(self) -> dict:
        return self.ladder.report()

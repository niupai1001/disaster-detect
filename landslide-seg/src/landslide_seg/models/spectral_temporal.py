"""主模型的多帧时间聚合与光谱关系编码（项目适配实现）。

对应计划 ``docs/planned_temporal_spectral_experiments_2026-10-20.md`` §3。

**这不是上游预训练的光谱/时序网络。** 空间编码器、分割解码器、原网格细节分支
全部复用官方 SegFormer（``SegFormerSeg``，direct10 + stem）；本模块只做两件可独立
开关的事：

1. ``DateAttentionPool``：在四级空间特征上按「内容 + 真实日期间隔」做时间注意力，
   T=1 时严格退化为原空间特征（softmax 只有一个元素，权重恒为 1）；
2. ``BandRelationBlock``：每个 4×4 小块内把十个波段当成十个 token，
   含波长连续位置编码与可学习关系偏置（波长差 / 局部反射率差 / 归一化差），
   残差重构十个波段，输出投影零初始化 → 初始恒等。

设计边界（计划 §3.2 要求写死在实现里，不在训练中搜索）：

- 关系值在**还原反射率**后计算，不在各波段独立 z-score 上套比值；
- 负反射率先 clamp 到 0，近零分母用 ``RELATION_EPS``；
- 关系偏置末层零初始化 → 初始注意力等于普通自注意力；
- 光谱输出投影零初始化 → 初始响应与关闭该模块逐位一致；
- 4×4 只是计算分块，输出重构回 128×128，不声称提高真实空间分辨率。
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import resolve_path
from ..data.segformer_dataset import S2_10_BANDS
from .segformer import SegFormerSeg
from .spectral_contract import wavelengths_for

#: 归一化差的分母保护与裁剪范围（计划 §3.2：实现前写入，随后固定）
RELATION_EPS = 1e-3
RELATION_CLIP = 1.0
#: 十波段在 4×4 分块内的 token 数
BAND_TOKENS = 10

DATE_SPEC = {
    "reference": "2015-12-03",
    "unit": "days",
    "att_pool": "dt=(date-anchor_date)/365.25 years; [dt,sin,cos,sin2,cos2]",
}

SPECTRAL_SPEC = {
    "patch": 4,
    "width": 64,
    "heads": 4,
    "layers": 2,
    "mlp_ratio": 4,
    "wavelength_encoding": "fourier(wl_um, freq=2^k*pi, k=0..3) + wl, 9 dims -> MLP(width)",
    "relation_inputs": [
        "|wl_i - wl_j| (um)",
        "rho_i - rho_j",
        "(rho_i - rho_j) / (rho_i + rho_j + eps)",
    ],
    "relation_eps": RELATION_EPS,
    "relation_clip": RELATION_CLIP,
    "relation_init": "final linear zero-init (starts as plain attention)",
    "readout_init": "zero-init (identity at start)",
    "reflectance_source": "z-score 反变换到 DN 后除以 10000；负值 clamp 到 0",
    "missing_policy": "缺失像素在 z-score 空间填 0（TRAIN 均值），关系值在填充后计算",
}

TEMPORAL_SPEC = {
    "pooling": "per-level content conv + date MLP -> 1x1 score -> softmax over T -> weighted sum",
    "time_features": "dt=(date - anchor_date)/365.25（天→年），[dt, sin, cos, sin2, cos2]",
    "init": "score conv zero-init（初始为有效帧等权平均）",
    "t1_identity": "单个有效帧的 softmax 权重恒为 1 → 严格恒等",
    "padding": "keep 掩码把 score 置 -inf；填充帧无法影响结果",
}


def digest(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def wavelength_features(wl: torch.Tensor) -> torch.Tensor:
    """``[N]`` 微米 → ``[N,9]``：波长本身 + 4 个倍频的正弦/余弦。"""
    freq = 2.0 ** torch.arange(4, device=wl.device, dtype=wl.dtype) * torch.pi
    a = wl[:, None] * freq[None, :]
    return torch.cat([wl[:, None], a.sin(), a.cos()], dim=-1)


class _BandTokenLayer(nn.Module):
    """波段 token 自注意力层；关系偏置以 float mask 形式加到注意力 logits。"""

    def __init__(self, width: int, heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.width = int(width)
        self.heads = int(heads)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_ratio * width), nn.GELU(), nn.Linear(mlp_ratio * width, width)
        )

    def forward(self, z: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, p, n, w = z.shape
        h = self.heads
        qkv = self.qkv(self.norm1(z)).reshape(b, p, n, 3, h, w // h)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5)          # 各 [B,P,H,N,d]
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        att = att.transpose(2, 3).reshape(b, p, n, w)
        z = z + self.proj(att)
        z = z + self.mlp(self.norm2(z))
        return z


class BandRelationBlock(nn.Module):
    """``[N,10,H,W]`` 归一化输入 → 同形状残差修正。"""

    def __init__(self, *, patch: int = 4, width: int = 64, heads: int = 4, layers: int = 2,
                 mlp_ratio: int = 4, stats: dict | None = None,
                 relation_mode: str = "measured", identity: str = "wavelength") -> None:
        super().__init__()
        if stats is None:
            raise ValueError("光谱模块需要 TRAIN 统计以还原反射率")
        if relation_mode not in ("measured", "none", "global"):
            raise ValueError(f"relation_mode 必须是 measured/none/global，得到 {relation_mode!r}")
        if identity not in ("wavelength", "channel"):
            raise ValueError(f"identity 必须是 wavelength/channel，得到 {identity!r}")
        self.relation_mode = relation_mode
        self.identity = identity
        self.patch = int(patch)
        self.width = int(width)
        self.heads = int(heads)
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32).flatten()
        std = torch.as_tensor(stats["std"], dtype=torch.float32).flatten()
        if mean.numel() != BAND_TOKENS or std.numel() != BAND_TOKENS:
            raise ValueError("光谱模块需要十波段统计")
        self.register_buffer("dn_mean", mean.clone())
        self.register_buffer("dn_std", std.clamp_min(1e-6).clone())
        self.register_buffer("waves", torch.tensor(wavelengths_for(S2_10_BANDS), dtype=torch.float32))
        self.embed = nn.Linear(patch * patch, width)
        if identity == "wavelength":
            self.position = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
            self.channel_embed = None
        else:
            self.position = None
            self.channel_embed = nn.Parameter(torch.zeros(BAND_TOKENS, width))
            nn.init.normal_(self.channel_embed, std=0.02)
        self.relation = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads))
        nn.init.zeros_(self.relation[-1].weight)
        nn.init.zeros_(self.relation[-1].bias)
        self.layers = nn.ModuleList(
            [_BandTokenLayer(width, heads, mlp_ratio) for _ in range(int(layers))]
        )
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, patch * patch)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    # ------------------------------------------------------------------
    def reflectance(self, x: torch.Tensor) -> torch.Tensor:
        """z-score → DN → 反射率（0–1），负值按 0 处理。"""
        dn = x * self.dn_std[None, :, None, None] + self.dn_mean[None, :, None, None]
        return (dn / 10000.0).clamp_min(0.0)

    def relation_bias(self, dn: torch.Tensor) -> torch.Tensor:
        b, c, h, w = dn.shape
        if self.relation_mode == "none":
            return torch.zeros(b, (h // self.patch) * (w // self.patch), self.heads, c, c,
                               device=dn.device, dtype=dn.dtype)
        rho = F.avg_pool2d(dn, self.patch).flatten(2).transpose(1, 2)        # [B,P,10]
        if self.relation_mode == "global":
            rho = rho.mean(dim=1, keepdim=True).expand(-1, rho.shape[1], -1)
        diff = rho.unsqueeze(-1) - rho.unsqueeze(-2)                          # [B,P,10,10]
        nd = (diff / (rho.unsqueeze(-1) + rho.unsqueeze(-2) + RELATION_EPS)).clamp(
            -RELATION_CLIP, RELATION_CLIP
        )
        dwl = (self.waves[:, None] - self.waves[None, :]).abs().expand(b, rho.shape[1], c, c)
        feat = torch.stack([dwl, diff, nd], dim=-1)                           # [B,P,10,10,3]
        bias = self.relation(feat.to(dwl.dtype)).permute(0, 1, 4, 2, 3)       # [B,P,H,10,10]
        return bias.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch
        if c != BAND_TOKENS or h % p or w % p:
            raise ValueError(f"光谱模块需要十波段与可整除网格，得到 {tuple(x.shape)}")
        gh, gw = h // p, w // p
        tokens = x.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c, p * p)
        if self.identity == "wavelength":
            pos = self.position(wavelength_features(self.waves))              # [10,W]
        else:
            pos = self.channel_embed
        z = self.embed(tokens) + pos[None, None]
        if self.relation_mode == "none":
            bias = torch.zeros(b, gh * gw, self.heads, c, c, device=x.device, dtype=x.dtype)
        else:
            bias = self.relation_bias(self.reflectance(x))
        for layer in self.layers:
            z = layer(z, bias)
        delta = self.readout(self.norm(z)).reshape(b, gh, gw, c, p, p)
        delta = delta.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return x + delta


class DateAttentionPool(nn.Module):
    """``[B,T,C,h,w]`` + 真实日期 → ``[B,C,h,w]``。T=1 严格恒等。"""

    def __init__(self, channels: int, width: int = 32) -> None:
        super().__init__()
        self.width = int(width)
        self.content = nn.Conv2d(channels, width, 1)
        self.time = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.score = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor, dates: torch.Tensor, keep: torch.Tensor, anchor: torch.Tensor):
        b, t, c, h, w = x.shape
        ref = dates.gather(1, anchor[:, None])
        dt = (dates - ref) / 365.25                                  # 天 → 年
        feat = torch.stack([dt, dt.sin(), dt.cos(), (2 * dt).sin(), (2 * dt).cos()], dim=-1)
        z = self.content(x.reshape(b * t, c, h, w)).reshape(b, t, self.width, h, w)
        z = z + self.time(feat)[..., None, None]
        s = self.score(z.tanh().reshape(b * t, self.width, h, w)).reshape(b, t, 1, h, w)
        s = s.masked_fill(~keep[:, :, None, None, None], float("-inf"))
        weight = s.softmax(dim=1)
        return (weight * x).sum(dim=1)


class TemporalSpectralSegFormer(nn.Module):
    """主模型：共享空间编码器 + 解码前时间聚合 + 可选光谱关系编码。"""

    def __init__(self, *, spectral: bool = False, backbone: str = "b0",
                 pretrained_dir: str = "weights/segformer/segformer-b0-finetuned-ade-512-512",
                 stats: dict | None = None, patch: int = 4, width: int = 64,
                 heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.core = SegFormerSeg(
            adapter="direct10", backbone=backbone, stem=True, refine="none", exchange="none",
            pretrained_dir=str(resolve_path(pretrained_dir)), in_chans=BAND_TOKENS,
            band_order=list(S2_10_BANDS),
        )
        self.spectral_enabled = bool(spectral)
        self.spectral = (
            BandRelationBlock(patch=patch, width=width, heads=heads, layers=layers, stats=stats)
            if spectral else None
        )
        hidden = [int(v) for v in self.core.model.config.hidden_sizes]
        self.hidden_sizes = hidden
        self.pools = nn.ModuleList([DateAttentionPool(c) for c in hidden])
        # 训练器/checkpoint 直接读取的身份字段
        self.adapter_type = self.core.adapter_type
        self.backbone = backbone
        self.use_stem = True
        self.refine = "none"
        self.fusion_mode = None
        self.exchange_mode = None
        self.load_info = self.core.load_info
        self.adapter_report = {
            "type": "temporal_spectral_segformer",
            "spatial_pretrained": self.core.adapter_report,
            "spectral_enabled": self.spectral_enabled,
            "spectral_spec": SPECTRAL_SPEC if self.spectral_enabled else None,
            "temporal_spec": TEMPORAL_SPEC,
            "date_spec": DATE_SPEC,
            "bands": list(S2_10_BANDS),
            "wavelengths_um": list(wavelengths_for(S2_10_BANDS)),
            "hidden_sizes": hidden,
            "note": "时间聚合与光谱模块无预训练权重，随机初始化后监督训练；空间编码器/解码器/细节分支复用官方权重",
        }
        self.core_init_digest = digest(self.core)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, dates: torch.Tensor | None = None,
                keep: torch.Tensor | None = None, anchor: torch.Tensor | None = None) -> dict:
        if x.ndim == 4:
            x = x[:, None]
        b, t, c, h, w = x.shape
        if c != BAND_TOKENS:
            raise ValueError(f"主模型只用十波段（不加 DEM），得到 {c} 通道")
        if keep is None:
            keep = torch.ones(b, t, dtype=torch.bool, device=x.device)
        if anchor is None:
            anchor = torch.zeros(b, dtype=torch.long, device=x.device)
        if dates is None:
            dates = torch.zeros(b, t, dtype=torch.float32, device=x.device)
        dates = dates.to(dtype=x.dtype)

        inp = x.reshape(b * t, c, h, w)
        if self.spectral is not None:
            inp = self.spectral(inp)
        hidden = self.core.model.segformer(
            pixel_values=inp, output_hidden_states=True
        ).hidden_states
        if len(hidden) != len(self.pools):
            raise RuntimeError(f"编码器输出 {len(hidden)} 级，期望 {len(self.pools)} 级")
        pooled = []
        for i, (feat, pool) in enumerate(zip(hidden, self.pools)):
            ch = self.hidden_sizes[i]
            if feat.ndim != 4 or feat.shape[1] != ch:
                raise RuntimeError(f"第 {i} 级特征形状意外: {tuple(feat.shape)}")
            f = feat.reshape(b, t, ch, feat.shape[2], feat.shape[3])
            pooled.append(pool(f, dates, keep, anchor))

        self.core._decoder_feat = None
        logits = self.core.model.decode_head(tuple(pooled))
        feat = self.core._decoder_feat
        if feat is None:
            raise RuntimeError("未能捕获解码器特征")
        base = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        selected = inp.reshape(b, t, c, h, w)[torch.arange(b, device=x.device), anchor]
        stem = self.core.stem(selected)
        delta = self.core.fusion(feat, stem, (h, w))
        return {"logits": base + delta, "base_logits": base}

    # ------------------------------------------------------------------
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = self.core.param_groups()
        extra = list(self.pools.parameters())
        if self.spectral is not None:
            extra += list(self.spectral.parameters())
        groups["new"] = groups["new"] + extra
        return groups

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        groups = self.param_groups()
        spectral = sum(p.numel() for p in self.spectral.parameters()) if self.spectral is not None else 0
        temporal = sum(p.numel() for p in self.pools.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "encoder": sum(p.numel() for p in groups["encoder"]),
            "decoder": sum(p.numel() for p in groups["decoder"]),
            "new": sum(p.numel() for p in groups["new"]),
            "temporal_aggregation": temporal,
            "spectral": spectral,
        }

    def spatial_digest(self) -> str:
        """空间主干（编码器/解码器/细节分支）的初始化摘要，用于跨组一致性验收。"""
        return digest(self.core)

    def module_activity(self) -> dict:
        """训练后诊断：时间/光谱新增参数是否真的离开零初始化。"""
        out = {
            "temporal": {
                "score_weight_abs_sum": float(sum(
                    p.score.weight.detach().abs().sum() for p in self.pools
                )),
                "content_weight_abs_sum": float(sum(
                    p.content.weight.detach().abs().sum() for p in self.pools
                )),
                "time_weight_abs_sum": float(sum(
                    p.time[0].weight.detach().abs().sum() for p in self.pools
                )),
            },
            "spectral": None,
        }
        if self.spectral is not None:
            out["spectral"] = {
                "readout_weight_abs_sum": float(self.spectral.readout.weight.detach().abs().sum()),
                "relation_last_weight_abs_sum": float(
                    self.spectral.relation[-1].weight.detach().abs().sum()
                ),
                "layer0_qkv_weight_abs_sum": float(
                    self.spectral.layers[0].qkv.weight.detach().abs().sum()
                ),
                "layer1_mlp_weight_abs_sum": float(
                    self.spectral.layers[-1].mlp[0].weight.detach().abs().sum()
                ),
            }
        return out

    def projection_change(self) -> None:
        return self.core.projection_change()


def build_temporal_model(cfg_segformer: dict, stats: dict) -> TemporalSpectralSegFormer:
    """按 ``segformer.model`` 配置构造主模型；训练与评测共用同一入口。"""
    m = cfg_segformer["model"]
    if str(m.get("type", "temporal_spectral")) != "temporal_spectral":
        raise ValueError("该入口只构造 temporal_spectral 主模型")
    return TemporalSpectralSegFormer(
        spectral=bool(m.get("spectral", False)),
        backbone=str(m.get("backbone", "b0")),
        pretrained_dir=str(m.get("pretrained_dir",
                                "weights/segformer/segformer-b0-finetuned-ade-512-512")),
        stats=stats,
        patch=int(m.get("spectral_patch", 4)),
        width=int(m.get("spectral_width", 64)),
        heads=int(m.get("spectral_heads", 4)),
        layers=int(m.get("spectral_layers", 2)),
    )

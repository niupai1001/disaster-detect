"""S4：把真正的十波段关系表示作为独立 ``F_spec``，在单一尺度上融合。

与 ``SpectralRelationSeg``（入口前 ``x+Δx``）不同，这里视觉主路仍从**原始十波段**
进入预训练 MiT；关系编码器只产生 ``F_spec``，在官方解码器的融合特征 ``F_vis``
处注入一次：

    F_fused = F_vis + alpha * gate([F_vis, F_spec]) * F_spec

``fusion``：

- ``add``：无门控、同尺度加法（gate 恒为 1）；
- ``fixed``：常数门（固定 ``fixed_gate``，不学习）；
- ``learn``：输入依赖的空间门（3×3 卷积，零初始化 → 初始 0.5）。

``alpha`` 初值 0；``F_spec`` 投影正常初始化，因此任务梯度先推动 ``alpha``，
再由 ``alpha`` 把梯度传到关系编码器与门控。初始 ``alpha=0`` 保证融合支路对输出
零贡献，与同权重基线逐位一致。原网格细节分支仍读融合前的解码特征与原始影像。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.segformer_dataset import S2_10_BANDS
from .spectral_relation import BandRelationFeature

FUSIONS = ("add", "fixed", "learn")


class SpectralGateSeg(nn.Module):
    def __init__(self, visual: nn.Module, stats: dict, *, fusion: str = "learn",
                 patch: int = 4, width: int = 64, heads: int = 4, layers: int = 2,
                 relation_mode: str = "measured", identity: str = "wavelength",
                 fixed_gate: float = 0.5) -> None:
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion 必须是 {FUSIONS}，得到 {fusion!r}")
        self.visual = visual
        self.fusion_mode = fusion
        self.fixed_gate = float(fixed_gate)
        if fusion == "fixed" and not 0.0 <= self.fixed_gate <= 1.0:
            raise ValueError("fixed_gate 必须在 [0,1]")
        dec_ch = int(visual.model.config.decoder_hidden_size)
        self.feature = BandRelationFeature(
            out_ch=dec_ch, patch=patch, width=width, heads=heads, layers=layers,
            stats=stats, relation_mode=relation_mode, identity=identity,
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gate = None
        if fusion == "learn":
            self.gate = nn.Conv2d(2 * dec_ch, 1, 3, padding=1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self._last_gate: torch.Tensor | None = None
        self._last_corr: torch.Tensor | None = None
        for attr in ("adapter_type", "backbone", "use_stem", "refine",
                     "exchange_mode", "adapter_report", "load_info"):
            if hasattr(visual, attr):
                setattr(self, attr, getattr(visual, attr))

    # ------------------------------------------------------------------
    def _fuse(self, feat: torch.Tensor, fspec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if fspec.shape[-2:] != feat.shape[-2:]:
            fspec = F.interpolate(fspec, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        if self.fusion_mode == "add":
            gate = None
            corr = fspec
        elif self.fusion_mode == "fixed":
            gate = torch.full_like(fspec[:, :1], self.fixed_gate)
            corr = gate * fspec
        else:
            gate = torch.sigmoid(self.gate(torch.cat([feat, fspec], dim=1)))
            corr = gate * fspec
        return self.alpha * corr, gate

    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        if x.ndim != 4 or x.shape[1] != len(S2_10_BANDS):
            raise ValueError(f"期望 [B,10,H,W]，得到 {tuple(x.shape)}")
        visual = self.visual
        inp = visual.projection(x) if visual.projection is not None else x
        hidden = visual.model.segformer(pixel_values=inp, output_hidden_states=True,
                                        return_dict=True).hidden_states
        decoder = visual.model.decode_head
        projected = []
        for feat, mlp in zip(hidden, decoder.linear_c):
            if feat.ndim != 4:
                raise ValueError("S4 显式解码要求四维分层特征")
            b, _, h, w = feat.shape
            z = mlp(feat).permute(0, 2, 1).reshape(b, -1, h, w)
            projected.append(F.interpolate(z, size=hidden[0].shape[-2:], mode="bilinear",
                                           align_corners=False))
        feat = decoder.linear_fuse(torch.cat(projected[::-1], dim=1))
        feat = decoder.dropout(decoder.activation(decoder.batch_norm(feat)))
        fspec = self.feature(x)
        correction, gate = self._fuse(feat, fspec)
        low_logits = decoder.classifier(feat + correction)
        stem = visual.stem(x) if visual.use_stem else None
        base = F.interpolate(low_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        detail = visual.fusion(feat, stem, x.shape[-2:]) if visual.use_stem else None
        out = {"logits": base + detail if detail is not None else base,
               "base_logits": base,
               "spectral_info": {
                   "fusion": self.fusion_mode,
                   "alpha": float(self.alpha.detach()),
                   "fspec_abs_mean": float(fspec.detach().abs().mean()),
                   "corr_abs_mean": float(correction.detach().abs().mean()),
                   "gate_mean": None if gate is None else float(gate.detach().mean()),
                   "gate_var": None if gate is None else float(gate.detach().var(unbiased=False)),
               }}
        self._last_gate = None if gate is None else gate.detach()
        self._last_corr = correction.detach().abs().mean(dim=1, keepdim=True)
        if kwargs.get("return_features"):
            out["decoder_feat"] = feat
        return out

    # ------------------------------------------------------------------
    @property
    def last_gate(self) -> torch.Tensor | None:
        return self._last_gate

    @property
    def last_corr_mag(self) -> torch.Tensor | None:
        return self._last_corr

    def diagnostics(self) -> dict:
        """最近一次前向的门与修正幅度（供评测脚本记录，不参与梯度）。"""
        return {"gate": self._last_gate, "corr_mag": self._last_corr,
                "alpha": float(self.alpha.detach()), "fusion": self.fusion_mode}

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
        unused = (sum(p.numel() for p in self.feature.block.relation.parameters())
                  if self.feature.block.relation_mode == "none" else 0)
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "encoder": sum(p.numel() for p in g["encoder"]),
                "decoder": sum(p.numel() for p in g["decoder"]),
                "new": sum(p.numel() for p in g["new"]),
                "feature_branch": sum(p.numel() for p in self.feature.parameters()),
                "relation_block": sum(p.numel() for p in self.feature.block.parameters()),
                "gate": sum(p.numel() for p in self.gate.parameters()) if self.gate is not None else 0,
                "unused_relation_params": unused,
                "effective_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad) - unused}

    def spectral_report(self) -> dict:
        return {"module": "SpectralGateSeg", "fusion": self.fusion_mode,
                "fixed_gate": self.fixed_gate if self.fusion_mode == "fixed" else None,
                "alpha_init": 0.0, "alpha": float(self.alpha.detach()),
                "feature": self.feature.describe(),
                "branch_params": sum(p.numel() for p in self.feature.parameters()),
                "gate_params": sum(p.numel() for p in self.gate.parameters()) if self.gate is not None else 0,
                "location": "官方解码器融合特征处、分类器之前；视觉主路仍读原始十波段"}

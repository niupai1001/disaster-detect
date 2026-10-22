"""编码阶段的单次通道残差交换（v4r4）。

这是执行计划 v4r4 的核心：把原网格 Stem 的细节**提前**送进编码器阶段 2 的输出，
让它在第三阶段之前就参与计算，而不是只在末端与解码特征拼接。

两个变体共用同一套参数结构，**参数量精确相等**，唯一区别是信息从哪里取：

``self``（容量对照）::

    E_new = E + α_E · H_E(P(E))
    D_new = D + α_D · H_D(D)

``cross``（真正的跨路径交换）::

    E_new = E + α_E · H_E(adaptive_avg_pool(D, 16×16))
    D_new = D + α_D · H_D(bilinear_upsample(P(E), H×W))

三个刻意的约束：

1. ``α_E / α_D`` 按通道零初始化 → 初始 ``E_new ≡ E``、``D_new ≡ D``，整网与
   LateDetail 逐像素一致；零初始化会让 ``H_*`` 在第一次 backward 前拿不到梯度，
   属于预期行为，需在验证脚本里用多步窗口确认而不是首步断言。
2. ``P / H_E / H_D`` 的 bias 策略与初始化方式在两组之间完全一致，构造顺序也一致，
   因此"同 seed 构造"可以真正得到相同参数（仍要用哈希核对，不能只信 seed）。
3. 交换块内只做投影与残差加和，不引入 concat / 注意力 / 额外归一化；
   上下采样方式固定，不做多倍率搜索。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EXCHANGE_MODES = ("none", "self", "cross")


def _init_conv(conv: nn.Conv2d) -> None:
    """两组统一的明确初始化：kaiming-normal(relu) + bias 置零。"""
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class ChannelwiseResidualExchange(nn.Module):
    """``(E, D) → (E_new, D_new)``，``E`` 为阶段 2 特征，``D`` 为原网格细节特征。

    Parameters
    ----------
    e_ch:
        编码器阶段 2 的通道数（MiT-B0 为 64）。
    d_ch:
        原网格 Stem 的通道数（本项目固定 32）。``P`` 把 ``E`` 投到这组通道，
        使两条路径能在同一通道空间里被同一套卷积处理。
    mode:
        ``self`` 或 ``cross``。
    """

    def __init__(self, *, e_ch: int = 64, d_ch: int = 32, mode: str = "cross") -> None:
        super().__init__()
        if mode not in ("self", "cross"):
            raise ValueError(f"exchange.mode 必须是 self/cross，得到 {mode!r}")
        self.e_ch = int(e_ch)
        self.d_ch = int(d_ch)
        self.mode = mode

        # 编码特征 → 细节通道空间；只在 cross 里用于回写 D，在 self 里用于给 H_E 供料
        self.p = nn.Conv2d(self.e_ch, self.d_ch, 1, bias=True)
        # 细节 → 编码（输出回到 e_ch，才能加在 E 上）
        self.h_e = nn.Sequential(
            nn.Conv2d(self.d_ch, self.d_ch, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.d_ch, self.e_ch, 1, bias=True),
        )
        # 编码 → 细节（输出保持在 d_ch）
        self.h_d = nn.Sequential(
            nn.Conv2d(self.d_ch, self.d_ch, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.d_ch, self.d_ch, 1, bias=True),
        )
        # 逐通道残差系数，零初始化
        self.alpha_e = nn.Parameter(torch.zeros(self.e_ch))
        self.alpha_d = nn.Parameter(torch.zeros(self.d_ch))

        for conv in (self.p, self.h_e[0], self.h_e[2], self.h_d[0], self.h_d[2]):
            _init_conv(conv)

        self.last_shapes: dict | None = None

    # ------------------------------------------------------------------
    def forward(self, e: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if e.ndim != 4 or d.ndim != 4:
            raise ValueError(f"E/D 必须是 4 维，得到 {tuple(e.shape)} / {tuple(d.shape)}")
        if e.shape[1] != self.e_ch:
            raise ValueError(f"E 通道应为 {self.e_ch}，得到 {e.shape[1]}")
        if d.shape[1] != self.d_ch:
            raise ValueError(f"D 通道应为 {self.d_ch}，得到 {d.shape[1]}")
        if d.shape[0] != e.shape[0]:
            raise ValueError(f"batch 不一致: {e.shape[0]} vs {d.shape[0]}")
        # E 与 D 必须是同一空间尺度下的两套表示，否则"原网格"名不副实
        if d.shape[-2] % e.shape[-2] or d.shape[-1] % e.shape[-1]:
            raise ValueError(f"D {tuple(d.shape[-2:])} 与 E {tuple(e.shape[-2:])} 尺度不整除")
        self.last_shapes = {"E": tuple(e.shape), "D": tuple(d.shape)}

        size_e = e.shape[-2:]
        size_d = d.shape[-2:]

        if self.mode == "cross":
            src_e = F.adaptive_avg_pool2d(d, size_e)          # [B,d_ch,h,w]
            src_d = F.interpolate(self.p(e), size=size_d, mode="bilinear", align_corners=False)
        else:
            src_e = self.p(e)
            src_d = d

        k_e = self.h_e(src_e)
        k_d = self.h_d(src_d)
        e_new = e + self.alpha_e[None, :, None, None] * k_e
        d_new = d + self.alpha_d[None, :, None, None] * k_d
        return e_new, d_new

    # ------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "e_ch": self.e_ch,
            "d_ch": self.d_ch,
            "params": sum(p.numel() for p in self.parameters()),
            "alpha_init_max_abs": max(
                float(self.alpha_e.detach().abs().max()), float(self.alpha_d.detach().abs().max())
            ),
            "modules": {
                "P": "Conv2d(e_ch,d_ch,1,bias=True)",
                "H_E": "Conv2d(d_ch,d_ch,3,p=1) → GELU → Conv2d(d_ch,e_ch,1)",
                "H_D": "Conv2d(d_ch,d_ch,3,p=1) → GELU → Conv2d(d_ch,d_ch,1)",
                "alpha_E": f"[{self.e_ch}] zeros",
                "alpha_D": f"[{self.d_ch}] zeros",
            },
            "init": "kaiming_normal(fan_out,relu) + zero bias; alpha 零初始化",
            "note": "self/cross 参数结构逐项相同，参数量精确相等",
        }

    def alphas(self) -> dict:
        """训练后报告用：两方向残差系数的幅度。"""
        return {
            "alpha_e_abs_mean": float(self.alpha_e.detach().abs().mean()),
            "alpha_e_abs_max": float(self.alpha_e.detach().abs().max()),
            "alpha_d_abs_mean": float(self.alpha_d.detach().abs().mean()),
            "alpha_d_abs_max": float(self.alpha_d.detach().abs().max()),
        }

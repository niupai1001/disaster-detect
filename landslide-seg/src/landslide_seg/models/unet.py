"""SMP ResNet18 U-Net 对照（10 通道适配 + ImageNet 预训练加载报告）。

用途（执行计划 §5 的 U0）：一个成熟的卷积编码器/解码器参照，与 SegFormer 共用
数据、预处理、预算与指标；明确标注它不是"原版 U-Net 的逐层复现"。

10 通道适配与 direct10 一致：先加载 3 通道 ImageNet 权重，再把首层换成 10 通道，
R/G/B 权重拷到 B04/B03/B02，其余 7 通道零初始化且可训练。首层替换前后都记录
加载信息；若 ImageNet 权重不可用，则显式记录 `pretrained=False` 并回退随机初始化，
绝不把随机初始化结果冒充预训练对照。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

RGB_TO_BAND_INDEX = {"R": 2, "G": 1, "B": 0}


class UNetSeg(nn.Module):
    """``[B,10,H,W] → {"logits": [B,1,H,W], ...}``，与 SegFormerSeg 输出契约一致。"""

    def __init__(
        self,
        *,
        in_chans: int = 10,
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        if encoder_weights == "imagenet":
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        self.in_chans = in_chans
        self.encoder_name = encoder_name
        reason = None
        pretrained = False
        try:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
            )
            pretrained = encoder_weights == "imagenet"
        except Exception as exc:  # 权重下载失败：显式回退，不伪装成预训练
            reason = f"{type(exc).__name__}: {exc}"
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            pretrained = False

        self.adapter_report = self._expand_first_conv(in_chans)
        self.load_info = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "pretrained": pretrained,
            "fallback_reason": reason,
            "adapter": self.adapter_report,
        }

    def _expand_first_conv(self, in_chans: int) -> dict:
        old = self.model.encoder.conv1
        if not isinstance(old, nn.Conv2d) or old.in_channels != 3:
            raise RuntimeError(f"未找到 3 通道首层卷积: {old!r}")
        new = nn.Conv2d(
            in_chans,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, RGB_TO_BAND_INDEX["R"]] = old.weight[:, 0]  # R → B04
            new.weight[:, RGB_TO_BAND_INDEX["G"]] = old.weight[:, 1]  # G → B03
            new.weight[:, RGB_TO_BAND_INDEX["B"]] = old.weight[:, 2]  # B → B02
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.model.encoder.conv1 = new
        zeros = [i for i in range(in_chans) if i not in RGB_TO_BAND_INDEX.values()]
        return {
            "type": "unet_first_conv_10ch",
            "old": f"Conv2d(3,{old.out_channels},k={tuple(old.kernel_size)},s={tuple(old.stride)})",
            "new": f"Conv2d({in_chans},{old.out_channels})",
            "zero_init_channels": zeros,
        }

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim != 4 or x.shape[1] != self.in_chans:
            raise ValueError(f"期望 [B,{self.in_chans},H,W]，得到 {tuple(x.shape)}")
        logits = self.model(x)
        return {"logits": logits, "base_logits": logits, "delta_logits": None}

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        enc, dec = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("model.encoder."):
                enc.append(p)
            else:
                dec.append(p)
        return {"encoder": enc, "decoder": dec, "new": []}

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        groups = self.param_groups()
        return {
            "total": total,
            "trainable": trainable,
            "encoder": sum(p.numel() for p in groups["encoder"]),
            "decoder": sum(p.numel() for p in groups["decoder"]),
            "new": 0,
        }

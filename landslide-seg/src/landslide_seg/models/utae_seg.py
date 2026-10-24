"""官方 U-TAE 的封装，供本项目训练循环使用。

严格的"官方实现"：直接 import 锁定 commit 下 ``vendor/official/Sen12Landslides/src/models/utae.py``
的 ``UTAE``，不改结构、不改默认超参；只做输入/输出契约适配。

- 输入 ``[B,T,C,H,W]``，C=11（十波段 + 广播的 DEM），与官方 ``Sen12Landslides`` 数据集一致；
- 输出 ``{"logits": [B,1,H,W]}``；
- ``param_groups()`` 把所有参数放进 ``encoder`` 组，使优化器使用单一学习率（官方 AdamW 1e-3 单组）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..config import resolve_path

OFFICIAL_DIR = "vendor/official/Sen12Landslides"


def _load_utae():
    root = resolve_path(OFFICIAL_DIR)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.models.utae import UTAE  # type: ignore
    return UTAE


class UTAESeg(nn.Module):
    def __init__(self, in_channels: int = 11, num_classes: int = 1,
                 agg_mode: str = "att_group") -> None:
        super().__init__()
        UTAE = _load_utae()
        self.net = UTAE(in_channels=in_channels, num_classes=num_classes, encoder=False,
                        agg_mode=agg_mode)
        self.in_channels = int(in_channels)
        self.adapter_type = "official_utae"
        self.backbone = "utae"
        self.use_stem = False
        self.refine = "none"
        self.fusion_mode = None
        self.exchange_mode = None
        self.load_info = {"official_source": OFFICIAL_DIR,
                          "impl": "src/models/utae.py", "from_scratch": True,
                          "note": "官方未提供该任务 checkpoint，按计划单 seed 复训"}
        self.adapter_report = {"type": "official_utae", "in_channels": self.in_channels,
                               "num_classes": num_classes, "dem": "广播到每个时相",
                               "agg_mode": agg_mode, "encoder_norm": "group"}
        self._init_digest = self.digest()

    def forward(self, x: torch.Tensor, dates: torch.Tensor | None = None) -> dict:
        if x.ndim != 5 or x.shape[2] != self.in_channels:
            raise ValueError(f"期望 [B,T,{self.in_channels},H,W]，得到 {tuple(x.shape)}")
        out = self.net({"img": x, "dates": dates})
        return {"logits": out["segmentation"]}

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"encoder": [p for p in self.parameters() if p.requires_grad], "decoder": [], "new": []}

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": train, "encoder": train, "decoder": 0, "new": 0}

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    @property
    def init_digest(self) -> str:
        return self._init_digest

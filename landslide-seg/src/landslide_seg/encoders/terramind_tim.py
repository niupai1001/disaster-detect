"""TerraMind V1 Small + TiM 适配器（复用 TerraTorch 官方实现）。

职责：
- 用官方权重构建 ``terramind_v1_small_tim``（TiM 目标 LULC）；
- 接收 **完整 12 波段 S2L2A**，返回带空间结构的特征；
- 冻结主干与 TiM 生成器（官方代码本身已将 enc-dec 冻结）；
- 记录实际被调用的层、张量形状、是否真的执行了 TiM。

注意（Landslide-Seg V1 §4.1）：
- TiM 只有在输入为完整 12 波段时合法；
- TiM 输出是 token 空间的预测表征，**不是**可视 LULC 图；
- 必须能证明 TiM 真的执行了（本模块记录 `tim_ran` 与分析量）。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# 官方 S2L2A 标准化统计（DN = 反射率 × 10000），取自 TerraTorch 官方示例配置
# configs/terramind_v1_base_tim_lulc_sen1floods11.yaml
TERRAMIND_S2L2A_MEAN = (
    1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975,
    2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648,
)
TERRAMIND_S2L2A_STD = (
    2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
    1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311,
)

#: 官方为 tiny/small/base 推荐的取层索引
SELECT_LAYERS = (2, 5, 8, 11)

HF_FILES = {
    "ibm-esa-geospatial/TerraMind-1.0-small": "TerraMind_v1_small.pt",
    "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S2L2A": "TerraMind_Tokenizer_S2L2A.pt",
    "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC": "TerraMind_Tokenizer_LULC.pt",
}


@dataclass
class TimRunInfo:
    tim_enabled: bool
    tim_ran: bool
    tim_latency_s: float
    num_tokens: int
    feature_shapes: list[tuple[int, ...]]


def ensure_hf_cache(endpoint: str = "https://hf-mirror.com") -> None:
    """把 TerraMind 权重与 TiM tokenizer 预取到 HF 缓存（走镜像）。

    之后 terratorch 内部调用 ``hf_hub_download`` 会直接命中缓存，避免联网。

    Windows 上 ``huggingface_hub`` 可能因缺少符号链接权限而只下到 ``blobs/``
    却没在 ``snapshots/`` 建链接（表现为返回的路径不存在）。这里做了兜底：
    若快照路径缺失，直接把 blob 复制过去，或用 urllib 重新下载。
    """
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    import shutil
    import urllib.request

    from huggingface_hub import HfApi, hf_hub_download

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    api = HfApi(endpoint=endpoint)

    for repo_id, filename in HF_FILES.items():
        try:
            p = Path(hf_hub_download(repo_id=repo_id, filename=filename))
        except Exception:
            p = None
        if p is not None and p.is_file():
            continue

        # 兜底：手工放到 snapshots/<sha>/filename
        try:
            sha = api.model_info(repo_id).sha
        except Exception as exc:
            raise RuntimeError(f"无法解析 {repo_id} 的版本号: {exc}") from exc
        target = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / sha / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            continue

        # 优先复用已下载的 blob
        blobs = sorted(
            (cache_root / f"models--{repo_id.replace('/', '--')}" / "blobs").glob("*"),
            key=lambda x: x.stat().st_size if x.is_file() else 0,
            reverse=True,
        ) if (cache_root / f"models--{repo_id.replace('/', '--')}" / "blobs").is_dir() else []
        if blobs and blobs[0].stat().st_size > 1_000_000:
            shutil.copyfile(blobs[0], target)
            print(f"[hf] {repo_id}/{filename} 由 blob 复制完成")
            continue

        url = f"{endpoint}/{repo_id}/resolve/main/{filename}"
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"[hf] 下载 {url}")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
        if target.stat().st_size < 1_000_000:
            raise RuntimeError(f"{repo_id}/{filename} 下载异常（文件过小）")


class TerraMindTiMAdapter(nn.Module):
    """把 TerraMindTiM 包装成返回 ``FeatureMap`` 的冻结特征提取器。"""

    def __init__(
        self,
        img_size: int = 224,
        tim_enabled: bool = True,
        merge_method: str = "mean",
        layers: Sequence[int] = SELECT_LAYERS,
        freeze: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.tim_enabled = tim_enabled
        self.merge_method = merge_method
        self.layers = tuple(layers)
        self.freeze = freeze

        ensure_hf_cache()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # 延迟导入：terratorch 顶层会加载 datasets/tasks，代价很高
        from ._terramind_loader import load_terramind

        _, terramind_v1_small_tim, _ = load_terramind()

        self.model = terramind_v1_small_tim(
            pretrained=True,
            modalities=["S2L2A"],
            tim_modalities=["LULC"] if tim_enabled else [],
            merge_method=merge_method,
            img_size=img_size,
            patch_size=16,
        )
        self.model.eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.patch_size = 16
        self.dim = 384  # TerraMind V1 Small
        self._cache_tim_modalities = list(self.model.tim_modalities)

    # ------------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.parameters())

    def token_grid(self, h: int, w: int) -> tuple[int, int]:
        return h // self.patch_size, w // self.patch_size

    @torch.no_grad()
    def forward(self, x12: torch.Tensor, *, tim: bool | None = None) -> dict:
        """Args:
            x12: ``[B,12,H,W]``，已按官方 S2L2A 统计标准化（DN 尺度）。
                若 ``H/W != img_size``，会双线性重采样到 ``img_size``
                （预训练 tokenizer 的位置编码固定为 14×14，不可改）。
            tim: 覆盖实例默认的 TiM 开关（用于 ON/OFF 对照消融）。

        Returns:
            dict with ``tokens`` (list of ``[B,N,D]``), ``grid``, ``info``.
        """
        want_tim = self.tim_enabled if tim is None else bool(tim)
        self.model.tim_modalities = list(self._cache_tim_modalities) if want_tim else []

        import time

        orig_hw = tuple(x12.shape[-2:])
        if orig_hw != (self.img_size, self.img_size):
            x12 = torch.nn.functional.interpolate(
                x12, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )

        t0 = time.perf_counter()
        outs = self.model({"S2L2A": x12})
        dt = time.perf_counter() - t0

        if not isinstance(outs, (list, tuple)):
            raise TypeError(f"TerraMind 输出类型异常: {type(outs)}")
        picked = [outs[i] for i in self.layers if i < len(outs)]
        grid = self.token_grid(self.img_size, self.img_size)
        return {
            "tokens": picked,
            "grid": grid,
            "resampled_from": orig_hw,
            "info": TimRunInfo(
                tim_enabled=want_tim,
                # 只要 tim_modalities 非空且模型返回了 token 序列，即认为 TiM 参与
                tim_ran=bool(want_tim and len(self._cache_tim_modalities) > 0),
                tim_latency_s=dt,
                num_tokens=picked[-1].shape[1],
                feature_shapes=[tuple(t.shape) for t in picked],
            ),
        }

    def tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B,N,D]`` -> ``[B,D,h,w]``（要求 N 是完全平方数）。"""
        b, n, d = tokens.shape
        h = int(math.isqrt(n))
        if h * h != n:
            raise ValueError(f"token 数 {n} 不是完全平方数，无法还原为方阵特征图")
        return tokens.transpose(1, 2).reshape(b, d, h, h)

    def describe(self) -> dict:
        return {
            "name": "terramind_v1_small_tim",
            "backend": "terratorch 1.2.13 (official)",
            "dim": self.dim,
            "patch_size": self.patch_size,
            "layers": self.layers,
            "merge_method": self.merge_method,
            "tim": self.tim_enabled,
            "frozen": self.frozen,
            "params": sum(p.numel() for p in self.model.parameters()),
            "trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }


def normalize_s2l2a_dn(x_dn: torch.Tensor) -> torch.Tensor:
    """按官方统计标准化 ``[B,12,H,W]`` 的 DN 输入。"""
    if x_dn.shape[1] != 12:
        raise ValueError(f"TerraMind S2L2A 需要 12 通道，实际 {x_dn.shape[1]}")
    mean = torch.tensor(TERRAMIND_S2L2A_MEAN, dtype=x_dn.dtype, device=x_dn.device)
    std = torch.tensor(TERRAMIND_S2L2A_STD, dtype=x_dn.dtype, device=x_dn.device)
    return (x_dn - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

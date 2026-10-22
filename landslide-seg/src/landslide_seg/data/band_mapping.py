"""Sentinel-2 波段契约层（single source of truth）。

本模块只做两件事：
1. 声明各模型/数据集**官方**的波段顺序与原生分辨率；
2. 在把某个数据集的张量喂给某个骨干之前，**显式校验**波段是否齐全，
   缺失时抛出 `MissingBandError`，而不是静默补零。

设计约束（来自 Landslide-Seg V1 §3.1 / §3.2）：
- TerraMind V1 TiM 对 S2L2A 要求完整 12 个预训练波段，**不支持子集**；
- AnySat S2 分支要求 10 波段 + 日期；
- 缺失波段补零 / 用近邻波段顶替都属于分布外输入，只能用于 **接口 smoke test**，
  必须在样本元数据里记录 `synthetic_bands`，且不得用于有效性结论或论文评测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 官方波段定义
# ---------------------------------------------------------------------------

#: Sentinel-2 L2A 完整 12 波段，TerraMind 官方顺序
S2_L2A_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)

#: 各波段原生地面分辨率（米）。注意 B01/B09 原生 60m。
S2_L2A_NATIVE_RES_M: tuple[int, ...] = (
    60, 10, 10, 10, 20, 20,
    20, 10, 20, 60, 20, 20,
)

#: AnySat S2 分支官方 10 波段（顺序即输入通道顺序）
ANY_SAT_S2_BANDS: tuple[str, ...] = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)

#: Landslide4Sense 14 通道（官方数据卡：S2 B1–B12 + ALOS PALSAR 坡度 + DEM）
#: 注意：**包含 B10（卷云），不包含 B8A**。
L4S_BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B09", "B10", "B11", "B12",
    "SLOPE", "DEM",
)

#: Landslide4Sense 各通道物理含义（用于文档与可视化，不参与计算）
L4S_BAND_ROLE: dict[str, str] = {
    **{b: "S2-L2A" for b in L4S_BANDS[:12]},
    "SLOPE": "ALOS-PALSAR-slope",
    "DEM": "ALOS-PALSAR-dem",
}


class MissingBandError(RuntimeError):
    """所需的 Sentinel-2 波段在源数据中不存在。"""

    def __init__(self, target: str, missing: Sequence[str], available: Sequence[str]):
        self.target = target
        self.missing = list(missing)
        self.available = list(available)
        super().__init__(
            f"[{target}] 缺少必需波段 {self.missing}；"
            f"源数据可用波段为 {self.available}。"
            "按 Landslide-Seg V1 §3.1，禁止补零或近邻波段顶替后声称有效输入。"
        )


def band_index(bands: Sequence[str], name: str) -> int:
    """返回波段名在给定顺序中的下标。"""
    try:
        return list(bands).index(name)
    except ValueError as exc:  # pragma: no cover - 明显错误
        raise KeyError(f"波段 {name!r} 不在 {list(bands)} 中") from exc


def missing_bands(available: Iterable[str], required: Sequence[str]) -> list[str]:
    have = set(available)
    return [b for b in required if b not in have]


# ---------------------------------------------------------------------------
# 输入解析结果
# ---------------------------------------------------------------------------


@dataclass
class ResolvedInput:
    """一次前向所需的全部波段张量，以及它们的来源标注。

    Attributes:
        tensor: ``[C, H, W]`` float32，通道顺序与 ``target_bands`` 一致。
        target_bands: 该张量对应的目标协议波段顺序。
        source_bands: 张量中每个通道实际来自源数据的哪个波段。
        synthetic_bands: 通道名 -> 合成方式说明。空 dict 表示完全来自真实观测。
    """

    tensor: np.ndarray
    target_bands: tuple[str, ...]
    source_bands: tuple[str, ...]
    synthetic_bands: dict[str, str] = field(default_factory=dict)

    @property
    def is_fully_observed(self) -> bool:
        return not self.synthetic_bands

    def provenance(self) -> dict:
        return {
            "target_bands": list(self.target_bands),
            "source_bands": list(self.source_bands),
            "synthetic_bands": dict(self.synthetic_bands),
            "fully_observed": self.is_fully_observed,
        }


def _synthesize_from_neighbor(
    arr: np.ndarray,
    source_bands: Sequence[str],
    name: str,
) -> tuple[np.ndarray, str]:
    """用物理上最接近的可用波段线性近似一个缺失波段（**仅供接口 smoke test**）。

    B8A(865nm, 20m) 与 B08(842nm, 10m) 相邻，取 B08 作为近似；
    这只保证张量形状与分布大致合理，**不构成有效观测**。
    """
    NEIGHBOR = {"B8A": "B08", "B09": "B8A", "B01": "B02", "B10": "B11"}
    ref = NEIGHBOR.get(name)
    if ref is None or ref not in source_bands:
        raise MissingBandError("synthetic", [name], list(source_bands))
    idx = band_index(source_bands, ref)
    return arr[idx].copy(), f"copied_from:{ref}"


def resolve(
    array: np.ndarray,
    source_bands: Sequence[str],
    target_bands: Sequence[str],
    *,
    target_name: str,
    policy: str = "strict",
) -> ResolvedInput:
    """把源张量重排/校验为目标骨干的波段顺序。

    Args:
        array: ``[C, H, W]`` 源张量，通道顺序与 ``source_bands`` 一致。
        source_bands: 源数据的波段顺序。
        target_bands: 目标模型要求的波段顺序。
        target_name: 仅用于报错信息，如 ``"TerraMind-S2L2A"``。
        policy: ``"strict"``（默认，缺失即报错）或 ``"synthetic_smoke"``
            （用近邻波段近似缺失波段，并在返回值中标记）。

    Raises:
        MissingBandError: ``policy="strict"`` 且存在缺失波段时。
    """
    if array.ndim != 3:
        raise ValueError(f"期望 [C,H,W]，实际 {array.shape}")
    if array.shape[0] != len(source_bands):
        raise ValueError(
            f"通道数 {array.shape[0]} 与 source_bands 长度 {len(source_bands)} 不一致"
        )

    gaps = missing_bands(source_bands, target_bands)
    if gaps and policy == "strict":
        raise MissingBandError(target_name, gaps, source_bands)
    if policy not in ("strict", "synthetic_smoke"):
        raise ValueError(f"未知 policy: {policy}")

    channels: list[np.ndarray] = []
    sources: list[str] = []
    synthetic: dict[str, str] = {}

    for name in target_bands:
        if name in source_bands:
            channels.append(array[band_index(source_bands, name)].astype(np.float32))
            sources.append(name)
        else:
            approx, how = _synthesize_from_neighbor(array, source_bands, name)
            channels.append(approx.astype(np.float32))
            sources.append(f"{name}<-{how}")
            synthetic[name] = how

    return ResolvedInput(
        tensor=np.stack(channels, axis=0),
        target_bands=tuple(target_bands),
        source_bands=tuple(sources),
        synthetic_bands=synthetic,
    )


def to_terramind(
    array: np.ndarray,
    source_bands: Sequence[str] = L4S_BANDS,
    *,
    policy: str = "strict",
) -> ResolvedInput:
    """构造 TerraMind ``S2L2A`` 的 12 波段输入。"""
    return resolve(
        array, source_bands, S2_L2A_BANDS, target_name="TerraMind-S2L2A", policy=policy
    )


def to_anysat(
    array: np.ndarray,
    source_bands: Sequence[str] = L4S_BANDS,
    *,
    policy: str = "strict",
) -> ResolvedInput:
    """构造 AnySat ``s2`` 的 10 波段输入。"""
    return resolve(
        array, source_bands, ANY_SAT_S2_BANDS, target_name="AnySat-s2", policy=policy
    )


# ---------------------------------------------------------------------------
# 归一化统计（官方来源）
# ---------------------------------------------------------------------------

#: AnySat 官方 demo notebook 给出的 S2 统计，单位是 **DN = 反射率 × 10000**。
ANY_SAT_S2_MEAN = np.array(
    [4304.3296, 4159.2666, 4057.7766, 4328.9512, 4571.2212,
     4644.8711, 4837.2471, 4700.2578, 2823.2644, 2319.9702],
    dtype=np.float32,
)
ANY_SAT_S2_STD = np.array(
    [3537.9976, 3324.2349, 3270.0701, 3250.5300, 2897.3914,
     2754.4971, 2821.5215, 2625.9524, 1731.5630, 1549.3029],
    dtype=np.float32,
)


def normalize_anysat(s2_dn):
    """按 AnySat 官方统计做逐通道标准化。

    支持 numpy 与 torch，形状 ``[C,H,W]`` 或 ``[B,C,H,W]``（``C=10``），
    返回与输入同类型、同形状。
    """
    if s2_dn.shape[-3] != 10:
        raise ValueError(f"AnySat 需要 10 通道，实际 shape={tuple(s2_dn.shape)}")
    c = s2_dn.shape[-3]
    view = [1] * s2_dn.ndim
    view[-3] = c
    view = tuple(view)
    # 注意：NumPy 2.x 的 ndarray 也有 .device 属性，不能靠它判断类型
    is_torch = type(s2_dn).__module__.split(".")[0] == "torch"
    if is_torch:
        import torch

        mean = torch.as_tensor(ANY_SAT_S2_MEAN, dtype=s2_dn.dtype, device=s2_dn.device).view(view)
        std = torch.as_tensor(ANY_SAT_S2_STD, dtype=s2_dn.dtype, device=s2_dn.device).view(view)
    else:
        mean = ANY_SAT_S2_MEAN.astype(s2_dn.dtype).reshape(view)
        std = ANY_SAT_S2_STD.astype(s2_dn.dtype).reshape(view)
    return (s2_dn - mean) / std


#: TerraMind 官方 S2L2A 统计由 TerraTorch 从权重文件内的 metadata 读取，
#: 这里只保留占位说明，实际值在 encoders/terramind_tim.py 中从 checkpoint 读取并断言。
TERRAMIND_STATS_SOURCE = "terratorch TerramindBackbone (checkpoint 内 data_mean/data_std)"


def estimate_reflectance_scale(array: np.ndarray, source_bands: Sequence[str]) -> dict:
    """估计源数据的辐射量级，用于判断能否直接套用预训练统计。

    返回每个波段的 p50 / p99，以及推断出的量级类型。**不做任何自动换算**，
    只报告事实，由人工决定是否可用。
    """
    out = {"per_band_median": {}, "per_band_p99": {}, "verdict": ""}
    for i, name in enumerate(source_bands):
        band = array[i]
        out["per_band_median"][name] = float(np.median(band))
        out["per_band_p99"][name] = float(np.percentile(band, 99))

    med = np.median([v for v in out["per_band_median"].values()])
    if 0.0 < med <= 1.5:
        out["verdict"] = "反射率量级(0-1)或已缩放"
    elif 500.0 < med <= 10000.0:
        out["verdict"] = "DN 量级(0-10000)"
    else:
        out["verdict"] = f"未知量级(median={med:.4g})"
    return out

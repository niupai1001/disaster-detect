"""把样本组装成模型输入。

这里有两个来源：
1. **真实数据**（Landslide4Sense）：``strict`` 策略下会因为缺少 B8A 直接报错，
   这是被设计成"拒绝执行"的行为，用来证明我们没有偷偷补零；
2. **合成数据**：仅用于接口 smoke test，生成过程完全公开（见
   :func:`make_synthetic_batch`），每条样本都带 ``synthetic=True`` 标记。

两者都不会被混淆：``ModelBatch.provenance`` 明确记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .band_mapping import (
    L4S_BANDS,
    S2_L2A_BANDS,
    ANY_SAT_S2_BANDS,
    to_anysat,
    to_terramind,
)
from .loader import Sample


@dataclass
class ModelBatch:
    x12: torch.Tensor          # [B,12,H,W] DN 尺度
    x10: torch.Tensor          # [B,10,H,W] DN 尺度
    s2_dates: torch.Tensor     # [B,1] 一年内日序
    mask: torch.Tensor         # [B,H,W] {0,1}
    valid: torch.Tensor        # [B,H,W] bool
    sample_ids: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)

    def to(self, device: torch.device | str) -> "ModelBatch":
        for k in ("x12", "x10", "s2_dates", "mask", "valid"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    @property
    def fully_observed(self) -> bool:
        return all(p.get("fully_observed", False) for p in self.provenance)


def batches_from_samples(
    samples: Sequence[Sample],
    *,
    policy: str = "strict",
    dn_scale: float | None = None,
    s2_doy: int = 180,
) -> ModelBatch:
    """从真实样本构建批次。

    Args:
        policy: ``strict`` 或 ``synthetic_smoke``。
        dn_scale: 把存储值换算成 DN 的系数。L4S 官方未公开辐射标定，
            因此 ``None`` 表示"不做换算"，此时数值量级与预训练统计不匹配，
            **不能**用于有效结论。
        s2_doy: 一年内日序。L4S 没有采集日期，只能用占位值，并记录 provenance。
    """
    x12_list, x10_list, masks, valids, provs, ids = [], [], [], [], [], []
    for s in samples:
        tm = to_terramind(s.array, L4S_BANDS, policy=policy)
        an = to_anysat(s.array, L4S_BANDS, policy=policy)
        x12, x10 = tm.tensor, an.tensor
        if dn_scale is not None:
            x12 = x12 * dn_scale
            x10 = x10 * dn_scale
        x12_list.append(x12)
        x10_list.append(x10)
        masks.append(s.mask)
        valids.append(s.valid)
        ids.append(s.meta.sample_id)
        prov = {
            "sample_id": s.meta.sample_id,
            "terramind": tm.provenance(),
            "anysat": an.provenance(),
            "fully_observed": tm.is_fully_observed and an.is_fully_observed,
            "dn_scale": dn_scale,
            "s2_doy": s2_doy,
            "s2_doy_synthetic": True,   # L4S 无日期
        }
        provs.append(prov)

    return ModelBatch(
        x12=torch.from_numpy(np.stack(x12_list)).float(),
        x10=torch.from_numpy(np.stack(x10_list)).float(),
        s2_dates=torch.full((len(x12_list), 1), float(s2_doy)),
        mask=torch.from_numpy(np.stack(masks)).long(),
        valid=torch.from_numpy(np.stack(valids)).bool(),
        sample_ids=ids,
        provenance=provs,
    )


# ---------------------------------------------------------------------------
# 合成数据（仅接口 smoke test）
# ---------------------------------------------------------------------------

#: 合成时使用的、物理上大致合理的反射率均值（仅用于让特征分布不至于离谱）
_SYNTH_REFL = np.array(
    [0.12, 0.10, 0.09, 0.08, 0.12, 0.20, 0.26, 0.30, 0.31, 0.30, 0.20, 0.12],
    dtype=np.float32,
)


def make_synthetic_batch(
    batch_size: int = 2,
    size: int = 128,
    seed: int = 0,
    device: torch.device | str = "cpu",
    with_scar: bool = True,
) -> ModelBatch:
    """生成 12/10 波段合成张量（DN 尺度），用于验证形状、权重加载与梯度通路。

    生成方式（完全公开）：
    - 用低分辨率高斯随机场上采样得到每个波段的空间纹理；
    - 波段 i 的反射率 = 均值_i × (1 + 0.3·纹理) ；
    - 在中心区域人工"挖"一个椭圆疤痕：植被波段下降、SWIR 上升；
    - 乘以 10000 得到 DN（与 TerraMind/AnySat 预训练统计同量级）。

    **这不是 Sentinel-2 观测，不构成任何有效性证据。**
    """
    rng = np.random.default_rng(seed)
    b = batch_size
    x12 = np.zeros((b, 12, size, size), dtype=np.float32)
    mask = np.zeros((b, size, size), dtype=np.int64)

    yy, xx = np.mgrid[0:size, 0:size]
    for k in range(b):
        base = rng.normal(size=(4, 4)).astype(np.float32)
        tex = np.kron(base, np.ones((size // 4, size // 4), dtype=np.float32))
        tex = tex[:size, :size]
        tex = (tex - tex.mean()) / (tex.std() + 1e-6)

        cy, cx = size * (0.35 + 0.3 * rng.random()), size * (0.35 + 0.3 * rng.random())
        ry, rx = size * (0.10 + 0.08 * rng.random()), size * (0.08 + 0.10 * rng.random())
        scar = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) < 1.0
        if with_scar and k % 2 == 0:
            mask[k][scar] = 1

        for i in range(12):
            ref = _SYNTH_REFL[i] * (1 + 0.35 * tex) + rng.normal(0, 0.004, size=(size, size))
            if with_scar and k % 2 == 0:
                # 植被波段(NIR/red-edge)下降，SWIR 上升：模拟裸露滑坡疤痕
                delta = -0.45 if i in (5, 6, 7, 8) else (0.35 if i in (10, 11) else -0.08)
                ref = np.where(scar, ref * (1 + delta), ref)
            x12[k, i] = np.clip(ref, 0.0, 1.0) * 10000.0

    idx10 = [S2_L2A_BANDS.index(n) for n in ANY_SAT_S2_BANDS]
    x10 = x12[:, idx10].copy()
    valid = np.ones((b, size, size), dtype=bool)
    doy = np.full((b, 1), 180.0, dtype=np.float32)

    prov = [
        {
            "sample_id": f"synthetic_{k}",
            "terramind": {"target_bands": list(S2_L2A_BANDS), "synthetic_bands": {},
                          "fully_observed": False},
            "anysat": {"target_bands": list(ANY_SAT_S2_BANDS), "synthetic_bands": {},
                       "fully_observed": False},
            "fully_observed": False,
            "synthetic": True,
            "note": "synthetic interface smoke test, not a Sentinel-2 observation",
            "s2_doy": 180,
            "s2_doy_synthetic": True,
        }
        for k in range(b)
    ]

    return ModelBatch(
        x12=torch.from_numpy(x12).to(device),
        x10=torch.from_numpy(x10).to(device),
        s2_dates=torch.from_numpy(doy).to(device),
        mask=torch.from_numpy(mask).to(device),
        valid=torch.from_numpy(valid).to(device),
        sample_ids=[f"synthetic_{k}" for k in range(b)],
        provenance=prov,
    )


def batches_from_sen12(
    files: Sequence,
    *,
    limit: int | None = None,
    min_mask_px: int = 1,
    keep_negatives: bool = True,
    max_negatives: int | None = None,
    batch_size: int = 1,
) -> list["ModelBatch"]:
    """从 Sen12Landslides 构造真实批次（含 B01/B09 回补）。

    - 正样本：post 时相掩膜非空；
    - 负样本：post 时相掩膜为空（整块无该事件滑坡），**默认保留**，
      否则模型学不到"什么时候该输出空结果"；
    - 未知标签（mask < 0）按无效像素处理，不参与任何监督。
    - 只接受**同一天**能检索到 L2A 产品的样本；否则跳过并记录，不跨日替代。
    """
    from .sen12 import assemble_full, load_sen12

    rows: list[dict] = []
    skipped: list[dict] = []
    n_neg = 0
    for p in (files[:limit] if limit else files):
        s = load_sen12(p)
        a = assemble_full(s)
        if a is None:
            skipped.append({"file": str(p), "reason": "no same-day L2A product"})
            continue
        px = int((a["mask"] > 0).sum())
        if px < min_mask_px:
            if not keep_negatives:
                skipped.append({"file": str(p), "reason": "empty mask (negatives disabled)"})
                continue
            if max_negatives is not None and n_neg >= max_negatives:
                skipped.append({"file": str(p), "reason": "negative quota reached"})
                continue
            n_neg += 1
        rows.append(a)

    if skipped:
        print(f"[sen12] 跳过 {len(skipped)} 个样本，示例: {skipped[:3]}")

    batches: list[ModelBatch] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        masks = []
        for a in chunk:
            m = a["mask"].copy()
            v = a["valid"].copy()
            v &= m >= 0            # 未知标签 -> 无效
            m[m < 0] = 0
            masks.append((m, v))
        batches.append(
            ModelBatch(
                x12=torch.from_numpy(np.stack([a["x12"] for a in chunk])).float(),
                x10=torch.from_numpy(np.stack([a["x10"] for a in chunk])).float(),
                s2_dates=torch.tensor([[float(a["doy"])] for a in chunk]),
                mask=torch.from_numpy(np.stack([m for m, _ in masks])).long(),
                valid=torch.from_numpy(np.stack([v for _, v in masks])).bool(),
                sample_ids=[a["provenance"]["file"] for a in chunk],
                provenance=[a["provenance"] for a in chunk],
            )
        )
    if batches:
        print(f"[sen12] 构造 {len(batches)} 个批次（batch={batch_size}，负样本 {n_neg} 个）")
    return batches

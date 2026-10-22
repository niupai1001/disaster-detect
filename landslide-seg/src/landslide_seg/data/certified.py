"""读取认证后的缓存样本（data/certified/*.npz + manifests/certified.jsonl）。

训练只读这份缓存：不联网、不重新做 STAC 回补，从而保证同一次实验可复现。
缓存里每个样本都带 ``cache_sha256``，训练开始前会校验。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .pipeline import ModelBatch


def read_manifest(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"认证清单不存在: {p}")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def verify_cache(rec: dict) -> None:
    """校验缓存文件哈希，防止清单与缓存不一致。"""
    p = Path(rec["cache"])
    if not p.is_file():
        raise FileNotFoundError(f"缓存缺失: {p}")
    got = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    if got != rec.get("cache_sha256"):
        raise RuntimeError(f"{p} 哈希不匹配: 清单 {rec.get('cache_sha256')} / 实际 {got}")


def load_sample(rec: dict, boa_offset: float = 0.0) -> dict:
    """读缓存。``boa_offset`` 是加到所有波段上的 DN 偏移。

    为什么需要它：2026-10-18 的跨源比对（scripts/verify_radiometry.py）发现
    Sen12 的 DN 比官方当前 L2A 产品系统性低 **1000 DN**（= 0.1 反射率），
    这正是处理基线 04.00 的 ``BOA_ADD_OFFSET = -1000`` 没被加回。
    缓存里保存的是原始 DN（不改数据），偏移在读取时施加，方便做对照实验。
    """
    d = np.load(rec["cache"])
    x12 = d["x12"].astype(np.float32)
    x10 = d["x10"].astype(np.float32)
    # B01/B09 回补窗口在边界会有 0（nodata）。若直接喂 0，模型会把它当成
    # 极端暗值——实测训练后 100% 的高置信预测都堆在这些像素上。
    # 这里用最近邻填补，保留 valid 掩膜不变（监督仍然排除它们）。
    gap = (x12[0] <= 0) | (x12[9] <= 0)
    if gap.any() and (~gap).any():
        from scipy.ndimage import distance_transform_edt

        _, (iy, ix) = distance_transform_edt(gap, return_indices=True)
        for b in (0, 9):
            x12[b] = x12[b][iy, ix]
    if boa_offset:
        x12 = x12 + float(boa_offset)
        x10 = x10 + float(boa_offset)
    return {
        "x12": x12,
        "x10": x10,
        "mask": d["mask"].astype(np.int64),
        "valid": d["valid"].astype(bool),
        "doy": int(d["doy"]),
        "meta": rec,
    }


def make_batches(
    records: Sequence[dict],
    batch_size: int = 2,
    shuffle: bool = False,
    seed: int = 0,
    verify: bool = True,
    boa_offset: float = 0.0,
) -> list[ModelBatch]:
    if verify:
        for r in records:
            verify_cache(r)
    order = list(range(len(records)))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)

    batches: list[ModelBatch] = []
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        rows = [load_sample(records[j], boa_offset=boa_offset) for j in idx]
        masks, valids = [], []
        for r in rows:
            m = r["mask"].copy()
            v = r["valid"].copy()
            v &= m >= 0
            m[m < 0] = 0
            masks.append(m)
            valids.append(v)
        batches.append(
            ModelBatch(
                x12=torch.from_numpy(np.stack([r["x12"] for r in rows])).float(),
                x10=torch.from_numpy(np.stack([r["x10"] for r in rows])).float(),
                s2_dates=torch.tensor([[float(r["doy"])] for r in rows]),
                mask=torch.from_numpy(np.stack(masks)).long(),
                valid=torch.from_numpy(np.stack(valids)).bool(),
                sample_ids=[r["meta"]["sample_id"] for r in rows],
                provenance=[r["meta"] for r in rows],
            )
        )
    return batches


def filter_records(
    records: Iterable[dict],
    split: str | None = None,
    event: str | None = None,
) -> list[dict]:
    out = []
    for r in records:
        if split is not None and r.get("split") != split:
            continue
        if event is not None and r.get("event_id") != event:
            continue
        out.append(r)
    return out

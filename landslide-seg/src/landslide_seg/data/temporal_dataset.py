"""U-TAE 时序配对的双模式数据集（单期 / 15 期），基于 memmap。

设计要点（对应执行计划 §3 C 与 §4）：
- **同一份 memmap、同一批样本、同一顺序**，单期只是取 ``time_index`` 那一帧；
  因此单期组的输入与 15 期组中对应帧逐像素一致（可在验收脚本中核对）。
- ``valid`` 与 ``mask`` 沿用单期缓存，两组评价范围完全相同。
- 归一化用同一份统计（单期 TRAIN 的十波段 + DEM 拟合），两组的 normalizer 完全相同。
- 增强：翻转/90° 旋转**在所有时间帧上同步**施加，影像、mask、valid 一起变换。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import resolve_path

BANDS = 10


class TemporalPairDataset(Dataset):
    def __init__(self, rows: list[dict], stats: dict, *, mode: str = "multi",
                 augment: bool = False, seed: int = 1234, root: str = "data/s2_temporal_v5") -> None:
        if mode not in ("single", "multi"):
            raise ValueError("mode 必须是 single / multi")
        self.rows = list(rows)
        self.stats = stats
        self.mode = mode
        self.augment = augment
        self.seed = int(seed)
        self.epoch = 0
        root_p = resolve_path(root)
        idx = json.loads((root_p / "index.json").read_text(encoding="utf-8"))
        self.pos = {r["sample_id"]: r["row"] for r in idx["rows"]}
        self.X = np.load(root_p / "X15.npy", mmap_mode="r")
        self.DEM = np.load(root_p / "dem.npy", mmap_mode="r")
        missing = [r["sample_id"] for r in self.rows if r["sample_id"] not in self.pos]
        if missing:
            raise KeyError(f"时序缓存缺少 {len(missing)} 个样本，例如 {missing[:3]}")
        self.mean = np.asarray(stats["mean"][:BANDS], dtype=np.float32)[:, None, None]
        self.std = np.maximum(np.asarray(stats["std"][:BANDS], dtype=np.float32)[:, None, None], 1e-6)
        self.dem_mean = float(np.asarray(stats["mean"]).ravel()[BANDS])
        self.dem_std = max(float(np.asarray(stats["std"]).ravel()[BANDS]), 1e-6)
        self.n_channels = BANDS + 1     # 十波段 + DEM（按官方做法广播到每个时相）

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict:
        rec = self.rows[index]
        row = self.pos[rec["sample_id"]]
        x = np.asarray(self.X[row], dtype=np.float32)          # (T,10,H,W)
        dem = np.asarray(self.DEM[row], dtype=np.float32)      # (H,W)
        cache = np.load(resolve_path(rec["cache"]))
        mask = np.asarray(cache["mask"], dtype=np.float32)
        valid = np.asarray(cache["valid"], dtype=bool)

        if self.mode == "single":
            t = int(rec["time_index"])
            x = x[t:t + 1]
        # z-score 十波段 + DEM，并广播 DEM 到每个时相
        x = (x - self.mean[None]) / self.std[None]
        d = ((dem - self.dem_mean) / self.dem_std)[None, None]
        x = np.concatenate([x, np.broadcast_to(d, (x.shape[0], 1, x.shape[2], x.shape[3]))], axis=1)
        x = x.astype(np.float32)

        if self.augment:
            rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 7_919 + index) % (2**63 - 1))
            k = int(rng.integers(0, 4))
            flip = bool(rng.integers(0, 2))
            if flip:
                x = x[:, :, :, ::-1]
                mask = mask[:, ::-1]
                valid = valid[:, ::-1]
            if k:
                x = np.rot90(x, k, axes=(2, 3))
                mask = np.rot90(mask, k, axes=(0, 1))
                valid = np.rot90(valid, k, axes=(0, 1))
            x = np.ascontiguousarray(x)
            mask = np.ascontiguousarray(mask)
            valid = np.ascontiguousarray(valid)

        T = x.shape[0]
        return {"image": torch.from_numpy(x), "mask": torch.from_numpy(mask),
                "valid": torch.from_numpy(valid),
                "dates": torch.arange(T, dtype=torch.float32),
                "sample_id": rec["sample_id"], "event_id": rec["event_id"]}


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in ("image", "mask", "valid", "dates")}
    out["sample_id"] = [b["sample_id"] for b in batch]
    out["event_id"] = [b["event_id"] for b in batch]
    return out

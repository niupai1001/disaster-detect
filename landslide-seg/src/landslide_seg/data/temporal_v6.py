"""日期感知的多帧输入（v6）：真实日期 + 显式缺失标记 + 单一评价范围。

对应计划 §2「输入与评测契约」的修改要求：

1. **真实日期**：日期来自 ``data/s2_temporal_v5/index.json`` 的逐样本 15 个观测日期，
   单位是「距 2015-12-03 的天数」（int32→float32）；**不用帧序号冒充时间间隔**。
   单期模式取 ``time_index`` 那一帧的真实日期；侧表存在时逐样本断言两侧日期一致。
2. **缺失只按显式规则**：``SCL==0``、源数据非有限值、显式有限 ``_FillValue`` 才算缺失；
   云/阴影/雪/未知 SCL 只记录，不当作缺失；缺失像素在 z-score 空间填 0（= TRAIN 均值）。
3. **评价范围不变**：``valid`` 与 ``mask`` 一律沿用单期缓存，多期不扩大也不缩小评价区域。
4. **单期 = 多期中的同一帧**：``single`` 模式只取 ``time_index`` 那一帧，逐像素一致。
5. **增强与历史单期实现一致**：水平翻转 / 垂直翻转 / 90° 旋转，同一随机流作用于
   所有时相、mask 与 valid；随机流只由 ``(seed, epoch, index)`` 决定。

日期基准与侧表构建脚本一致（``DATE_EPOCH``）。返回的 ``dates`` 是**绝对天数**，
需要相对间隔时由模型自己减 anchor；这样单期/多期共享同一时间基准。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import resolve_path

BANDS = 10
#: 日期基准；与侧表构建脚本一致（2015-12-03 为第 0 天）
DATE_EPOCH = np.datetime64("2015-12-03")
DATE_REFERENCE = "2015-12-03"
DATE_UNIT = "days"
MODES = ("single", "multi", "history")


class TemporalV6(Dataset):
    def __init__(self, rows: list[dict], stats: dict, *, mode: str = "single",
                 augment: bool = False, seed: int = 1234, dem: bool = False,
                 root: str = "data/s2_temporal_v5",
                 quality: str = "data/s2_temporal_v6_quality",
                 preload: bool = False) -> None:
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}")
        self.rows = list(rows)
        self.stats = stats
        self.mode = mode
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.dem = bool(dem)
        root_p = resolve_path(root)
        index_bytes = (root_p / "index.json").read_bytes()
        index = json.loads(index_bytes.decode("utf-8"))
        self.pos = {r["sample_id"]: r["row"] for r in index["rows"]}
        self.index_dates = {r["sample_id"]: r["dates"] for r in index["rows"]}
        self.X = np.load(root_p / "X15.npy", mmap_mode="r")
        self.D = np.load(root_p / "dem.npy", mmap_mode="r") if self.dem else None
        # 多期训练从磁盘随机读 64×15 帧会成为瓶颈；显式 opt-in 预载到内存。
        self.preloaded = bool(preload)
        if self.preloaded:
            self.X = np.array(self.X, dtype=np.int16, copy=True)
            if self.D is not None:
                self.D = np.array(self.D, dtype=np.int16, copy=True)
        missing = [r["sample_id"] for r in self.rows if r["sample_id"] not in self.pos]
        if missing:
            raise KeyError(f"时序缓存缺少 {len(missing)} 个样本，例如 {missing[:3]}")

        # 缺失侧表：存在才启用；启用时逐样本核对日期，避免侧表与缓存错位
        self.nodata = None
        self.dem_nodata = None
        self.sidecar = None
        q = resolve_path(quality)
        meta_path = q / "complete.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            got = hashlib.sha256(index_bytes).hexdigest()
            if meta["index_sha256"] != got:
                raise ValueError("v6 质量侧表与当前 index.json 不一致，拒绝使用")
            self.nodata = np.load(q / "nodata.npy", mmap_mode="r")
            self.dem_nodata = np.load(q / "dem_nodata.npy", mmap_mode="r")
            self.sidecar = meta

        self.mean = np.asarray(stats["mean"][:BANDS], dtype=np.float32)[:, None, None]
        self.std = np.maximum(np.asarray(stats["std"][:BANDS], dtype=np.float32)[:, None, None], 1e-6)
        self.dem_mean = float(np.asarray(stats["mean"]).ravel()[BANDS]) if self.dem else 0.0
        self.dem_std = max(float(np.asarray(stats["std"]).ravel()[BANDS]), 1e-6) if self.dem else 1.0
        self.n_channels = BANDS + (1 if self.dem else 0)
        self.date_reference = DATE_REFERENCE
        self.date_unit = DATE_UNIT

        self.labels: list[tuple[np.ndarray, np.ndarray]] = []
        for r in self.rows:
            with np.load(resolve_path(r["cache"])) as z:
                self.labels.append((z["mask"].astype(np.float32), z["valid"].astype(bool)))
            expected = int((np.datetime64(r["date"]) - DATE_EPOCH).astype(int))
            actual = int((np.datetime64(self.index_dates[r["sample_id"]][int(r["time_index"])])
                          - DATE_EPOCH).astype(int))
            if expected != actual:
                raise ValueError(f"{r['sample_id']} 单期清单日期与缓存日期不一致: {expected} vs {actual}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def date_days(self, sample_id: str) -> np.ndarray:
        all_dates = np.asarray(self.index_dates[sample_id], dtype="datetime64[D]")
        return (all_dates - DATE_EPOCH).astype(np.int32)

    def __getitem__(self, i: int) -> dict:
        rec = self.rows[i]
        row = self.pos[rec["sample_id"]]
        anchor = int(rec["time_index"])
        days = self.date_days(rec["sample_id"])

        if self.mode == "single":
            sl = slice(anchor, anchor + 1)
            frame_index = np.array([anchor], dtype=np.int32)
        elif self.mode == "history":
            sl = slice(0, anchor + 1)
            frame_index = np.arange(0, anchor + 1, dtype=np.int32)
        else:
            sl = slice(None)
            frame_index = np.arange(len(days), dtype=np.int32)

        x = np.array(self.X[row, sl], dtype=np.float32)                       # (T,10,H,W)
        x = (x - self.mean[None]) / self.std[None]
        if self.nodata is not None:
            bad = np.asarray(self.nodata[row, sl])                           # (T,H,W)
            if bad.any():
                x = np.where(bad[:, None], 0.0, x)                           # z-score 空间 0 = TRAIN 均值
        if self.dem:
            d = (np.asarray(self.D[row], dtype=np.float32) - self.dem_mean) / self.dem_std
            if self.dem_nodata is not None:
                dbad = np.asarray(self.dem_nodata[row])
                if dbad.any():
                    d = np.where(dbad, 0.0, d)
            x = np.concatenate([x, np.broadcast_to(d[None, None], (x.shape[0], 1, 128, 128))], axis=1)

        mask, valid = self.labels[i]
        if self.augment:
            # 与 S2SegDataset._augment 完全相同的抽取顺序与判据
            rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 7_919 + i) % (2**63 - 1))
            if rng.random() < 0.5:
                x, mask, valid = x[:, :, :, ::-1], mask[:, ::-1], valid[:, ::-1]
            if rng.random() < 0.5:
                x, mask, valid = x[:, :, ::-1, :], mask[::-1], valid[::-1]
            k = int(rng.integers(0, 4))
            if k:
                x = np.rot90(x, k, axes=(2, 3))
                mask = np.rot90(mask, k)
                valid = np.rot90(valid, k)
            x = np.ascontiguousarray(x)
            mask = np.ascontiguousarray(mask)
            valid = np.ascontiguousarray(valid)

        if not np.isfinite(x).all():
            raise FloatingPointError(f"{rec['sample_id']} 输入存在非有限值")
        return {
            "image": torch.from_numpy(x.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "valid": torch.from_numpy(valid),
            "dates": torch.from_numpy(days[frame_index].astype(np.float32)),
            "frame_index": torch.from_numpy(frame_index.astype(np.int32)),
            "time_index": int(anchor),
            "anchor": 0 if self.mode == "single" else int(np.searchsorted(frame_index, anchor)),
            "sample_id": rec["sample_id"],
            "event_id": rec["event_id"],
        }


def collate_v6(items: list[dict]) -> dict:
    n = len(items)
    t_max = max(item["image"].shape[0] for item in items)
    c = items[0]["image"].shape[1]
    image = torch.zeros(n, t_max, c, 128, 128)
    dates = torch.zeros(n, t_max)
    keep = torch.zeros(n, t_max, dtype=torch.bool)
    frame_index = torch.full((n, t_max), -1, dtype=torch.int32)
    for i, item in enumerate(items):
        t = item["image"].shape[0]
        image[i, :t] = item["image"]
        dates[i, :t] = item["dates"]
        keep[i, :t] = True
        frame_index[i, :t] = item["frame_index"]
    return {
        "image": image,
        "dates": dates,
        "keep": keep,
        "frame_index": frame_index,
        "anchor": torch.tensor([item["anchor"] for item in items], dtype=torch.long),
        "time_index": torch.tensor([item["time_index"] for item in items], dtype=torch.long),
        "mask": torch.stack([item["mask"] for item in items]),
        "valid": torch.stack([item["valid"] for item in items]),
        "sample_id": [item["sample_id"] for item in items],
        "event_id": [item["event_id"] for item in items],
    }

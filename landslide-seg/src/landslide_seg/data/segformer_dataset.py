"""SegFormer 单期 10 波段数据集（s2_10band_v1）。

与旧的 12 波段 / 多时相缓存完全隔离：
- 输入 ``[10,128,128]``，波段固定 ``B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12``；
- ``valid`` 由 ``prepare_segformer_data.py`` 按
  "SCL 允许清单 ∩ 波段有限/非 nodata ∩ 标签已知" 直接生成并缓存；
- 标准化统计只来自训练集有效像素（``stats.json``），val/test 复用同一份；
- 增强只有翻转/90° 旋转三类，影像、mask、valid 同步变换，
  随机数流由 ``(seed, epoch, index)`` 决定，与模型初始化无关。

训练/验证/测试都走 ``S2SegDataset`` 这一个类，因此预处理路径只有一条。
``preload=True`` 时把 int16 影像、mask、valid 一次性读进内存（约 0.73 GB / 2,000 样本），
避免每个 epoch 重复解压 npz。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import PROJECT_ROOT

#: 与 Sen12 原生波段一致的固定顺序（不要依赖集合/字典顺序）
S2_10_BANDS: tuple[str, ...] = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
)
#: 可选的附加静态通道；顺序即通道在输入里的位置（接在十波段之后）
#: ``both`` = DEM 在前、坡度在后，与 band_order 的写法保持一致
EXTRA_CHANNELS = ("none", "dem", "slope", "both")
EXTRA_KEYS = {"dem": ("dem",), "slope": ("slope",), "both": ("dem", "slope")}

#: 真彩色在 10 波段里的位置：R=B04(2), G=B03(1), B=B02(0)
RGB_INDEX = (2, 1, 0)

#: 允许的 SCL 类别（与 prepare 脚本保持同一口径）
SCL_VALID = (4, 5, 6, 7)
#: 反过来的表述，仅供阅读/兼容旧引用；新代码请用 ``SCL_VALID``
SCL_INVALID = (0, 1, 2, 3, 8, 9, 10, 11, 255)


def read_manifest(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"清单不存在: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_s2(x10: np.ndarray, stats: dict) -> np.ndarray:
    """逐波段 z-score。``x10``: ``[10,H,W]`` float32（DN）。

    输入产品须先明确辐射口径。仅对所有样本共同的常数偏移有平移不变性，
    train-only z-score 不能消除只发生在部分日期/事件上的辐射偏移。
    """
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    return (x10.astype(np.float32) - mean) / np.maximum(std, 1e-6)


def _augment(
    img: np.ndarray, mask: np.ndarray, valid: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机水平/垂直翻转 + 90° 旋转（k∈{0..3}），三者同步。"""
    if rng.random() < 0.5:
        img, mask, valid = img[..., ::-1], mask[:, ::-1], valid[:, ::-1]
    if rng.random() < 0.5:
        img, mask, valid = img[..., ::-1, :], mask[::-1], valid[::-1]
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k, axes=(1, 2))
        mask = np.rot90(mask, k)
        valid = np.rot90(valid, k)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask), np.ascontiguousarray(valid)


class S2SegDataset(Dataset):
    """懒加载或预载入的 npz 数据集。每个样本返回 float32 影像、float32 mask、bool valid。"""

    def __init__(
        self,
        records: Sequence[dict],
        stats: dict,
        *,
        augment: bool = False,
        seed: int = 1234,
        preload: bool = False,
        extra: str = "none",
        sidecar: str | None = None,
    ) -> None:
        self.records = list(records)
        self.stats = stats
        self.augment = augment
        self.seed = int(seed)
        self.extra = str(extra).lower()
        if self.extra not in EXTRA_CHANNELS:
            raise ValueError(f"extra 必须是 {EXTRA_CHANNELS}，得到 {extra!r}")
        n_stat = len(np.asarray(stats["mean"]).ravel())
        n_expect = len(S2_10_BANDS) + len(EXTRA_KEYS.get(self.extra, ()))
        if n_stat != n_expect:
            raise ValueError(
                f"extra={self.extra!r} 需要 {n_expect} 个通道的统计，但 stats 里有 {n_stat} 个"
            )
        self.epoch = 0
        self._x: list[np.ndarray] | None = None
        self._mask: list[np.ndarray] | None = None
        self._valid: list[np.ndarray] | None = None
        self._extra: list[np.ndarray] | None = None
        # 侧表：原 npz 冻结（哈希登记在协议里）时，DEM/坡度按 sample_id 从独立
        # 数组读取，避免改写缓存或换清单路径。
        self._side: dict | None = None
        self.sidecar_path = sidecar
        if sidecar and self.extra != "none":
            root = Path(sidecar) if Path(sidecar).is_absolute() else (PROJECT_ROOT / sidecar)
            meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
            pos = {sid: i for i, sid in enumerate(meta["rows"])}
            self._side = {
                "root": root,
                "rows": [pos[r["sample_id"]] for r in self.records],
                "dem": np.load(root / "dem.npy", mmap_mode="r"),
                "slope": np.load(root / "slope.npy", mmap_mode="r"),
                "keys": EXTRA_KEYS[self.extra],
            }
        if preload:
            self._preload()

    def _side_stack(self, index: int) -> np.ndarray:
        assert self._side is not None
        row = self._side["rows"][index]
        parts = []
        for key in self._side["keys"]:
            if key == "dem":
                parts.append(np.asarray(self._side["dem"][row], dtype=np.int16))
            elif key == "slope":
                parts.append(np.asarray(self._side["slope"][row], dtype=np.int16))
            else:  # pragma: no cover - EXTRA_KEYS 目前只有 dem/slope
                raise KeyError(key)
        return np.stack(parts, axis=0)

    def _preload(self) -> None:
        xs, masks, valids, extras = [], [], [], []
        for rec in self.records:
            d = np.load(rec["cache"])
            xs.append(np.asarray(d["x10"], dtype=np.int16))
            masks.append(np.asarray(d["mask"], dtype=np.uint8))
            valids.append(np.asarray(d["valid"], dtype=bool))
            if self.extra != "none":
                if self._side is not None:
                    extras.append(self._side_stack(len(xs) - 1))
                else:
                    keys = EXTRA_KEYS[self.extra]
                    missing = [k for k in keys if k not in d]
                    if missing:
                        raise KeyError(f"{rec['cache']} 缺少额外通道 {missing}")
                    arr = np.stack([np.asarray(d[k], dtype=np.int16) for k in keys], axis=0)
                    extras.append(arr if len(keys) > 1 else arr[0])
        self._x, self._mask, self._valid = xs, masks, valids
        self._extra = extras if self.extra != "none" else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        if self._x is not None:
            raw = self._x[index]
            mask = self._mask[index].astype(np.float32) if self._mask is not None else None
            valid = self._valid[index] if self._valid is not None else None
            ext = self._extra[index] if self._extra is not None else None
            if mask is None or valid is None:  # pragma: no cover - 结构保证
                raise RuntimeError("preload 状态不完整")
        else:
            d = np.load(rec["cache"])
            raw = d["x10"]
            mask = d["mask"].astype(np.float32)
            valid = d["valid"].astype(bool)
            ext = (np.stack([np.asarray(d[k], dtype=np.int16) for k in EXTRA_KEYS[self.extra]], axis=0)
                   if (self.extra != "none" and self._side is None) else None)
            if self.extra != "none" and self._side is not None:
                ext = self._side_stack(index)

        if ext is not None:
            # 额外通道接在十波段之后；统计量已按同一顺序扩展
            ext = np.atleast_3d(ext) if ext.ndim == 2 else ext
            raw = np.concatenate([np.asarray(raw, dtype=np.int16), ext.astype(np.int16)], axis=0)
        x10 = normalize_s2(raw.astype(np.float32), self.stats)

        if self.augment:
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + self.epoch * 7_919 + index) % (2**63 - 1)
            )
            x10, mask, valid = _augment(x10, mask, valid, rng)

        return {
            "image": torch.from_numpy(x10),
            "mask": torch.from_numpy(mask),
            "valid": torch.from_numpy(valid),
            "sample_id": rec["sample_id"],
            "event_id": rec["event_id"],
        }


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "valid": torch.stack([b["valid"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "event_id": [b["event_id"] for b in batch],
    }


def filter_split(records: Sequence[dict], split: str) -> list[dict]:
    return [r for r in records if r.get("split") == split]

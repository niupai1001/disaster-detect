"""Landslide4Sense 读取与样本元数据。

数据事实（实测，见 docs/data_contract.md）：
- 影像 h5: key ``img``，``(128,128,14)`` float64；
- 掩膜 h5: key ``mask``，``(128,128)`` {0,1}；
- 14 通道 = S2 ``B01..B12`` + ``SLOPE`` + ``DEM``，**无 B8A、有 B10**；
- **没有**地理坐标、采集日期、产品 ID、云掩膜 -> 无法回溯原始 L2A 补齐 B8A。

因此本模块只负责"如实读取 + 如实标注缺失"，不做任何补齐。
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np

from .band_mapping import L4S_BANDS, ResolvedInput, estimate_reflectance_scale

SPLITS = ("train", "validation", "test")


@dataclass
class SampleMeta:
    """训练/评测必须留存的数据清单字段（Landslide-Seg V1 §3.3）。"""

    sample_id: str
    split: str
    source_uri: str
    bands_present: list[str]
    # 以下字段 L4S 无法提供，保持 None 并在数据契约中说明，不做猜测
    scene_id: str | None = None
    event_id: str | None = None
    date: str | None = None
    bbox: list[float] | None = None
    crs: str | None = None
    cloud_fraction: float | None = None
    mask_coverage: float = 0.0
    label_status: str = "official-train-labels"
    usable_for_full_model: bool = False
    unusable_reason: str = ""


@dataclass
class Sample:
    array: np.ndarray          # [14,H,W] float32，原始单位（未标准化）
    mask: np.ndarray           # [H,W] int64，{0,1}
    valid: np.ndarray          # [H,W] bool，可用像素
    meta: SampleMeta
    extras: dict = field(default_factory=dict)


class L4SDataset:
    """Landslide4Sense 数据集（只读，懒加载）。"""

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "train",
        limit: int | None = None,
        indices: Sequence[int] | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split 必须是 {SPLITS}，实际 {split}")
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images" / split
        self.msk_dir = self.root / "annotations" / split
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"影像目录不存在: {self.img_dir}")

        ids = sorted(
            int(p.stem.split("_")[1])
            for p in self.img_dir.glob("image_*.h5")
        )
        if indices is not None:
            keep = set(int(i) for i in indices)
            ids = [i for i in ids if i in keep]
        if limit is not None:
            ids = ids[:limit]
        self.ids = ids
        self.stats = estimate_reflectance_scale(
            self._probe_stack(), L4S_BANDS
        ) if ids else {}

    # -- 内部 ---------------------------------------------------------------
    def _probe_stack(self, n: int = 4) -> np.ndarray:
        frames = []
        for i in self.ids[:n]:
            with h5py.File(self.img_dir / f"image_{i}.h5", "r") as f:
                frames.append(f["img"][:].transpose(2, 0, 1).astype(np.float32))
        return np.concatenate(frames, axis=1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, k: int) -> Sample:
        sid = self.ids[k]
        img_path = self.img_dir / f"image_{sid}.h5"
        msk_path = self.msk_dir / f"mask_{sid}.h5"

        with h5py.File(img_path, "r") as f:
            arr = f["img"][:].transpose(2, 0, 1).astype(np.float32)  # [14,H,W]
        with h5py.File(msk_path, "r") as f:
            mask = f["mask"][:].astype(np.int64)

        if arr.shape[0] != len(L4S_BANDS):
            raise ValueError(
                f"{img_path} 通道数 {arr.shape[0]} != {len(L4S_BANDS)}"
            )
        if mask.shape != arr.shape[1:]:
            raise ValueError(f"掩膜 {mask.shape} 与影像 {arr.shape[1:]} 不匹配")

        valid = self.valid_mask(arr)
        meta = SampleMeta(
            sample_id=f"l4s_{self.split}_{sid}",
            split=self.split,
            source_uri=str(img_path).replace("\\", "/"),
            bands_present=list(L4S_BANDS),
            mask_coverage=float((mask > 0).mean()),
            label_status="official-train-labels",
            # 12 波段 TiM 缺 B8A；10 波段 AnySat 同样缺 B8A -> 不能进入正式训练
            usable_for_full_model=False,
            unusable_reason="missing B8A for TerraMind S2L2A(12) and AnySat s2(10); "
                            "L4S 无场景地理/日期信息，无法回溯原始 L2A 补齐",
        )
        return Sample(array=arr, mask=mask, valid=valid, meta=meta)

    @staticmethod
    def valid_mask(arr: np.ndarray) -> np.ndarray:
        """可用像素掩膜。

        L4S 官方**未提供**云/阴影掩膜，因此这里只做最低限度的物理有效性检查：
        有限值、非全零。**不能**据此宣称已排除云遮挡，数据契约中明确记录。
        """
        finite = np.isfinite(arr).all(axis=0)
        nonzero = (arr > 0).any(axis=0)
        return finite & nonzero

    def __iter__(self) -> Iterator[Sample]:
        for k in range(len(self)):
            yield self[k]


def write_manifest(samples: Sequence[Sample], path: str | os.PathLike) -> None:
    """把样本元数据写成 CSV，便于审计与复现。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s.meta) for s in samples]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["bands_present"] = "|".join(r["bands_present"])
            for k in ("bbox",):
                r[k] = "" if r[k] is None else "|".join(map(str, r[k]))
            w.writerow(r)

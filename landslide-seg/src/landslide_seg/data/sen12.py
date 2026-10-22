"""Sen12Landslides（Sentinel-2 分支）读取器。

为什么需要它：Landslide4Sense 缺 B8A、无地理信息，无法满足 TerraMind/AnySat 的波段协议；
Sen12Landslides 的 S2 文件经实测包含 **B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12**
（正好是 AnySat 官方 10 波段），并附带：

- `SCL`：Sentinel-2 场景分类（可派生云/阴影/无效像素掩膜）
- `MASK`：{0,1} 滑坡掩膜
- `DEM`、`x/y`、`crs`（EPSG）、`time`（days since 某日）、`event_date`、`pre_post`
- 数值为 **int16 DN（0–10000 量级）**，与两个骨干的预训练统计同量级

仍缺 B01/B09（TiM 必需的 60m 波段），因此 TiM 分支需要按 crs+坐标+日期回源补齐。
本模块对此保持严格：缺就是缺。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .band_mapping import ANY_SAT_S2_BANDS, S2_L2A_BANDS, MissingBandError, missing_bands

#: Sen12 文件里实际存在的 S2 波段（与 AnySat 官方 10 波段一致）
SEN12_S2_BANDS: tuple[str, ...] = ANY_SAT_S2_BANDS

#: SCL 中"不可用作有效观测"的类别：
#: 0 无数据 / 1 饱和 / 2 地形阴影 / 3 云影 / 8 中概率云 / 9 高概率云 / 10 薄云 / 11 雪
SCL_INVALID = (0, 1, 2, 3, 8, 9, 10, 11)


@dataclass
class Sen12Sample:
    """一个 Sen12 图块（含 15 个时相）。"""

    path: Path
    bands: dict[str, np.ndarray]          # name -> [T,H,W] int16 (DN)
    mask: np.ndarray                      # [T,H,W] uint8 {0,1}
    scl: np.ndarray                       # [T,H,W] int16
    dem: np.ndarray                       # [T,H,W] int16
    time: np.ndarray                      # [T] int (days since epoch)
    attrs: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return int(self.mask.shape[0])

    def event_index(self) -> tuple[int, int]:
        """返回 (灾前索引, 灾后索引)。解析失败时退回 (0, T-1)。"""
        pp = self.attrs.get("pre_post")
        try:
            if isinstance(pp, dict) and "pre" in pp and "post" in pp:
                return int(pp["pre"]), int(pp["post"])
        except Exception:
            pass
        return 0, self.num_steps - 1

    def doy(self, index: int) -> int:
        """把 `days since reference` 换成 AnySat 要求的一年内日序。

        AnySat 官方约定（README）：**01/01 = 0，31/12 = 364**，即 0-based。
        Python 的 ``tm_yday`` 是 1-based（1..365），必须减 1，否则整年偏一天。
        """
        import datetime as _dt

        ref = self.attrs.get("_time_ref")
        if ref is None:
            return 180
        d = ref + _dt.timedelta(days=int(self.time[index]))
        return int(d.timetuple().tm_yday) - 1

    # ------------------------------------------------------------------
    def date_at(self, index: int):
        """该时相的实际采集日期；无时间基准时返回 None。"""
        import datetime as _dt

        ref = self.attrs.get("_time_ref")
        if ref is None:
            return None
        return ref + _dt.timedelta(days=int(self.time[index]))

    def cloud_fraction(self, index: int) -> float:
        """SCL 派生的云+阴影占比（2/3/8/9/10）。"""
        return float(np.isin(self.scl[index], (2, 3, 8, 9, 10)).mean())

    def select_observation(self, max_cloud: float = 0.5) -> tuple[int | None, dict]:
        """挑选一个**灾后且最清晰**的时相。

        为什么不能直接用 ``pre_post['post']``：
        实测发现很多样本的 post 时相整幅被云覆盖（italy 某样本 SCL=9 占 100%），
        而同一景更早/更晚的灾后时相是晴空。Sen12 的掩膜是**事件滑坡编目**、
        对所有时相相同，因此任何 ``date >= event_date`` 的观测都满足"灾后单期观测"，
        在这些候选里取云量最低的一个，既符合任务定义，也不构成标签替换。

        Returns:
            (index, info)。无可用时相时 index=None，info 里给出原因。
        """
        import datetime as _dt

        info: dict = {"candidates": [], "chosen": None, "reason": ""}
        ed = self.attrs.get("event_date")
        if not ed:
            info["reason"] = "missing event_date"
            return None, info
        try:
            event_dt = _dt.date.fromisoformat(str(ed)[:10])
        except Exception:
            info["reason"] = f"unparsable event_date {ed!r}"
            return None, info
        if self.attrs.get("_time_ref") is None:
            info["reason"] = "missing time reference"
            return None, info

        cands = []
        for i in range(self.num_steps):
            d = self.date_at(i)
            if d is None or d < event_dt:
                continue
            c = self.cloud_fraction(i)
            cands.append({"index": i, "date": d.isoformat(), "cloud": round(c, 4)})
        info["candidates"] = cands
        if not cands:
            info["reason"] = "no post-event timestep"
            return None, info

        # 云量最低优先；并列时取离事件日期最近的（更有针对性）
        cands.sort(key=lambda r: (r["cloud"], _dt.date.fromisoformat(r["date"])))
        best = cands[0]
        if best["cloud"] > max_cloud:
            info["reason"] = f"all post-event timesteps too cloudy (best {best['cloud']:.0%})"
            return None, info
        info["chosen"] = best
        return int(best["index"]), info

    def valid_mask(self, index: int) -> np.ndarray:
        """由 SCL 派生有效像素（True = 可用观测）。"""
        scl = self.scl[index]
        ok = np.ones_like(scl, dtype=bool)
        for code in SCL_INVALID:
            ok &= scl != code
        return ok

    def as_model_inputs(self, index: int | None = None) -> dict:
        """构造模型输入。

        Returns:
            dict with
              - ``x10``: ``[10,H,W]`` float32 DN（AnySat 可直接使用）
              - ``mask``: ``[H,W]`` int64
              - ``valid``: ``[H,W]`` bool（由 SCL 派生）
              - ``doy``: int
              - ``meta``: dict

        Raises:
            MissingBandError: 请求 TiM 的 12 波段时（本数据缺 B01/B09）。
        """
        if index is None:
            index = self.event_index()[1]
        x10 = np.stack([self.bands[b][index] for b in SEN12_S2_BANDS]).astype(np.float32)
        gaps = missing_bands(SEN12_S2_BANDS, S2_L2A_BANDS)
        meta = {
            "path": str(self.path).replace("\\", "/"),
            "index": index,
            "doy": self.doy(index),
            "event_date": self.attrs.get("event_date"),
            "crs": self.attrs.get("crs"),
            "center": (self.attrs.get("center_x"), self.attrs.get("center_y")),
            "bands_present": list(SEN12_S2_BANDS),
            "missing_for_terramind": gaps,
        }
        return {
            "x10": x10,
            "mask": self.mask[index].astype(np.int64),
            "valid": self.valid_mask(index),
            "doy": self.doy(index),
            "meta": meta,
        }

    def assert_terramind_ready(self) -> None:
        gaps = missing_bands(SEN12_S2_BANDS, S2_L2A_BANDS)
        if gaps:
            raise MissingBandError("TerraMind-S2L2A", gaps, list(SEN12_S2_BANDS))


def _parse_pp(value) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    try:
        import ast

        v = ast.literal_eval(str(value))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def load_sen12(path: str | os.PathLike) -> Sen12Sample:
    """读取一个 ``*_s2_*.nc`` 文件。"""
    import datetime as _dt

    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        bands = {b: f[b][:] for b in SEN12_S2_BANDS if b in f}
        missing = [b for b in SEN12_S2_BANDS if b not in bands]
        if missing:
            raise KeyError(f"{path.name} 缺少预期波段: {missing}")
        mask = f["MASK"][:]
        scl = f["SCL"][:] if "SCL" in f else np.zeros_like(mask, dtype=np.int16)
        dem = f["DEM"][:] if "DEM" in f else np.zeros_like(mask, dtype=np.int16)
        time = f["time"][:]
        attrs = {k: v for k, v in f.attrs.items()}
        units = f["time"].attrs.get("units", b"")
        if isinstance(units, bytes):
            units = units.decode("utf-8", "ignore")
        attrs["_time_units"] = units
        # 解析 "days since YYYY-MM-DD"
        try:
            ref = units.split("since")[-1].strip()
            attrs["_time_ref"] = _dt.date.fromisoformat(ref)
        except Exception:
            attrs["_time_ref"] = None

    for k in ("event_date", "pre_post_dates", "crs", "satellite", "annotated"):
        if k in attrs and isinstance(attrs[k], bytes):
            attrs[k] = attrs[k].decode("utf-8", "ignore")
    attrs["pre_post"] = _parse_pp(attrs.get("pre_post_dates", ""))
    # 注意：Sen12 的 attrs 命名与实际含义相反 ——
    # ``center_lat`` 存的是 **northing**（投影坐标），``center_lon`` 存的是 **easting**。
    cx = attrs.get("center_lon")   # easting
    cy = attrs.get("center_lat")   # northing
    attrs["center_x"] = float(np.ravel(cx)[0]) if cx is not None else None
    attrs["center_y"] = float(np.ravel(cy)[0]) if cy is not None else None

    return Sen12Sample(
        path=path,
        bands=bands,
        mask=mask,
        scl=scl,
        dem=dem,
        time=time,
        attrs=attrs,
    )


def list_sen12_files(root: str | os.PathLike, limit: int | None = None) -> list[Path]:
    files = sorted(Path(root).rglob("*_s2_*.nc"))
    return files[:limit] if limit else files


def describe(sample: Sen12Sample) -> dict:
    """返回可用于数据清单的统计（不推测未知字段）。"""
    pre, post = sample.event_index()
    m = sample.mask[post]
    v = sample.valid_mask(post)
    return {
        "file": sample.path.name,
        "steps": sample.num_steps,
        "crs": sample.attrs.get("crs"),
        "event_date": sample.attrs.get("event_date"),
        "pre_post": {"pre": pre, "post": post},
        "doy_post": sample.doy(post),
        "mask_positive_px": int(m.sum()),
        "valid_ratio": float(v.mean()),
        "band_dtype": str(next(iter(sample.bands.values())).dtype),
        "band_median": {b: int(np.median(a[post])) for b, a in sample.bands.items()},
    }


def assemble_full(
    sample: Sen12Sample,
    index: int | None = None,
    *,
    require_same_day: bool = True,
) -> dict | None:
    """组装完整 12 波段输入（Sen12 的 10 波段 + STAC 回补的 B01/B09）。

    Returns:
        dict(x12, x10, mask, valid, doy, provenance) 或 ``None``（同一天无产品）。
    """
    from .s2_backfill import backfill_b01_b09

    if index is None:
        index, _sel = sample.select_observation()
        if index is None:
            return None      # 没有可用的灾后清晰观测
    bf = backfill_b01_b09(sample, index, require_same_day=require_same_day)
    if bf is None:
        return None

    x12 = np.zeros((12, *sample.mask.shape[1:]), dtype=np.float32)
    for i, name in enumerate(S2_L2A_BANDS):
        if name in bf.bands:
            x12[i] = bf.bands[name]
        else:
            x12[i] = sample.bands[name][index].astype(np.float32)

    x10 = np.stack([sample.bands[b][index] for b in SEN12_S2_BANDS]).astype(np.float32)
    valid = sample.valid_mask(index).copy()
    gap = (bf.bands["B01"] <= 0) | (bf.bands["B09"] <= 0)
    valid &= ~gap

    return {
        "x12": x12,
        "x10": x10,
        "mask": sample.mask[index].astype(np.int64),
        "valid": valid,
        "doy": sample.doy(index),
        "provenance": {
            "file": sample.path.name,
            "index": index,
            "event_date": sample.attrs.get("event_date"),
            "crs": sample.attrs.get("crs"),
            "backfill_item": bf.item_id,
            "backfill_datetime": bf.datetime,
            "backfill_cloud_cover": bf.cloud_cover,
            "backfill_method": bf.method,
            "backfill_gap_ratio": float(gap.mean()),
            "synthetic_bands": {},          # 无任何合成波段
            "fully_observed": True,         # B01/B09 来自同一 L2A 产品，不是合成值
            "resampled_bands": ["B01", "B09"],   # 60m -> 10m，已注明
        },
    }

"""从公开 STAC（AWS Earth Search）回补 B01/B09，构造完整的 12 波段 L2A 输入。

为什么可以这么做：
- Sen12Landslides 的 S2 文件提供了 ``crs`` + ``x/y`` + 每个时相的确切日期；
- 据此可以在 STAC 中检索到**同一采集日期**的 L2A 产品；
- B01/B09 与该产品的其余波段来自同一次过境，属于同一观测，不是"邻近日期凑数"。

约束（Landslide-Seg V1 §3.2）：
- 必须**日期严格相等**；找不到就返回 None 并记录，绝不跨日替代；
- B01/B09 原生 60 m，重采样到 10 m **不增加真实细节**，需在 provenance 中注明；
- 只补 TiM 需要的这两个波段，不修改 Sen12 提供的 10 个波段。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

STAC_URL = os.environ.get("STAC_URL", "https://earth-search.aws.element84.com/v1")

#: STAC asset 名 -> 我们的波段名
ASSET_MAP = {"coastal": "B01", "nir09": "B09"}


@dataclass
class BackfillResult:
    bands: dict[str, np.ndarray]      # {"B01": [H,W], "B09": [H,W]} float32 DN
    item_id: str
    datetime: str
    cloud_cover: float | None
    source_res_m: float
    method: str


def _stac_search(lon: float, lat: float, date: dt.date, window_days: int = 0) -> list[dict]:
    start = date - dt.timedelta(days=window_days)
    end = date + dt.timedelta(days=window_days)
    q = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 20,
        "query": {"eo:cloud_cover": {"lt": 95}},
    }
    req = urllib.request.Request(
        f"{STAC_URL}/search",
        data=json.dumps(q).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get("features", [])


def find_same_day_item(lon: float, lat: float, date: dt.date) -> Optional[dict]:
    """只在**同一天**检索；多个候选时取云量最低的。"""
    feats = _stac_search(lon, lat, date, window_days=0)
    same = [f for f in feats if f["properties"]["datetime"][:10] == date.isoformat()]
    if not same:
        return None
    same.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 999))
    return same[0]


def read_asset_window(
    href: str,
    bounds_utm: tuple[float, float, float, float],
    dst_crs: str,
    out_hw: tuple[int, int],
    transpose: bool = True,
) -> np.ndarray:
    """把 60m 的 COG 重采样到目标 10m 网格。

    Args:
        transpose: Sen12Landslides 的 S2 数组按 **(x, y)** 索引存储（与其 netCDF
            的 x/y 维定义相反），因此与 STAC 原始窗口相比需要转置才能对齐。
            实测依据：对 B02/B03/B04/B08/B11/B12/B8A 七个波段，转置后与 STAC 的
            相关系数为 0.972–0.996，不转置时为 −0.19–0.28。设为 False 仅用于诊断。
    """
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    xmin, ymin, xmax, ymax = bounds_utm
    h, w = out_hw
    with rasterio.open(href) as src:
        # STAC 的 href 是整景 COG，按目标 bbox 取窗口
        win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        data = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        src_transform = src.window_transform(win)
        src_crs = src.crs
        nodata = src.nodata

    dst = np.zeros((h, w), dtype=np.float32)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=_transform_from_bounds(xmin, ymin, xmax, ymax, h, w),
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata,
        dst_nodata=0,
    )
    return dst.T if transpose else dst


def _transform_from_bounds(xmin, ymin, xmax, ymax, h, w):
    from rasterio.transform import from_bounds

    return from_bounds(xmin, ymin, xmax, ymax, w, h)


def backfill_b01_b09(sample, index: int, require_same_day: bool = True) -> Optional[BackfillResult]:
    """为一个 Sen12 样本回补 B01/B09。

    Args:
        sample: ``Sen12Sample``
        index: 时相索引（与 ``sample.bands[..][index]`` 对齐）
        require_same_day: True 时若同一天没有产品则返回 None（默认，禁止跨日替代）
    """
    import datetime as _dt

    from pyproj import Transformer

    if sample.attrs.get("_time_ref") is None:
        raise RuntimeError("样本缺少时间基准，无法确定采集日期")
    date = sample.attrs["_time_ref"] + _dt.timedelta(days=int(sample.time[index]))

    crs = sample.attrs.get("crs", "EPSG:32736")
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(sample.attrs["center_x"], sample.attrs["center_y"])

    item = find_same_day_item(lon, lat, date)
    if item is None:
        if require_same_day:
            return None
        raise RuntimeError("未启用跨日替代；同一天无产品即视为不可用")

    # 目标网格：由 Sen12 的 x/y 推出 bbox
    from pyproj import Transformer as _T

    to_utm = _T.from_crs("EPSG:4326", crs, always_xy=True)
    resolution = 10.0          # Sen12 图块固定 10 m/px
    half = 128 * resolution / 2
    cx, cy = sample.attrs["center_x"], sample.attrs["center_y"]
    bounds = (cx - half, cy - half, cx + half, cy + half)

    bands = {}
    for asset, name in ASSET_MAP.items():
        a = item["assets"].get(asset)
        if a is None:
            raise KeyError(f"STAC item 缺少 asset {asset}")
        bands[name] = read_asset_window(a["href"], bounds, crs, (128, 128))

    return BackfillResult(
        bands=bands,
        item_id=item["id"],
        datetime=item["properties"]["datetime"],
        cloud_cover=item["properties"].get("eo:cloud_cover"),
        source_res_m=60.0,
        method="rasterio.reproject(bilinear, 60m->10m)",
    )

"""Shared machinery for the product adapters.

A patch is read straight from the remote cloud-optimised GeoTIFFs: only the
bytes covering the patch are transferred, and every band is resampled onto the
same 10 m target grid so the model sees one geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds


@dataclass
class Scene:
    product: str
    scene_id: str
    datetime: str
    epsg: int
    assets: dict = field(default_factory=dict)   # band key -> url
    header: str = ""                             # GDAL header file, "" if none
    extra: dict = field(default_factory=dict)    # product metadata (baseline, xml, ...)


def remote_env(header: str = ""):
    opts = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            # without these a stalled CDN waits forever instead of failing
            "GDAL_HTTP_TIMEOUT": "60",
            "GDAL_HTTP_CONNECTTIMEOUT": "20",
            "GDAL_HTTP_MAX_RETRY": "2",
            "GDAL_HTTP_RETRY_DELAY": "3"}
    if header:
        opts["GDAL_HTTP_HEADER_FILE"] = str(header)
    return opts


def read_grid(url, dst_transform, dst_crs, shape, header="", resampling=Resampling.bilinear):
    """Read the part of a remote raster that covers the destination grid."""
    height, width = shape
    dst_bounds = rasterio.transform.array_bounds(height, width, dst_transform)
    with rasterio.Env(**remote_env(header)):
        with rasterio.open(url) as src:
            left, bottom, right, top = transform_bounds(dst_crs, src.crs, *dst_bounds,
                                                        densify_pts=21)
            win = from_bounds(left, bottom, right, top, src.transform)
            win = win.round_offsets().round_lengths()
            col0 = max(0, int(win.col_off))
            row0 = max(0, int(win.row_off))
            col1 = min(src.width, int(win.col_off + win.width))
            row1 = min(src.height, int(win.row_off + win.height))
            if col1 - col0 < 1 or row1 - row0 < 1:
                return None
            win = Window(col0, row0, col1 - col0, row1 - row0)
            data = src.read(1, window=win)
            dst = np.zeros(shape, dtype=data.dtype)
            reproject(
                source=data,
                destination=dst,
                src_transform=src.window_transform(win),
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )
    return dst


def scene_raster_bounds(scene: Scene, key=None):
    """Bounds and CRS of the actual raster, needed to clip the patch grid."""
    key = key or next(iter(scene.assets))
    with rasterio.Env(**remote_env(scene.header)):
        with rasterio.open(scene.assets[key]) as src:
            return src.bounds, src.crs


def patch_grids(bbox_wgs84, bounds, crs, size=512, gsd=10):
    """All patch transforms covering the AOI ∩ scene footprint, snapped to gsd.

    Patching only the intersection is what keeps every transferred pixel usable:
    a scene edge strip is never filled up with invented data.
    """
    from pyproj import Transformer

    if not hasattr(bounds, "left"):          # accept a plain 4-tuple as well
        bounds = rasterio.coords.BoundingBox(*bounds)
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox_wgs84,
                                                densify_pts=21)
    left, bottom = max(left, bounds.left), max(bottom, bounds.bottom)
    right, top = min(right, bounds.right), min(top, bounds.top)
    if right - left < gsd or top - bottom < gsd:
        return []
    step = size * gsd
    x0 = np.ceil(left / step) * step
    y0 = np.ceil(bottom / step) * step
    out = []
    x = x0
    while x + step <= right + 1e-6:
        y = y0
        while y + step <= top + 1e-6:
            out.append(rasterio.transform.from_origin(x, y + step, gsd, gsd))
            y += step
        x += step
    return out


def read_patch(scene: Scene, keys, dst_transform, dst_crs, size=512):
    """Read the requested bands onto one square grid.

    Returns (arrays, band_metadata_rows, geo_metadata). Reflectance is scaled to
    float32 surface reflectance; QA stays integer and uses nearest resampling.
    """
    from .. import band_catalog

    bands = band_catalog.stack(scene.product, keys)
    arrays, scales = {}, {}
    for b in bands:
        url = scene.assets.get(b.key)
        if not url:
            raise KeyError(f"{scene.product}: scene has no asset {b.key}")
        if b.kind == "qa":
            arr = read_grid(url, dst_transform, dst_crs, (size, size),
                            header=scene.header, resampling=Resampling.nearest)
        else:
            arr = read_grid(url, dst_transform, dst_crs, (size, size),
                            header=scene.header, resampling=Resampling.bilinear)
        if arr is None:
            return None
        arrays[b.key] = arr

    image, valid, meta = [], None, []
    for b in bands:
        arr = arrays[b.key]
        if b.kind == "qa":
            meta.append({"key": b.key, "name": b.name, "kind": "qa",
                         "native_gsd_m": b.native_gsd_m, "resampled_to_m": 10})
            continue
        scaled = scale_reflectance(scene.product, b.key, arr, scene.extra)
        image.append(scaled)
        meta.append({"key": b.key, "name": b.name, "centre_nm": b.centre_nm,
                     "kind": "reflectance", "native_gsd_m": b.native_gsd_m,
                     "resampled_to_m": 10})
    if not image:
        return None

    from .. import qa_masks

    qa_layers = {b.key: arrays[b.key] for b in bands if b.kind == "qa"}
    valid = qa_masks.valid_mask(scene.product, qa_layers)

    geo = {"product": scene.product, "scene_id": scene.scene_id,
           "datetime": scene.datetime, "crs": f"EPSG:{dst_crs.to_epsg()}",
           "transform": list(dst_transform)[:6], "size": size,
           "gsd_m": 10, "bands": [m["name"] for m in meta]}
    return (np.stack(image).astype(np.float32), valid, meta, geo)


def scale_reflectance(product, key, dn, extra):
    """DN -> surface reflectance with the product's own rule."""
    dn = dn.astype(np.float32)
    if product == "sentinel2":
        q = extra.get("quantification", 10000.0)
        offset = extra.get("boa_offset", 0.0)
        return (dn + offset) / q
    if product == "landsat":
        return dn * 0.0000275 - 0.2
    if product == "hls":
        return dn * extra.get("hls_scale", 0.0001) + extra.get("hls_offset", 0.0)
    if product == "enmap":
        return dn * extra.get("enmap_scale", 1.0) + extra.get("enmap_offset", 0.0)
    raise KeyError(f"no scaling rule for {product}")

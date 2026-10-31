"""Build patch manifests by reading AOI windows straight from the COGs.

A patch exists only if the scene actually covers that ground and enough of it
is cloud/shadow/snow free at the patch's own location — catalogue cloud cover
is a scene-level number and says nothing about a single 5 km patch.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rasterio.crs import CRS

from . import qa_masks
from .adapters import _base, hls_v2, landsat_c2, sentinel2

ADAPTERS = {
    "sentinel2": (sentinel2, ("SCL",)),
    "landsat": (landsat_c2, ("qa_pixel", "qa_radsat")),
    "hls": (hls_v2, ("Fmask",)),
}


def _qa_key(product):
    return {"sentinel2": "SCL", "landsat": "qa_pixel", "hls": "Fmask"}[product]


def patch_valid_fraction(scene, transform, crs, product, size=512):
    """Cheap validity probe: read only the QA layer for one patch.

    A single unreadable asset must not abort the region, so any failure here is
    reported as "this patch is unusable".
    """
    key = _qa_key(product)
    try:
        arr = _base.read_grid(scene.assets[key], transform, crs, (size, size),
                              header=scene.header,
                              resampling=_base.Resampling.nearest)
    except Exception:
        return 0.0
    if arr is None:
        return 0.0
    if product == "sentinel2":
        return float(np.mean(~np.isin(arr, list(qa_masks.S2_SCL_BAD.values()))))
    if product == "hls":
        bad = sum(1 << b for b in (0, 1, 2, 3, 4))
        return float(np.mean((arr & bad) == 0))
    qa = arr
    bad = sum(1 << b for b in (0, 1, 2, 3, 4, 5))
    return float(np.mean((qa & bad) == 0))


def build_region(region, bbox, product, start, end, size=512, gsd=10,
                 min_valid=0.30, max_patches=0, cloud_max=30):
    """Manifest rows for one region and one product over a date window."""
    adapter, _ = ADAPTERS[product]
    scenes = adapter.scenes(bbox, start, end, limit=100, cloud_max=cloud_max)
    if not scenes:
        return [], {"region": region, "product": product, "scenes": 0,
                    "reason": "no scene in window"}, {}
    scenes.sort(key=lambda s: (s.extra.get("cloud_cover", 100), s.datetime))
    scenes = scenes[:3]          # the three clearest candidates are enough
    rows, used = [], 0
    for scene in scenes:
        if not scene.epsg:
            continue
        # the raster's own CRS is authoritative: the MGRS tile letter does not
        # always match the stored UTM hemisphere
        bounds, crs = _base.scene_raster_bounds(scene)
        grids = _base.patch_grids(bbox, bounds, crs, size=size, gsd=gsd)
        # QA probes are pure network waits: probe in chunks, stop once satisfied,
        # so a large scene is not scanned to its last tile.
        chunk = 12
        for start in range(0, len(grids), chunk):
            part = grids[start:start + chunk]
            with ThreadPoolExecutor(max_workers=8) as pool:
                fracs = list(pool.map(
                    lambda t: patch_valid_fraction(scene, t, crs, product, size), part))
            for j, (transform, frac) in enumerate(zip(part, fracs)):
                if frac < min_valid:
                    continue
                i = start + j
                rows.append({
                    "patch_id": f"{region}_{product}_{scene.scene_id}_{i:04d}",
                    "region": region,
                    "product": product,
                    "scene_id": scene.scene_id,
                    "datetime": scene.datetime,
                    "crs": f"EPSG:{crs.to_epsg()}",
                    "transform": [round(v, 3) for v in list(transform)[:6]],
                    "size": size,
                    "gsd_m": gsd,
                    "valid_fraction": round(frac, 4),
                    "qa_rule": qa_masks.lookalike_rule(product),
                })
                used += 1
                if max_patches and used >= max_patches:
                    break
            if max_patches and used >= max_patches:
                break
        if max_patches and used >= max_patches:
            break
    return rows, {"region": region, "product": product,
                  "scenes": len(scenes), "patches": used}, \
        {s.scene_id: s for s in scenes}


def save_patch(scene, row, outdir, product):
    """Read one manifest row and store it as npz (image, valid, band names)."""
    from affine import Affine
    from rasterio.crs import CRS

    keys = {"sentinel2": sentinel2.KEYS, "landsat": landsat_c2.KEYS,
            "hls": hls_v2.KEYS}[product]
    _, crs = _base.scene_raster_bounds(scene)
    out = _base.read_patch(scene, keys, Affine(*row["transform"]), crs,
                           size=row["size"])
    if out is None:
        return None
    image, valid, meta, geo = out
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{row['patch_id']}.npz"
    reflectance = [m["name"] for m in meta if m["kind"] == "reflectance"]
    np.savez_compressed(path, image=image, valid=valid,
                        bands=np.array(reflectance))
    return path





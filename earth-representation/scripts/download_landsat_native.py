"""Re-cut Landsat companions on the native 30 m grid - no resampling.

The earlier companions were reprojected onto the S2 10 m grid, which invents
detail Landsat does not have. This keeps the scene's own pixels: the patch bbox
is snapped to the Landsat grid and only whole native pixels inside it are read,
so a 1280 m patch becomes 43x43 (or 42) at 30 m.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import Window
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402
from src.data.adapters import landsat_c2  # noqa: E402

UA = {"User-Agent": "landsat-native/1.0"}


def native_window(ds_crs, ds_transform, bbox, crs):
    """Grid-aligned window covering the bbox, at native resolution."""
    from rasterio.warp import transform_bounds
    b = transform_bounds(crs, ds_crs, *bbox, densify_pts=21) if str(crs) != str(ds_crs) else bbox
    left = ds_transform.c
    top = ds_transform.f
    px = ds_transform.a
    col0 = int(np.floor((b[0] - left) / px))
    col1 = int(np.ceil((b[2] - left) / px))
    row0 = int(np.floor((top - b[3]) / px))
    row1 = int(np.ceil((top - b[1]) / px))
    return Window(col0, row0, max(1, col1 - col0), max(1, row1 - row0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "data" / "landsat_native"))
    args = ap.parse_args(argv)

    samples = load_samples()
    ids = sorted(samples)[: args.limit]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for sid in ids:
        m = samples[sid]
        import datetime as _dt
        d0 = _dt.date.fromisoformat(m["date"])
        scenes = landsat_c2.scenes(_wgs84(m),
                                   (d0 - _dt.timedelta(days=30)).isoformat(),
                                   (d0 + _dt.timedelta(days=30)).isoformat(), limit=20)
        if not scenes:
            print(f"{sid}: no scene near {m['date']}")
            continue
        def _gap(sc):
            d = sc.datetime
            d = d.date() if hasattr(d, "date") else _dt.date.fromisoformat(str(d)[:10])
            return abs((d - d0).days)
        scene = min(scenes, key=_gap)
        bands = []
        shape = None
        with rasterio.Env(GDAL_HTTP_HEADER_FILE=scene.header):
            for key in ("red", "green", "blue", "nir08", "swir16", "swir22", "coastal"):
                asset = scene.assets.get(key)
                if not asset:
                    continue
                with rasterio.open(asset) as ds:
                    win = native_window(ds.crs, ds.transform, m["bbox"], m["crs"])
                    a = ds.read(1, window=win)
                    bands.append(a)
                    shape = a.shape
        if not bands:
            print(f"{sid}: no bands")
            continue
        cube = np.stack(bands).astype(np.int16)
        dst = outdir / f"{sid}.nc"
        with netcdf_file(str(dst), "w", version=2) as nc:
            nc.createDimension("band", cube.shape[0])
            nc.createDimension("y", cube.shape[1])
            nc.createDimension("x", cube.shape[2])
            v = nc.createVariable("sr", cube.dtype.char, ("band", "y", "x"))
            v[:] = cube
            setattr(nc, "native_gsd_m", np.int32(30))
            setattr(nc, "resampled", b"False")
            setattr(nc, "scene_id", str(getattr(scene, "scene_id", "")).encode())
            setattr(nc, "bands", b",".join(b.encode() for b in ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22")))
        manifest.append({"sample_id": sid, "shape": list(cube.shape), "file": str(dst)})
        print(f"{sid}: {cube.shape} native 30 m -> {dst.name}")

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def _wgs84(m):
    from rasterio.warp import transform_bounds
    return list(transform_bounds(m["crs"], "EPSG:4326", *m["bbox"], densify_pts=21))


if __name__ == "__main__":
    raise SystemExit(main())

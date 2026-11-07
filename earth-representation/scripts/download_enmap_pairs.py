"""Cut EnMAP L2A cubes onto the existing S2/Landsat patch footprints.

Pairing rule: same ground location, any EnMAP acquisition (the archive starts in
2022 while most of the paired events are 2016-2019, so a same-date pair is
impossible for them). Each output file is 224 bands x 128 x 128 at the patch
bbox, mirroring the 128x128 companion layout.

Reads one union window per 0.1 degree cell, then cuts every patch in the cell -
one remote open per cell instead of one per patch.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _common as C  # noqa: E402
from download_companion import load_samples  # noqa: E402
from src.data.adapters import _base  # noqa: E402

UA = {"User-Agent": "enmap-pairs/1.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
OUT = ROOT / "data" / "enmap_pairs"
SIZE = 128


def login(session, href, user, password):
    import re
    import urllib.parse
    lp = session.get(href, headers=UA, timeout=120)
    if "login" not in lp.url and "sso" not in lp.url:
        return
    blk = re.search(r'<form[^>]*id="fm1"[^>]*>(.*?)</form>', lp.text, re.S)
    if not blk:
        raise RuntimeError("EOC login form not found")
    fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', blk.group(1)))
    for n in re.findall(r'<input[^>]*name="([^"]+)"', blk.group(1)):
        fields.setdefault(n, "")
    fields.update({"username": user, "password": password})
    r = session.post(urllib.parse.urljoin(lp.url, "login"), data=fields,
                     headers={**UA, "Referer": lp.url}, timeout=120)
    if "login" in r.url:
        raise RuntimeError("EOC login failed")


def scene_for(bbox_wgs84, cache):
    """Any EnMAP scene whose footprint covers this bbox centre."""
    lon = (bbox_wgs84[0] + bbox_wgs84[2]) / 2
    lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2
    key = (round(lon / 0.5), round(lat / 0.5))
    if key not in cache:
        doc = requests.post(f"{STAC}/search", json={
            "collections": ["ENMAP_HSI_L2A"],
            "bbox": [(key[0] - 0.3) * 0.5, (key[1] - 0.3) * 0.5,
                     (key[0] + 0.3) * 0.5, (key[1] + 0.3) * 0.5],
            "datetime": "2022-04-27T00:00:00Z/2026-12-31T23:59:59Z",
            "limit": 50}, timeout=120, headers=UA).json()
        cache[key] = doc.get("features", [])
    hits = [f for f in cache[key]
            if f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3]]
    if not hits:
        return None
    # the S2/Landsat pairs predate the EnMAP archive, so the closest possible
    # acquisition is the earliest one covering the point
    return min(hits, key=lambda f: str(f["properties"].get("datetime", "")))


def local_scene(session, feat, cache_dir):
    """Fetch the whole spectral cube once per scene (GDAL cannot auth here)."""
    url = feat["assets"]["image"]["href"]
    dst = Path(cache_dir) / (url.rsplit("/", 1)[-1])
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with session.get(url, headers=UA, stream=True, timeout=3600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(8 << 20):
                fh.write(chunk)
    tmp.replace(dst)
    print(f"    cached scene {dst.name[-40:]} {dst.stat().st_size/1e6:.0f} MB")
    return dst


def read_union(session, feat, bbox_wgs84, cache_dir):
    """Read all bands of one scene over the union bbox, at native 30 m."""
    url = str(local_scene(session, feat, cache_dir))
    left, bottom, right, top = bbox_wgs84
    # target grid in the scene CRS
    import rasterio
    with rasterio.open(url) as ds:
        crs = ds.crs
        a = transform_bounds("EPSG:4326", crs, left, bottom, right, top, densify_pts=21)
        width = max(16, int(round((a[2] - a[0]) / 30.0)))
        height = max(16, int(round((a[3] - a[1]) / 30.0)))
        grid = from_bounds(a[0], a[1], a[2], a[3], width, height)
        data = ds.read(window=rasterio.windows.from_bounds(
            *a, ds.transform).round_offsets().round_lengths(),
            out_shape=(ds.count, height, width),
            resampling=Resampling.bilinear)
    return data, grid, crs


def cut(data, grid, crs, bbox, patch_crs):
    dst = np.zeros((data.shape[0], SIZE, SIZE), dtype=data.dtype)
    dst_grid = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], SIZE, SIZE)
    for i in range(data.shape[0]):
        reproject(source=data[i], destination=dst[i], src_transform=grid,
                  src_crs=crs, dst_transform=dst_grid, dst_crs=patch_crs,
                  resampling=Resampling.bilinear)
    return dst


def write_nc(path, cube, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(str(path), "w", version=2) as nc:
        nc.createDimension("band", cube.shape[0])
        nc.createDimension("y", cube.shape[1])
        nc.createDimension("x", cube.shape[2])
        var = nc.createVariable("spectral", cube.dtype.char, ("band", "y", "x"))
        var[:] = cube
        for k, v in attrs.items():
            setattr(nc, k, v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    e = C.load_env()
    samples = load_samples()
    ids = sorted(samples)[: args.limit]
    print(f"patches: {len(ids)}")

    cells = defaultdict(list)
    for sid in ids:
        m = samples[sid]
        w = transform_bounds(m["crs"], "EPSG:4326", *m["bbox"], densify_pts=21)
        cells[(round((w[0] + w[2]) / 2 / 0.1), round((w[1] + w[3]) / 2 / 0.1))].append((sid, m, w))
    print(f"cells: {len(cells)}")

    session = requests.Session()
    cache = {}
    manifest = []
    done = 0
    for key, items in cells.items():
        w0 = min(i[2][0] for i in items), min(i[2][1] for i in items)
        w1 = max(i[2][2] for i in items), max(i[2][3] for i in items)
        feat = scene_for([w0[0], w0[1], w1[0], w1[1]], cache)
        if feat is None:
            print(f"  cell {key}: no scene")
            continue
        try:
            login(session, feat["assets"]["image"]["href"], e["ENMAP_USER"], e["ENMAP_PASS"])
        except Exception as exc:
            print("login failed:", exc)
            return 1
        # authenticated rasterio reads need the cookie as a header file
        cookie = "; ".join(f"{c.name}={c.value}" for c in session.cookies)
        hdr = Path(args.out).parent / ".tmp" / "enmap_header.txt"
        hdr.parent.mkdir(parents=True, exist_ok=True)
        hdr.write_text(f"Cookie: {cookie}\n", encoding="utf-8")
        try:
            data, grid, crs = read_union(session, feat, [w0[0], w0[1], w1[0], w1[1]],
                                         cache_dir=Path(args.out) / "_scenes")
        except Exception as exc:
            print(f"  cell {key}: read failed {type(exc).__name__}: {exc}")
            continue
        print(f"  cell {key}: scene {feat['id'][:28]} union {data.shape}")
        for sid, m, w in items:
            patch_crs = CRS.from_epsg(int(m["crs"].split(":")[1]))
            cube = cut(data, grid, crs, m["bbox"], patch_crs)
            dst = Path(args.out) / f"{sid}.nc"
            write_nc(dst, cube.astype(np.int16), {
                "source": "ENMAP_HSI_L2A", "scene_id": feat["id"],
                "sensing_date": feat["properties"].get("datetime", ""),
                "s2_sample_id": sid, "bands": f"{cube.shape[0]} (all L2A bands)",
                "native_gsd_m": 30, "resampled_to_m": 10,
            })
            manifest.append({"sample_id": sid, "scene": feat["id"],
                             "date": feat["properties"].get("datetime", ""),
                             "file": str(dst), "shape": list(cube.shape)})
            done += 1
        print(f"    wrote {len(items)} patches (total {done})")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("done:", done)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())

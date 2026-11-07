"""Cut EnMAP patches scene by scene, deleting each scene after use.

Measured layout of the L2A COG: tiles are 512x512 with all 224 bands inside one
tile (planar=1, ZSTD), so a single 128 px patch still drags a 20-83 MB tile.
Per-patch range reads are therefore *more* traffic than one 440 MB scene shared
by every patch that falls inside it. This groups patches by scene, downloads
once, cuts, then removes the file.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402
from scripts.enmap_download_eoc import login  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
OUT = ROOT / "data" / "enmap_pairs"
CACHE = ROOT / "data" / "enmap_scenes"
SIZE = 128


def env():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def scenes_for(bbox, samples_bbox=None):
    doc = requests.post(f"{STAC}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": bbox,
        "datetime": "2022-04-27T00:00:00Z/2026-12-31T23:59:59Z", "limit": 100},
        timeout=120, headers=UA).json()
    return doc.get("features", [])


def assign(samples, ids):
    """Map each patch to the earliest scene whose footprint contains its centre."""
    out = {}
    for sid in ids:
        m = samples[sid]
        w = transform_bounds(m["crs"], "EPSG:4326", *m["bbox"], densify_pts=21)
        lon, lat = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        feats = scenes_for([lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01])
        hits = [f for f in feats
                if f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3]]
        if hits:
            best = min(hits, key=lambda f: str(f["properties"].get("datetime", "")))
            out.setdefault(best["id"], {"feat": best, "items": []})["items"].append(sid)
    return out


def fetch_scene(session, feat):
    url = feat["assets"]["image"]["href"]
    dst = CACHE / (url.rsplit("/", 1)[-1])
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
    return dst


def cut_scene(path, samples, ids, outdir):
    written = 0
    with rasterio.open(path) as ds:
        for sid in ids:
            m = samples[sid]
            patch_crs = CRS.from_epsg(int(m["crs"].split(":")[1]))
            win = rasterio.windows.from_bounds(*m["bbox"], ds.transform).round_offsets().round_lengths()
            win = rasterio.windows.Window(win.col_off, win.row_off,
                                          max(1, win.width), max(1, win.height))
            data = ds.read(window=win)
            if data.size == 0:
                continue
            src_grid = rasterio.windows.transform(win, ds.transform)
            dst = np.zeros((data.shape[0], SIZE, SIZE), dtype=np.int16)
            dst_grid = from_bounds(*m["bbox"], SIZE, SIZE)
            for i in range(data.shape[0]):
                reproject(source=data[i], destination=dst[i], src_transform=src_grid,
                          src_crs=ds.crs, dst_transform=dst_grid, dst_crs=patch_crs,
                          resampling=Resampling.bilinear)
            with netcdf_file(str(outdir / f"{sid}.nc"), "w", version=2) as nc:
                nc.createDimension("band", dst.shape[0])
                nc.createDimension("y", SIZE)
                nc.createDimension("x", SIZE)
                v = nc.createVariable("spectral", dst.dtype.char, ("band", "y", "x"))
                v[:] = dst
                setattr(nc, "source", b"ENMAP_HSI_L2A")
                setattr(nc, "native_gsd_m", np.int32(30))
                setattr(nc, "resampled_to_m", np.int32(10))
                setattr(nc, "scene", Path(path).name.encode())
            written += 1
    return written


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--keep-scenes", action="store_true")
    args = ap.parse_args(argv)

    e = env()
    samples = load_samples()
    ids = sorted(samples)[: args.limit]
    groups = assign(samples, ids)
    print(f"patches {len(ids)} -> scenes {len(groups)}")

    session = requests.Session()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (scene_id, g) in enumerate(groups.items()):
        if i >= args.scenes:
            break
        feat = g["feat"]
        items = g["items"]
        try:
            login(session, feat["assets"]["image"]["href"], e["ENMAP_USER"], e["ENMAP_PASS"])
            path = fetch_scene(session, feat)
            print(f"  {scene_id[:26]}: {path.stat().st_size/1e6:.0f} MB, {len(items)} patches")
            n = cut_scene(path, samples, items, OUT)
            print(f"    wrote {n}")
            for sid in items:
                manifest.append({"sample_id": sid, "scene": scene_id})
            if not args.keep_scenes:
                path.unlink()
        except Exception as exc:
            print(f"  {scene_id[:26]}: failed {type(exc).__name__}: {str(exc)[:80]}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

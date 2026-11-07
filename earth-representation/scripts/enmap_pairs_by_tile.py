"""Scheme C: fetch only the COG tiles that a patch needs.

Layout measured on the L2A product: 512x512 tiles, all 224 bands inside one tile
(planar=1, ZSTD). A 128 px patch touches one tile, so pulling that tile's byte
range (20-83 MB) and decoding it serves every patch in the tile - instead of a
432 MB scene, and instead of 224 requests per patch.
"""

import argparse
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import requests
import tifffile
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from scipy.io import netcdf_file

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402
from scripts.enmap_cog_read import HttpRangeFile  # noqa: E402
from scripts.enmap_download_eoc import login  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
OUT = ROOT / "data" / "enmap_pairs"
CACHE = ROOT / "data" / "enmap_tiles"
SIZE = 128


def env():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def scene_epsg(page):
    try:
        keys = page.tags["GeoKeyDirectoryTag"].value
        for i in range(4, len(keys), 4):
            if keys[i] == 3072:          # ProjectedCSTypeGeoKey
                return int(keys[i + 3])
            if keys[i] == 2048:          # GeographicTypeGeoKey
                return int(keys[i + 3])
    except Exception:
        pass
    return 32632


def assign(samples, ids, cell=0.25):
    """One STAC query per cell instead of one per patch."""
    from collections import defaultdict as _dd
    cells = _dd(list)
    for sid in ids:
        m = samples[sid]
        w = transform_bounds(m["crs"], "EPSG:4326", *m["bbox"], densify_pts=21)
        lon, lat = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        cells[(round(lon / cell), round(lat / cell))].append((sid, lon, lat))
    out = {}
    for (cx, cy), items in cells.items():
        feats = []
        for attempt in range(4):
            try:
                feats = requests.post(f"{STAC}/search", json={
                    "collections": ["ENMAP_HSI_L2A"],
                    "bbox": [(cx - 1) * cell, (cy - 1) * cell, (cx + 1) * cell, (cy + 1) * cell],
                    "datetime": "2022-04-27T00:00:00Z/2026-12-31T23:59:59Z",
                    "limit": 200}, timeout=120, headers=UA).json().get("features", [])
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        for sid, lon, lat in items:
            hits = [f for f in feats
                    if f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3]]
            if not hits:
                continue
            best = min(hits, key=lambda f: str(f["properties"].get("datetime", "")))
            out.setdefault(best["id"], {"feat": best, "items": []})["items"].append(sid)
    return out


def cut_from_tile(tile, origin, epsg, bbox, crs_patch, nbands=224):
    """Reproject the patch window out of a decoded tile.

    tifffile returns the tile as (rows, cols, bands) for some pages and as
    (bands, rows, cols) for others, so the band axis is located by size rather
    than assumed.
    """
    if tile.shape[-1] != nbands and tile.shape[0] == nbands:
        tile = np.transpose(tile, (1, 2, 0))
    if tile.shape[-1] != nbands:
        raise ValueError(f"unexpected tile shape {tile.shape}")
    dst = np.zeros((nbands, SIZE, SIZE), dtype=np.int16)
    src_grid = from_origin(origin[0], origin[1], 30, 30)
    dst_grid = from_bounds(*bbox, SIZE, SIZE)
    for i in range(nbands):
        reproject(source=tile[:, :, i], destination=dst[i], src_transform=src_grid,
                  src_crs=CRS.from_epsg(epsg), dst_transform=dst_grid, dst_crs=crs_patch,
                  resampling=Resampling.bilinear)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--keep-tiles", action="store_true")
    args = ap.parse_args(argv)

    e = env()
    samples = load_samples()
    ids = sorted(samples)[: args.limit]
    groups = assign(samples, ids)
    print(f"patches {len(ids)} -> {len(groups)} scenes", flush=True)

    session = requests.Session()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = []

    for idx, (scene_id, g) in enumerate(groups.items()):
        if idx >= args.scenes:
            break
        feat = g["feat"]
        url = feat["assets"]["image"]["href"]
        login(session, url, e["ENMAP_USER"], e["ENMAP_PASS"])
        try:
            fobj = HttpRangeFile(url, session)
            tf = tifffile.TiffFile(fobj)
        except Exception as exc:
            print(f"  {scene_id[:24]}: open failed {type(exc).__name__}: {str(exc)[:60]}", flush=True)
            continue
        page = tf.pages[0]
        epsg = scene_epsg(page)
        tie = page.tags["ModelTiepointTag"].value
        ox, oy = tie[3], tie[4]
        tw = page.tilewidth
        across = (page.imagewidth + tw - 1) // tw

        # which tiles do these patches need?
        need = defaultdict(list)
        for sid in g["items"]:
            if (OUT / f"{sid}.nc").exists():
                continue
            m = samples[sid]
            tr = Transformer.from_crs(CRS.from_epsg(int(m["crs"].split(":")[1])),
                                      CRS.from_epsg(epsg), always_xy=True)
            xs, ys = tr.transform([m["bbox"][0], m["bbox"][2]], [m["bbox"][1], m["bbox"][3]])
            cols = [int((x - ox) // 30) for x in xs]
            rows = [int((oy - y) // 30) for y in ys]
            for r in range(min(rows) // tw, max(rows) // tw + 1):
                for c in range(min(cols) // tw, max(cols) // tw + 1):
                    need[(r, c)].append(sid)

        print(f"  {scene_id[:24]}: {len(g['items'])} patches need {len(need)} tiles", flush=True)
        for (r, c), sids in need.items():
            seg = r * across + c
            if seg >= len(page.dataoffsets):
                continue
            fobj.seek(page.dataoffsets[seg])
            raw = fobj.read(page.databytecounts[seg])
            decoded = page.decode(raw, seg)
            tile = decoded[0] if isinstance(decoded, tuple) else decoded
            origin = (ox + c * tw * 30, oy - r * tw * 30)
            # geometry guard: every patch centre must fall inside this tile,
            # otherwise the tile indexing is wrong and the output would be a
            # patch cut from the wrong part of the scene
            span = tw * 30
            for sid in sids:
                m = samples[sid]
                tr_ = Transformer.from_crs(CRS.from_epsg(int(m["crs"].split(":")[1])),
                                           CRS.from_epsg(epsg), always_xy=True)
                cx_, cy_ = tr_.transform((m["bbox"][0] + m["bbox"][2]) / 2,
                                         (m["bbox"][1] + m["bbox"][3]) / 2)
                if not (origin[0] - span <= cx_ <= origin[0] + span
                        and origin[1] - span <= cy_ <= origin[1] + span):
                    print(f"    WARN tile {r},{c}: {sid} centre outside tile", flush=True)
            for sid in sids:
                m = samples[sid]
                cube = cut_from_tile(tile, origin, epsg, m["bbox"],
                                     CRS.from_epsg(int(m["crs"].split(":")[1])))
                with netcdf_file(str(OUT / f"{sid}.nc"), "w", version=2) as nc:
                    nc.createDimension("band", cube.shape[0])
                    nc.createDimension("y", SIZE)
                    nc.createDimension("x", SIZE)
                    v = nc.createVariable("spectral", cube.dtype.char, ("band", "y", "x"))
                    v[:] = cube
                    setattr(nc, "source", b"ENMAP_HSI_L2A")
                    setattr(nc, "native_gsd_m", np.int32(30))
                    setattr(nc, "resampled_to_m", np.int32(10))
                    setattr(nc, "scene", scene_id[:60].encode())
                    setattr(nc, "tile", f"{r},{c}".encode())
                manifest.append({"sample_id": sid, "scene": scene_id, "tile": [r, c]})
            print(f"    tile {r},{c}: {len(sids)} patches "
                  f"(fetched {fobj.bytes_read/1e6:.0f} MB total)", flush=True)

        (OUT / "manifest_tile.json").write_text(json.dumps(manifest, indent=2),
                                                encoding="utf-8")
    print(f"wrote {len(manifest)} patches", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

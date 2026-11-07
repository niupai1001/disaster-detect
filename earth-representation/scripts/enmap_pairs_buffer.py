"""Scheme C, corrected: paste the needed tiles into one buffer, then cut patches.

The earlier version cut each patch from the single tile that contained its
centre. Patches whose footprint straddles a tile boundary (17 of 40 in one
measured scene) were then cut from the wrong pixels. Here the tiles a scene
needs are fetched once, pasted onto their true positions, and every patch is cut
from that buffer - so boundaries are handled by construction, not by luck.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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
SIZE = 128
GSD = 30


def env():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def assign(samples, ids, cell=0.25):
    cells = defaultdict(list)
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
                import time
                time.sleep(3 * (attempt + 1))
        for sid, lon, lat in items:
            hits = [f for f in feats
                    if f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3]]
            if not hits:
                continue
            best = min(hits, key=lambda f: str(f["properties"].get("datetime", "")))
            out.setdefault(best["id"], {"feat": best, "items": []})["items"].append(sid)
    return out


def scene_epsg(page):
    try:
        keys = page.tags["GeoKeyDirectoryTag"].value
        for i in range(4, len(keys), 4):
            if keys[i] in (2048, 3072):
                return int(keys[i + 3])
    except Exception:
        pass
    return 32632


def cut_from_buffer(buf, origin, epsg, bbox, crs_patch):
    dst = np.zeros((buf.shape[2], SIZE, SIZE), dtype=np.int16)
    src_grid = from_origin(origin[0], origin[1], GSD, GSD)
    dst_grid = from_bounds(*bbox, SIZE, SIZE)
    for i in range(buf.shape[2]):
        reproject(source=buf[:, :, i], destination=dst[i], src_transform=src_grid,
                  src_crs=CRS.from_epsg(epsg), dst_transform=dst_grid, dst_crs=crs_patch,
                  resampling=Resampling.bilinear)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--scenes", type=int, default=3)
    args = ap.parse_args(argv)

    e = env()
    samples = load_samples()
    ids = [s for s in sorted(samples) if not (OUT / f"{s}.nc").exists()][: args.limit]
    if not ids:
        print("nothing pending", flush=True)
        return 0
    groups = assign(samples, ids)
    print(f"pending {len(ids)} -> {len(groups)} scenes", flush=True)

    session = requests.Session()
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    manifest = []

    for gi, (scene_id, g) in enumerate(groups.items()):
        if gi >= args.scenes:
            break
        url = g["feat"]["assets"]["image"]["href"]
        try:
            login(session, url, e["ENMAP_USER"], e["ENMAP_PASS"])
            fobj = HttpRangeFile(url, session)
            tf = tifffile.TiffFile(fobj)
        except Exception as exc:
            print(f"  {scene_id[:24]}: open failed {type(exc).__name__}", flush=True)
            continue
        page = tf.pages[0]
        epsg = scene_epsg(page)
        tie = page.tags["ModelTiepointTag"].value
        ox, oy = tie[3], tie[4]
        tw = page.tilewidth
        across = (page.imagewidth + tw - 1) // tw

        # which tiles do these patches touch, and over what tile block?
        need, rows, cols = set(), [], []
        for sid in g["items"]:
            m = samples[sid]
            tr = Transformer.from_crs(CRS.from_epsg(int(m["crs"].split(":")[1])),
                                      CRS.from_epsg(epsg), always_xy=True)
            xs, ys = tr.transform([m["bbox"][0], m["bbox"][2]],
                                  [m["bbox"][1], m["bbox"][3]])
            cs = [int((x - ox) // GSD) for x in xs]
            rs = [int((oy - y) // GSD) for y in ys]
            for r in range(min(rs) // tw, max(rs) // tw + 1):
                for c in range(min(cs) // tw, max(cs) // tw + 1):
                    need.add((r, c))
            rows += rs
            cols += cs
        r0, r1 = min(r for r, _ in need), max(r for r, _ in need)
        c0, c1 = min(c for _, c in need), max(c for _, c in need)
        bh, bw = (r1 - r0 + 1) * tw, (c1 - c0 + 1) * tw
        print(f"  {scene_id[:24]}: {len(g['items'])} patches, tiles {len(need)} "
              f"block {r1-r0+1}x{c1-c0+1}", flush=True)
        if len(need) > 6:
            print("    block too large, skipping this scene", flush=True)
            continue
        buf = np.full((bh, bw, page.shape[-1]), -32768, dtype=np.int16)
        for (r, c) in sorted(need):
            seg = r * across + c
            if seg >= len(page.dataoffsets):
                continue
            fobj.seek(page.dataoffsets[seg])
            raw = fobj.read(page.databytecounts[seg])
            decoded = page.decode(raw, seg)
            tile = decoded[0] if isinstance(decoded, tuple) else decoded
            tile = np.asarray(tile)
            if tile.ndim == 4:
                tile = tile[0]
            if tile.shape[-1] != page.shape[-1] and tile.shape[0] == page.shape[-1]:
                tile = np.transpose(tile, (1, 2, 0))
            yy = (r - r0) * tw
            xx = (c - c0) * tw
            buf[yy:yy + tw, xx:xx + tw, :] = tile[:tw, :tw, :]

        origin = (ox + c0 * tw * GSD, oy - r0 * tw * GSD)
        for sid in g["items"]:
            m = samples[sid]
            cube = cut_from_buffer(buf, origin, epsg, m["bbox"],
                                   CRS.from_epsg(int(m["crs"].split(":")[1])))
            with netcdf_file(str(OUT / f"{sid}.nc"), "w", version=2) as nc:
                nc.createDimension("band", cube.shape[0])
                nc.createDimension("y", SIZE)
                nc.createDimension("x", SIZE)
                v = nc.createVariable("spectral", cube.dtype.char, ("band", "y", "x"))
                v[:] = cube
                setattr(nc, "source", b"ENMAP_HSI_L2A")
                setattr(nc, "native_gsd_m", np.int32(GSD))
                setattr(nc, "resampled_to_m", np.int32(10))
                setattr(nc, "scene", scene_id[:60].encode())
            manifest.append({"sample_id": sid, "scene": scene_id})
            written += 1
        print(f"    wrote {len(g['items'])} (total {written}, "
              f"{fobj.bytes_read/1e6:.0f} MB this scene)", flush=True)

    if manifest:
        path = OUT / f"manifest_{int(__import__('time').time())}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("done", written, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

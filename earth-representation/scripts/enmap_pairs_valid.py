"""Scheme C with real data validity: candidate scenes, buffer cut, validity retry.

Two facts drive this version:
  * a 128 px patch can straddle a tile boundary, so tiles are pasted into one
    buffer and patches are cut from the buffer;
  * EnMAP's STAC bbox is the outer rectangle of a tilted swath, so "inside the
    bbox" does not mean "has data". A patch whose cut is mostly nodata is
    rejected and retried against the next candidate scene.

Nothing is written for a patch that fails every candidate; those are reported.
"""

import argparse
import json
import sys
import time
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
SIZE, GSD, NODATA = 128, 30, -32768
MAX_NODATA = 0.20
MAX_TILES_PER_GROUP = 4
MAX_BUFFER_BYTES = 700 * 1024 * 1024


def env():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


_CELL_CACHE = {}


def _cell_features(cx, cy, cell=0.5):
    key = (cx, cy)
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    feats = []
    for attempt in range(4):
        try:
            feats = requests.post(f"{STAC}/search", json={
                "collections": ["ENMAP_HSI_L2A"],
                "bbox": [(cx - 0.5) * cell, (cy - 0.5) * cell,
                         (cx + 0.5) * cell, (cy + 0.5) * cell],
                "datetime": "2022-04-27T00:00:00Z/2026-12-31T23:59:59Z",
                "limit": 300}, timeout=120, headers=UA).json().get("features", [])
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    _CELL_CACHE[key] = feats
    return feats


def candidates(samples, sid, cell=0.25):
    m = samples[sid]
    w = transform_bounds(m["crs"], "EPSG:4326", *m["bbox"], densify_pts=21)
    lon, lat = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
    feats = _cell_features(round(lon / 0.5), round(lat / 0.5))
    hits = [f for f in feats
            if f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3]]
    return sorted(hits, key=lambda f: str(f["properties"].get("datetime", "")))


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


def load_page(session, url, creds):
    login(session, url, creds["ENMAP_USER"], creds["ENMAP_PASS"])
    fobj = HttpRangeFile(url, session)
    page = tifffile.TiffFile(fobj).pages[0]
    return fobj, page


def cut_scene(samples, session, feat, sids, creds, outdir):
    url = feat["assets"]["image"]["href"]
    fobj, page = load_page(session, url, creds)
    epsg = scene_epsg(page)
    tie = page.tags["ModelTiepointTag"].value
    ox, oy = tie[3], tie[4]
    tw = page.tilewidth
    across = (page.imagewidth + tw - 1) // tw
    nbands = page.shape[-1]

    # per patch: which tiles its footprint touches, and the tile holding its centre
    footprint = {}
    for sid in sids:
        m = samples[sid]
        tr = Transformer.from_crs(CRS.from_epsg(int(m["crs"].split(":")[1])),
                                  CRS.from_epsg(epsg), always_xy=True)
        xs, ys = tr.transform([m["bbox"][0], m["bbox"][2]],
                              [m["bbox"][1], m["bbox"][3]])
        cs = [int((x - ox) // GSD) for x in xs]
        rs = [int((oy - y) // GSD) for y in ys]
        tiles = {(r, c) for r in range(min(rs) // tw, max(rs) // tw + 1)
                 for c in range(min(cs) // tw, max(cs) // tw + 1)}
        centre = (int((oy - (ys[0] + ys[1]) / 2) // GSD) // tw,
                  int(((xs[0] + xs[1]) / 2 - ox) // GSD) // tw)
        footprint[sid] = (tiles, centre)

    groups = defaultdict(list)
    for sid, (_tiles, centre) in footprint.items():
        groups[centre].append(sid)

    out = {}
    # a whole-scene buffer reached ~1 GB and got the process killed; each centre
    # tile now gets its own bounded buffer that is released before the next one
    for centre, gids in groups.items():
        union = set()
        for sid in gids:
            union |= footprint[sid][0]
        if len(union) > MAX_TILES_PER_GROUP:
            gids = [s2 for s2 in gids if len(footprint[s2][0]) <= 2]
            union = set()
            for sid in gids:
                union |= footprint[sid][0]
            if not gids or len(union) > MAX_TILES_PER_GROUP:
                continue
        r0, c0 = min(r for r, _ in union), min(c for _, c in union)
        r1, c1 = max(r for r, _ in union), max(c for _, c in union)
        nbytes = (r1 - r0 + 1) * tw * (c1 - c0 + 1) * tw * nbands * 2
        if nbytes > MAX_BUFFER_BYTES:
            continue
        buf = np.full(((r1 - r0 + 1) * tw, (c1 - c0 + 1) * tw, nbands),
                      NODATA, dtype=np.int16)
        try:
            for (r, c) in sorted(union):
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
                if tile.shape[-1] != nbands and tile.shape[0] == nbands:
                    tile = np.transpose(tile, (1, 2, 0))
                yy, xx = (r - r0) * tw, (c - c0) * tw
                buf[yy:yy + tw, xx:xx + tw, :] = tile[:tw, :tw, :]
                del tile, decoded, raw

            origin = (ox + c0 * tw * GSD, oy - r0 * tw * GSD)
            for sid in gids:
                m = samples[sid]
                cube = cut_from_buffer(buf, origin, epsg, m["bbox"],
                                       CRS.from_epsg(int(m["crs"].split(":")[1])))
                frac = float((cube == NODATA).mean())
                if frac <= MAX_NODATA:
                    with netcdf_file(str(outdir / f"{sid}.nc"), "w", version=2) as nc:
                        nc.createDimension("band", cube.shape[0])
                        nc.createDimension("y", SIZE)
                        nc.createDimension("x", SIZE)
                        v = nc.createVariable("spectral", cube.dtype.char,
                                              ("band", "y", "x"))
                        v[:] = cube
                        setattr(nc, "source", b"ENMAP_HSI_L2A")
                        setattr(nc, "native_gsd_m", np.int32(GSD))
                        setattr(nc, "resampled_to_m", np.int32(10))
                        setattr(nc, "nodata_fraction", np.float32(frac))
                    out[sid] = frac
        finally:
            del buf
    mb = fobj.bytes_read / 1e6
    return out, mb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args(argv)

    creds = env()
    samples = load_samples()
    ids = [s for s in sorted(samples) if not (OUT / f"{s}.nc").exists()][: args.limit]
    print(f"pending {len(ids)}", flush=True)
    if not ids:
        return 0

    session = requests.Session()
    session.trust_env = False   # the system proxy here is broken; direct works
    OUT.mkdir(parents=True, exist_ok=True)
    cand = {sid: candidates(samples, sid) for sid in ids}
    depth = {sid: 0 for sid in ids}
    done, rejected = {}, []
    budget = args.scenes

    for round_ in range(args.passes):
        groups = defaultdict(list)
        for sid in ids:
            if sid in done:
                continue
            i = depth[sid]
            if i >= len(cand[sid]):
                rejected.append(sid)
                continue
            groups[cand[sid][i]["id"]].append(sid)
        if not groups:
            break
        print(f"round {round_}: {len(groups)} scenes, {sum(len(v) for v in groups.values())} patches",
              flush=True)
        for scene_id, sids in list(groups.items()):
            if budget <= 0:
                break
            feat = cand[sids[0]][depth[sids[0]]]
            out, info = None, None
            for attempt in range(2):
                try:
                    out, info = cut_scene(samples, session, feat, sids, creds, OUT)
                    break
                except Exception as exc:
                    print(f"  {scene_id[:22]}: attempt {attempt+1} failed "
                          f"{type(exc).__name__}: {str(exc)[:60]}", flush=True)
                    time.sleep(10 * (attempt + 1))
                    session = requests.Session()
                    session.trust_env = False
            if out is None:
                for sid in sids:
                    depth[sid] += 1
                continue
            budget -= 1
            if info == "too_many_tiles":
                print(f"  {scene_id[:22]}: skipped (too many tiles)", flush=True)
                for sid in sids:
                    depth[sid] += 1
                continue
            print(f"  {scene_id[:22]}: {len(out)}/{len(sids)} valid, {info:.0f} MB", flush=True)
            for sid in sids:
                if sid in out:
                    done[sid] = out[sid]
                else:
                    depth[sid] += 1

    (OUT / "rejected.json").write_text(json.dumps(
        {"rejected": rejected, "written": len(done)}, indent=2), encoding="utf-8")
    print(f"written {len(done)}, rejected {len(rejected)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

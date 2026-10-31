"""Shared helpers for the Phase 0 data-access audits."""

import hashlib
import json
import time
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parents[1]
CMR_STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
EDL = "https://urs.earthdata.nasa.gov"
UA = "earth-representation-audit/0.1"


def load_env():
    """Read .env next to the project root; values keep their literal characters."""
    env = {}
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def earthdata_token(env):
    """POST creates a token; EDL rejects that once the account is at its limit,
    in which case the existing tokens are listed instead."""
    auth = (env["EARTHDATA_USER"], env["EARTHDATA_PASS"])
    r = requests.post(f"{EDL}/api/users/token", auth=auth, timeout=120)
    if r.ok and r.json().get("access_token"):
        return r.json()["access_token"]
    r = requests.get(f"{EDL}/api/users/tokens", auth=auth, timeout=120)
    r.raise_for_status()
    return r.json()[0]["access_token"]


def search(collection, bbox, start, end, limit=1000, retries=3):
    """CMR-STAC item search; returns (features, matched)."""
    payload = {"collections": [collection], "bbox": list(bbox),
               "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": limit}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{CMR_STAC}/search", json=payload, timeout=180,
                              headers={"User-Agent": UA})
            r.raise_for_status()
            doc = r.json()
            return doc.get("features", []), doc.get("context", {}).get("matched")
        except Exception as exc:  # transient 5xx / timeouts
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"CMR-STAC search failed for {collection}: {last}")


def post_json(url, payload, retries=6, timeout=180):
    """POST with backoff; Earth Search answers 502 in bursts under load."""
    last = None
    waits = [5, 15, 30, 45, 60, 60]
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout,
                              headers={"User-Agent": UA})
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(waits[min(attempt, len(waits) - 1)])
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(waits[min(attempt, len(waits) - 1)])
    raise RuntimeError(f"POST {url} failed after {retries} tries: {last}")


def get_json(url, params=None, retries=4, timeout=120):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": UA})
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def download(url, dst, headers=None, retries=3):
    """Stream one asset to dst; returns (bytes, sha256). Skips work if present."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst.stat().st_size, file_sha256(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                h = hashlib.sha256()
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(2 * 1024 * 1024):
                        fh.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
            tmp.replace(dst)
            return n, h.hexdigest()
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed {url}: {last}")


def file_sha256(path, chunk=2 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def raster_report(path):
    """Structural description of a raster, for the read-verification records."""
    import rasterio

    with rasterio.open(path) as ds:
        return {
            "driver": ds.driver,
            "crs": str(ds.crs),
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "nodata": ds.nodata,
            "bounds": [round(v, 3) for v in ds.bounds],
            "compression": ds.profile.get("compress"),
            "tiled": bool(ds.profile.get("tiled")),
            "transform_ok": ds.transform.a != 0 and ds.transform.e != 0,
        }


def sample_stats(path, band=1, size=512):
    """Cheap pixel summary from the upper-left corner window (local file read)."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        win = Window(0, 0, min(size, ds.width), min(size, ds.height))
        a = ds.read(band, window=win).astype("float64")
        nd = ds.nodata
        if nd is not None:
            a[a == nd] = np.nan
        if a.size == 0 or np.all(np.isnan(a)):
            return {"min": None, "max": None, "mean": None, "nan_frac": 1.0}
        return {"min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
                "mean": round(float(np.nanmean(a)), 4),
                "nan_frac": round(float(np.isnan(a).mean()), 4)}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def header_file(token, path):
    """GDAL 3.12 ignores GDAL_HTTP_HEADERS here; a header file does work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Authorization: Bearer {token}\n", encoding="utf-8")
    return path


def remote_env(header=None):
    opts = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    if header:
        opts["GDAL_HTTP_HEADER_FILE"] = str(header)
    return opts


def window_read(url, bbox, dst, header=None):
    """Read only the AOI window of a remote COG and store it locally.

    Returns a description dict, or None when the AOI misses the scene. Nothing
    outside the AOI is transferred, so a full-granule download never happens.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    dst = Path(dst)
    with rasterio.Env(**remote_env(header)):
        with rasterio.open(url) as src:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *bbox, densify_pts=21)
            win = from_bounds(left, bottom, right, top, src.transform)
            win = win.round_offsets().round_lengths()
            # clip to the raster by hand: Window.intersection raises when empty
            col0 = max(0, int(win.col_off))
            row0 = max(0, int(win.row_off))
            col1 = min(src.width, int(win.col_off + win.width))
            row1 = min(src.height, int(win.row_off + win.height))
            if col1 - col0 < 1 or row1 - row0 < 1:
                return None
            win = Window(col0, row0, col1 - col0, row1 - row0)
            data = src.read(1, window=win)
            profile = src.profile.copy()
            for k in ("blockxsize", "blockysize", "tiled", "interleave"):
                profile.pop(k, None)
            profile.update(driver="GTiff", height=data.shape[0],
                           width=data.shape[1], compress="deflate",
                           transform=src.window_transform(win))
            dst.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst, "w", **profile) as out:
                out.write(data, 1)
    return {"width": data.shape[1], "height": data.shape[0],
            "crs": str(src.crs), "dtype": str(data.dtype),
            "min": int(data.min()), "max": int(data.max()),
            "bytes": dst.stat().st_size}


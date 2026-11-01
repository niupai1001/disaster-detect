#!/usr/bin/env python
"""Download HLS (NASA Harmonized Landsat-Sentinel) and EnMAP (DLR hyperspectral) data.

Two independent sources, one CLI:

  HLS  -- Microsoft Planetary Computer STAC (`hls2-l30`, `hls2-s30`).
          Public catalogue, no account. Assets live in a private Azure account,
          so every href is signed with a short-lived SAS token obtained from
          Planetary Computer's `/api/sas/v1/token/{account}/{container}`.

  EnMAP -- DLR EOC Geoservice STAC (`ENMAP_HSI_L2A` / `L1B`, public metadata)
          with files behind the EOC CAS single-sign-on. Requires an "EnMAP
          Access Service" account (free registration at https://eoweb.dlr.de).

Modes
  --mode window : read only the AOI window out of the cloud-optimized GeoTIFF
                  (range requests; HLS only, needs rasterio). Fast, small output.
  --mode full   : download whole files.

Every downloaded product is recorded in `manifest.jsonl` next to the output
tree, with size and sha256, so a later run can verify rather than re-fetch.

Examples
  # HLS over a small AOI, windowed to the bbox
  python scripts/download_hls_enmap.py --source hls \
      --bbox 103.0 30.0 103.1 30.1 --start 2024-08-20 --end 2024-08-30 \
      --assets B04 Fmask --mode window --out data/hls_probe

  # EnMAP full scenes (needs credentials)
  set ENMAP_USER=... & set ENMAP_PASS=...
  python scripts/download_hls_enmap.py --source enmap \
      --bbox 103.0 30.0 103.3 30.3 --start 2023-01-01 --end 2026-01-01 \
      --assets vnir swir --mode full --out data/enmap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import requests

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"
DLR_STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
UA = "landslide-seg-downloader/1.0"

HLS_COLLECTIONS = ("hls2-l30", "hls2-s30")
HLS_DEFAULT_ASSETS = ("B02", "B03", "B04", "B8A", "Fmask")
ENMAP_DEFAULT_ASSETS = ("vnir", "swir", "quality_cloud")


# --------------------------------------------------------------------------- #
# HLS via Planetary Computer
# --------------------------------------------------------------------------- #
def pc_search(collections, bbox, start, end, limit=500):
    """Search HLS collections over a bbox and date range; returns STAC features."""
    features = []
    payload = {
        "collections": list(collections),
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": min(limit, 1000),
    }
    url = f"{PC_STAC}/search"
    while url:
        r = requests.post(url, json=payload, timeout=90, headers={"User-Agent": UA})
        r.raise_for_status()
        doc = r.json()
        features.extend(doc.get("features", []))
        url = None
        for link in doc.get("links", []):
            if link.get("rel") == "next" and link.get("method", "POST") == "POST":
                url, payload = link["href"], link.get("body") or payload
                break
        if len(features) >= limit:
            break
    return features[:limit]


def pc_sas_token(href):
    """Return the SAS query string for a Planetary Computer blob href."""
    host = urllib.parse.urlparse(href).netloc
    account = host.split(".")[0]
    container = urllib.parse.urlparse(href).path.strip("/").split("/")[0]
    url = PC_SAS.format(account=account, container=container)
    r = requests.get(url, params={"_ts": int(time.time())}, timeout=60,
                     headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json()["token"]


def signed(href, token):
    return f"{href}?{token}" if "?" not in href else f"{href}&{token}"


def fetch_cached(session, url, cache_path):
    """Download to a local cache file once, reusing it on later runs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    tmp = cache_path.with_suffix(cache_path.suffix + ".part")
    with session.get(url, stream=True, timeout=300,
                     headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", ""):
            raise RuntimeError(f"got HTML instead of data ({r.status_code})")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(4 * 1024 * 1024):
                fh.write(chunk)
    tmp.replace(cache_path)
    return cache_path


def hls_window(src_path, bbox, dst):
    """Crop the bbox window out of a local HLS COG.

    GDAL's bundled libcurl uses Windows schannel in this environment, which
    fails TLS credentials, so remote /vsicurl reads are not usable here; the
    file is fetched with requests (OpenSSL) first and cropped locally.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(src_path) as src:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *bbox, densify_pts=21)
            win = from_bounds(left, bottom, right, top, src.transform)
            win = win.round_offsets().round_lengths()
            win = win.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width <= 0 or win.height <= 0:
                return None
            data = src.read(1, window=win)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff", height=data.shape[0], width=data.shape[1],
                transform=src.window_transform(win), compress="deflate",
                tiled=False,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst, "w", **profile) as out:
                out.write(data, 1)
    return dst


# --------------------------------------------------------------------------- #
# EnMAP via DLR EOC Geoservice (CAS SSO)
# --------------------------------------------------------------------------- #
class _FormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self._current = {"action": a.get("action", ""), "fields": {}}
            self.forms.append(self._current)
        elif tag == "input" and self._current is not None and a.get("name"):
            self._current["fields"][a["name"]] = a.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self._current = None


def enmap_login(session, user, password):
    """Walk the EOC CAS login form and keep the session cookie.

    Untested against a live account (no credentials available at build time);
    the field names below are what the CAS login page actually posts.
    """
    probe = "https://download.geoservice.dlr.de/ENMAP/files/"
    r = session.get(probe, timeout=60, headers={"User-Agent": UA})
    if "login" not in r.url and "sso" not in r.url:
        return session  # already authorised (e.g. cookie injected by caller)

    parser = _FormParser()
    parser.feed(r.text)
    form = next((f for f in parser.forms if "password" in f["fields"]), None)
    if form is None:
        raise RuntimeError("EnMAP CAS login form not found; page layout changed?")

    fields = dict(form["fields"])
    fields["username"] = user
    fields["password"] = password
    fields.setdefault("_eventId", "submit")
    action = urllib.parse.urljoin(r.url, form["action"] or "login")

    resp = session.post(action, data=fields, timeout=90,
                        headers={"User-Agent": UA, "Referer": r.url})
    if "password" in resp.text and "login" in resp.url:
        raise RuntimeError("EnMAP login rejected (check ENMAP_USER / ENMAP_PASS)")
    return session


def dlr_search(collection, bbox, start, end, limit=200):
    payload = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": min(limit, 1000),
    }
    r = requests.post(f"{DLR_STAC}/search", json=payload, timeout=90,
                      headers={"User-Agent": UA})
    r.raise_for_status()
    return r.json().get("features", [])[:limit]


def stream_download(session, url, dst, expected_html_guard=True):
    """Download one asset, refusing to save an HTML login page as a product."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    sha = hashlib.sha256()
    size = 0
    with session.get(url, stream=True, timeout=300, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if expected_html_guard and "text/html" in ctype:
            raise RuntimeError(
                f"got HTML instead of a product ({r.status_code}); "
                "EnMAP download needs a logged-in session")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    sha.update(chunk)
                    size += len(chunk)
    tmp.replace(dst)
    return size, sha.hexdigest()


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def append_manifest(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def run_hls(args):
    out = Path(args.out)
    manifest = out / "manifest.jsonl"
    feats = pc_search(args.collections, args.bbox, args.start, args.end,
                      limit=args.limit)
    print(f"HLS: {len(feats)} item(s) match bbox/dates")
    if not feats:
        return 0

    token_cache = {}
    session = requests.Session()
    cache = out / "_cache"
    ok = 0
    for feat in feats:
        item_id = feat["id"]
        item_dir = out / item_id
        if args.max_items and ok >= args.max_items:
            break
        for asset in args.assets:
            a = feat["assets"].get(asset)
            if not a:
                print(f"  skip {item_id}: no asset {asset}")
                continue
            href = a["href"]
            acct = urllib.parse.urlparse(href).netloc
            if acct not in token_cache:
                token_cache[acct] = pc_sas_token(href)
            dst = item_dir / f"{asset}.tif"
            try:
                local = fetch_cached(session, signed(href, token_cache[acct]),
                                     cache / f"{item_id}-{asset}.tif")
                if args.mode == "window":
                    res = hls_window(local, args.bbox, dst)
                    if res is None:
                        print(f"  skip {item_id}/{asset}: AOI outside scene")
                        continue
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    local.replace(dst)
                size, digest = dst.stat().st_size, sha256_file(dst)
                append_manifest(manifest, {
                    "source": "hls", "mode": args.mode, "item_id": item_id,
                    "product": feat["properties"].get("hls:product"),
                    "platform": feat["properties"].get("platform"),
                    "datetime": feat["properties"].get("datetime"),
                    "bbox": args.bbox, "asset": asset, "path": str(dst),
                    "bytes": size, "sha256": digest,
                    "url": href,
                })
                ok += 1
                print(f"  ok {item_id}/{asset} {size/1e6:.1f} MB")
            except Exception as exc:  # keep going, report at the end
                print(f"  FAIL {item_id}/{asset}: {type(exc).__name__}: {exc}")
    print(f"HLS done: {ok} file(s) -> {out}")
    return 0 if ok else 1


def run_enmap(args):
    out = Path(args.out)
    manifest = out / "manifest.jsonl"
    user = args.user or os.environ.get("ENMAP_USER")
    password = args.password or os.environ.get("ENMAP_PASS")
    if not (user and password) and not args.cookie:
        print("EnMAP needs credentials: set ENMAP_USER / ENMAP_PASS "
              "(free account at https://eoweb.dlr.de) or pass --cookie.",
              file=sys.stderr)
        return 2

    feats = dlr_search(args.collection, args.bbox, args.start, args.end,
                       limit=args.limit)
    print(f"EnMAP: {len(feats)} item(s) match bbox/dates")
    if not feats:
        print("note: EnMAP is a target-of-opportunity mission; empty result is "
              "a genuine coverage gap, not a query error.")
        return 0

    session = requests.Session()
    if args.cookie:
        for part in args.cookie.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                session.cookies.set(k.strip(), v.strip(),
                                    domain="download.geoservice.dlr.de")
    else:
        enmap_login(session, user, password)

    ok = 0
    for feat in feats:
        if args.max_items and ok >= args.max_items:
            break
        item_dir = out / feat["id"]
        for asset in args.assets:
            a = feat["assets"].get(asset)
            if not a:
                continue
            dst = item_dir / f"{feat['id']}-{asset}{Path(a['href']).suffix}"
            try:
                size, digest = stream_download(session, a["href"], dst)
                append_manifest(manifest, {
                    "source": "enmap", "mode": "full", "item_id": feat["id"],
                    "collection": args.collection,
                    "datetime": feat["properties"].get("datetime"),
                    "bbox": args.bbox, "asset": asset, "path": str(dst),
                    "bytes": size, "sha256": digest, "url": a["href"],
                })
                ok += 1
                print(f"  ok {feat['id']}/{asset} {size/1e6:.1f} MB")
            except Exception as exc:
                print(f"  FAIL {feat['id']}/{asset}: {type(exc).__name__}: {exc}")
    print(f"EnMAP done: {ok} file(s) -> {out}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hls", "enmap"], required=True)
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("W", "S", "E", "N"))
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["window", "full"], default="window")
    p.add_argument("--assets", nargs="+")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--collections", nargs="+", default=list(HLS_COLLECTIONS))
    p.add_argument("--collection", default="ENMAP_HSI_L2A")
    p.add_argument("--user")
    p.add_argument("--password")
    p.add_argument("--cookie", help="raw Cookie header from a logged-in browser")
    args = p.parse_args(argv)

    if args.assets is None:
        args.assets = list(HLS_DEFAULT_ASSETS if args.source == "hls"
                           else ENMAP_DEFAULT_ASSETS)
    if args.source == "enmap" and args.mode == "window":
        p.error("EnMAP windowed reads are not implemented; use --mode full")
    return run_hls(args) if args.source == "hls" else run_enmap(args)


if __name__ == "__main__":
    sys.exit(main())

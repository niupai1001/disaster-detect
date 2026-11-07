"""Can GDAL window-read the EnMAP COG directly (no whole-scene download)?

Tries three auth transports: header file, Netscape cookie jar, and both. The
goal is per-patch range reads of ~4 MB instead of a 440 MB scene download.
"""

import sys
import time
from pathlib import Path

import rasterio
import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.enmap_download_eoc import login  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

feat = requests.post(f"{STAC}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
url = feat["assets"]["image"]["href"]

s = requests.Session()
login(s, url, env["ENMAP_USER"], env["ENMAP_PASS"])
print("cookies:", [(c.name, c.domain) for c in s.cookies])

tmp = ROOT / ".tmp"
tmp.mkdir(exist_ok=True)
hdr = tmp / "em_header.txt"
hdr.write_text("Cookie: " + "; ".join(f"{c.name}={c.value}" for c in s.cookies) + "\n",
               encoding="utf-8")

jar = tmp / "em_cookies.txt"
lines = ["# Netscape HTTP Cookie File"]
for c in s.cookies:
    dom = c.domain or "download.geoservice.dlr.de"
    lines.append(f"{dom}\tFALSE\t/\tFALSE\t0\t{c.name}\t{c.value}")
jar.write_text("\n".join(lines) + "\n", encoding="utf-8")

variants = {
    "header file": {"GDAL_HTTP_HEADER_FILE": str(hdr)},
    "cookie jar": {"GDAL_HTTP_COOKIEFILE": str(jar)},
    "both": {"GDAL_HTTP_HEADER_FILE": str(hdr), "GDAL_HTTP_COOKIEFILE": str(jar)},
}
base = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".TIF",
        "GDAL_HTTP_TIMEOUT": "60"}

for name, extra in variants.items():
    t0 = time.time()
    try:
        with rasterio.Env(**{**base, **extra}):
            with rasterio.open(url) as ds:
                w = rasterio.windows.Window(600, 500, 64, 64)
                a = ds.read(1, window=w)
                print(f"{name:12s} OK bands={ds.count} window {a.shape} "
                      f"range {int(a.min())}..{int(a.max())} {time.time()-t0:.1f}s")
                break
    except Exception as exc:
        print(f"{name:12s} FAIL {type(exc).__name__}: {str(exc)[:90]}")

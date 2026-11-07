"""COG range reads through an authenticated file object.

GDAL's /vsicurl does not carry our DLR session cookie (tried header file and
cookie jar), so this exposes the same remote GeoTIFF as a seekable Python file
that issues HTTP Range requests with the session's cookies. rasterio can open
such an object directly, which keeps the traffic proportional to the window
being read instead of the whole 440 MB scene.
"""

import io
import sys
import time
from pathlib import Path

import rasterio
import requests
from rasterio.windows import Window

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.enmap_download_eoc import login  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable view over an HTTP resource, one Range request per read."""

    def __init__(self, url, session, block=1 << 20):
        self.url = url
        self.session = session
        self.pos = 0
        self.bytes_read = 0
        self.requests_made = 0
        r = session.get(url, headers={**UA, "Range": "bytes=0-0"}, timeout=120)
        cr = r.headers.get("content-range", "")
        self.size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers.get("content-length", 0))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = self.session.get(self.url, headers={
            **UA, "Range": f"bytes={self.pos}-{end}"}, timeout=300)
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        self.bytes_read += len(data)
        self.requests_made += 1
        return data


def main():
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

    f = HttpRangeFile(url, s)
    print(f"size {f.size/1e6:.1f} MB")
    t0 = time.time()
    with rasterio.open(f, driver="GTiff") as ds:
        print(f"opened: {ds.count} bands, {ds.width}x{ds.height}, {ds.crs}")
        a = ds.read(1, window=Window(600, 500, 64, 64))
        print(f"band1 window {a.shape} range {int(a.min())}..{int(a.max())}")
        b = ds.read([1, 100, 224], window=Window(600, 500, 32, 32))
        print(f"3-band window {b.shape}")
    print(f"elapsed {time.time()-t0:.1f}s | downloaded {f.bytes_read/1e6:.1f} MB "
          f"in {f.requests_made} range requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

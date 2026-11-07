"""Authenticated range reads for the EnMAP COG, paced for the DLR service.

Three things this needs beyond a naive reader:
  * the session cookie has to be sent with every byte-range request;
  * the service answers 503 when requests arrive in a burst, so requests are
    spaced and retried with backoff;
  * tifffile reads the header repeatedly, so a small read-ahead cache keeps the
    request count proportional to the tiles actually needed.
"""

import io
import sys
import time
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}


class HttpRangeFile(io.RawIOBase):
    MAX_CHUNK = 4 << 20      # the service drops long single ranges

    def __init__(self, url, session, block=1 << 19, min_interval=0.35):
        self.url = url
        self.session = session
        self.block = block
        self.min_interval = min_interval
        self.pos = 0
        self.bytes_read = 0
        self.requests_made = 0
        self._last = 0.0
        self._cache_start = 0
        self._cache = b""
        r = session.get(url, headers={**UA, "Range": "bytes=0-0"}, timeout=120)
        if r.status_code != 206:
            raise RuntimeError(f"cannot probe size: {r.status_code}")
        self.size = int(r.headers["content-range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def _fetch(self, start, n):
        """Fetch n bytes, splitting long ranges into chunks."""
        if n > self.MAX_CHUNK:
            out = []
            got = 0
            while got < n:
                piece = min(self.MAX_CHUNK, n - got)
                out.append(self._fetch_small(start + got, piece))
                got += piece
            return b"".join(out)
        return self._fetch_small(start, n)

    def _fetch_small(self, start, n):
        gap = time.time() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        end = min(start + n, self.size) - 1
        for attempt in range(8):
            try:
                r = self.session.get(self.url,
                                     headers={**UA, "Range": f"bytes={start}-{end}"},
                                     timeout=300)
            except requests.RequestException:
                time.sleep(min(60, 2 * 2 ** attempt))
                continue
            self._last = time.time()
            if r.status_code == 206:
                self.bytes_read += len(r.content)
                self.requests_made += 1
                return r.content
            if r.status_code in (429, 500, 502, 503):
                time.sleep(min(90, 3 * 2 ** attempt))
                continue
            r.raise_for_status()
        raise RuntimeError(f"range request kept failing at {start}+{n}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        start, stop = self.pos, min(self.pos + n, self.size)
        if self._cache_start <= start and stop <= self._cache_start + len(self._cache):
            out = self._cache[start - self._cache_start:stop - self._cache_start]
        else:
            span = max(self.block, n)
            data = self._fetch(start, span)
            self._cache_start, self._cache = start, data
            out = data[:stop - start]
        self.pos = stop
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_cog(url, session):
    import tifffile
    import zarr
    fobj = HttpRangeFile(url, session)
    tf = tifffile.TiffFile(fobj)
    arr = zarr.open_array(tf.aszarr(level=0), mode="r")
    return arr, fobj


def main():
    from scripts.enmap_download_eoc import login

    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

    feat = requests.post("https://geoservice.dlr.de/eoc/ogc/stac/v1/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
        "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
        timeout=120, headers=UA).json()["features"][0]
    url = feat["assets"]["image"]["href"]

    s = requests.Session()
    s.trust_env = False
    login(s, url, env["ENMAP_USER"], env["ENMAP_PASS"])

    t0 = time.time()
    arr, fobj = open_cog(url, s)
    print(f"shape {arr.shape} dtype {arr.dtype} chunks {arr.chunks}")
    print(f"open: {fobj.bytes_read/1e6:.2f} MB / {fobj.requests_made} req / {time.time()-t0:.1f}s")

    t1 = time.time()
    block = arr[:, 600:728, 500:628]
    print(f"window {block.shape} -> {int(block.min())}..{int(block.max())}")
    print(f"total: {fobj.bytes_read/1e6:.2f} MB / {fobj.requests_made} req / "
          f"{time.time()-t1:.1f}s for the window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

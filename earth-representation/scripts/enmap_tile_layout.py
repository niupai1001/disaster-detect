"""Inspect the COG tile layout: are bands' tiles adjacent in the file?

If the tiles for band b, tile (row, col) are stored next to band b+1's tile at
the same position, one range request can serve many bands, and a patch needs a
handful of requests instead of 224.
"""

import sys
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.enmap_cog_read import HttpRangeFile  # noqa: E402
from scripts.enmap_download_eoc import login  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}

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
login(s, url, env["ENMAP_USER"], env["ENMAP_PASS"])

import tifffile
fobj = HttpRangeFile(url, s)
tf = tifffile.TiffFile(fobj)
page = tf.pages[0]
print(f"pages={len(tf.pages)} shape={page.shape} tileshape={page.tilewidth}x{page.tilelength} "
      f"compression={page.compression} planar={page.planarconfig}")
print(f"requests so far: {fobj.requests_made}, {fobj.bytes_read/1024:.0f} KB")

offsets = page.dataoffsets
counts = page.databytecounts
print(f"tile count={len(offsets)} first offsets={offsets[:6]}")
print(f"byte counts sample={counts[:6]}")
# adjacency: offsets should increase by roughly the byte count if tiles are packed
adjacent = sum(1 for i in range(min(300, len(offsets) - 1))
               if offsets[i + 1] - offsets[i] <= counts[i] + 1024)
print(f"adjacent tiles in first 300: {adjacent}/299")
print(f"total requests: {fobj.requests_made}")

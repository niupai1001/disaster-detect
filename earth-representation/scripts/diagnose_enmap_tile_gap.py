"""Why do some patches land outside the tile that was fetched?

For one scene, print per patch: centre in scene CRS, the tiles its bbox touches,
the tile that holds the centre, and whether the current code would cut it from
the wrong tile.
"""

import sys
from pathlib import Path

import numpy as np
import requests
import tifffile
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402
from scripts.enmap_cog_read import HttpRangeFile  # noqa: E402
from scripts.enmap_download_eoc import login  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

env = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

samples = load_samples()
# the scene that produced the warnings
target = "dominicamaria"
ids = [s for s in sorted(samples) if s.startswith(target)][:40]
w = transform_bounds(samples[ids[0]]["crs"], "EPSG:4326",
                     *samples[ids[0]]["bbox"], densify_pts=21)
lon, lat = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
feats = requests.post(f"{STAC}/search", json={
    "collections": ["ENMAP_HSI_L2A"],
    "bbox": [lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02],
    "datetime": "2022-04-27T00:00:00Z/2026-12-31T23:59:59Z", "limit": 20},
    timeout=120, headers=UA).json().get("features", [])
hit = [f for f in feats
       if f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3]]
feat = min(hit, key=lambda f: str(f["properties"].get("datetime", "")))
url = feat["assets"]["image"]["href"]
print("scene:", feat["id"][:50])

s = requests.Session()
login(s, url, env["ENMAP_USER"], env["ENMAP_PASS"])
fobj = HttpRangeFile(url, s)
tf = tifffile.TiffFile(fobj)
page = tf.pages[0]

keys = page.tags["GeoKeyDirectoryTag"].value
epsg = 32632
for i in range(4, len(keys), 4):
    if keys[i] in (2048, 3072):
        epsg = int(keys[i + 3])
        break
tie = page.tags["ModelTiepointTag"].value
ox, oy = tie[3], tie[4]
tw = page.tilewidth
across = (page.imagewidth + tw - 1) // tw
print(f"epsg={epsg} origin=({ox},{oy}) tile={tw} across={across} "
      f"size={page.imagewidth}x{page.imagelength}")

bad = 0
for sid in ids:
    m = samples[sid]
    tr = Transformer.from_crs(CRS.from_epsg(int(m["crs"].split(":")[1])),
                              CRS.from_epsg(epsg), always_xy=True)
    corners_x = [m["bbox"][0], m["bbox"][2]]
    corners_y = [m["bbox"][1], m["bbox"][3]]
    xs, ys = tr.transform(corners_x, corners_y)
    cols = [int((x - ox) // 30) for x in xs]
    rows = [int((oy - y) // 30) for y in ys]
    tiles = {(r // tw, c // tw) for r in rows for c in cols}
    cx = (xs[0] + xs[1]) / 2
    cy = (ys[0] + ys[1]) / 2
    centre_tile = (int((oy - cy) // 30) // tw, int((cx - ox) // 30) // tw)
    if len(tiles) > 1:
        bad += 1
        if bad <= 5:
            print(f"  {sid}: centre_tile={centre_tile} bbox_tiles={sorted(tiles)} "
                  f"cols={cols} rows={rows}")
print(f"patches touching more than one tile: {bad}/{len(ids)}")

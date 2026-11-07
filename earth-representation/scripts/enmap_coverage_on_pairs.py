"""How many of the existing S2/Landsat patch footprints have EnMAP coverage?

The companion download worked per event group; this asks the same question per
paired patch, because EnMAP is what has to be joined to those pairs - not the
other way round.

Groups patches by a coarse cell so one STAC query serves many patches.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import requests
from rasterio.warp import transform_bounds

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))

from download_companion import load_samples  # noqa: E402

UA = {"User-Agent": "enmap-coverage/1.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
WINDOW_DAYS = 120


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--cell", type=float, default=0.5, help="grouping cell in degrees")
    ap.add_argument("--any-date", action="store_true", help="ignore patch dates (EnMAP archive only)")
    args = ap.parse_args(argv)

    samples = load_samples()
    ids = sorted(samples)[: args.limit]
    print(f"patches checked: {len(ids)} of {len(samples)}")

    cells = defaultdict(list)
    for sid in ids:
        m = samples[sid]
        w = transform_bounds(m["crs"], "EPSG:4326", *m["bbox"], densify_pts=21)
        lon = (w[0] + w[2]) / 2
        lat = (w[1] + w[3]) / 2
        cells[(round(lon / args.cell), round(lat / args.cell))].append((sid, w, m["date"]))
    print(f"cells: {len(cells)}")

    covered, total = 0, 0
    per_cell = []
    for key, items in cells.items():
        w0 = min(i[1][0] for i in items), min(i[1][1] for i in items)
        w1 = max(i[1][2] for i in items), max(i[1][3] for i in items)
        dates = sorted(i[2] for i in items if i[2])
        bbox = [w0[0], w0[1], w1[0], w1[1]]
        if args.any_date:
            start, end = "2022-04-27T00:00:00Z", "2026-12-31T23:59:59Z"
        else:
            start = f"{dates[0]}T00:00:00Z" if dates else "2022-04-27T00:00:00Z"
            end = f"{dates[-1]}T23:59:59Z" if dates else "2026-12-31T23:59:59Z"
        try:
            doc = requests.post(f"{STAC}/search", json={
                "collections": ["ENMAP_HSI_L2A"], "bbox": bbox,
                "datetime": f"{start}/{end}", "limit": 200}, timeout=120,
                headers=UA).json()
            feats = doc.get("features", [])
        except Exception as exc:
            print("query failed", key, type(exc).__name__)
            continue
        for sid, w, date in items:
            total += 1
            hit = any(f["bbox"][0] <= (w[0] + w[2]) / 2 <= f["bbox"][2]
                      and f["bbox"][1] <= (w[1] + w[3]) / 2 <= f["bbox"][3]
                      for f in feats)
            covered += hit
        per_cell.append((key, len(items), len(feats)))
        if len(per_cell) <= 8:
            print(f"  cell {key}: {len(items):3d} patches, {len(feats):3d} EnMAP scenes")

    print(f"covered: {covered}/{total} = {covered/max(total,1):.1%}")
    print(f"cells with no scene: {sum(1 for c in per_cell if c[2] == 0)}/{len(per_cell)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

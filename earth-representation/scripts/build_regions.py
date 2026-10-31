"""Build configs/regions.yaml: one AOI per existing landslide event.

The patch manifest (landslide-seg) stores per-patch bboxes in a projected CRS,
so the event AOI is the union of its patch bboxes converted to WGS84.
"""

import json
from collections import defaultdict
from pathlib import Path

import yaml
from pyproj import Transformer

HERE = Path(__file__).resolve().parents[1]
SRC = HERE.parent / "landslide-seg" / "manifests" / "s2_10band_v4_p20.jsonl"
OUT = HERE / "configs" / "regions.yaml"

events = {}
with open(SRC, encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line)
        e = d["event_id"]
        b = d["bbox"]
        t = Transformer.from_crs(d["crs"], "EPSG:4326", always_xy=True)
        xs, ys = t.transform([b[0], b[2]], [b[1], b[3]])
        agg = events.setdefault(e, {"w": 9e9, "s": 9e9, "e": -9e9, "n": -9e9,
                                    "n_samples": 0, "dates": set()})
        agg["w"] = min(agg["w"], min(xs))
        agg["e"] = max(agg["e"], max(xs))
        agg["s"] = min(agg["s"], min(ys))
        agg["n"] = max(agg["n"], max(ys))
        agg["n_samples"] += 1
        if d.get("event_date"):
            for one in d["event_date"].split(","):
                one = one.strip()
                if one[:4].isdigit() and len(one) == 10:
                    agg["dates"].add(one)

regions = {}
for name, a in sorted(events.items()):
    regions[name] = {
        "bbox": [round(a["w"], 4), round(a["s"], 4), round(a["e"], 4), round(a["n"], 4)],
        "event_dates": sorted(a["dates"]),
        "n_samples": a["n_samples"],
    }

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(yaml.safe_dump({"regions": regions}, sort_keys=True), encoding="utf-8")
for name, r in regions.items():
    w, s, e, n = r["bbox"]
    print(f"{name:16s} {r['n_samples']:5d} patches  "
          f"{w:8.3f},{s:7.3f} -> {e:8.3f},{n:7.3f}  ({e-w:.2f} x {n-s:.2f} deg)")

"""Phase 0: Sentinel-2 L2A availability audit and AOI-window read test.

Catalogue: Planetary Computer (same Copernicus L2A product as CDSE).
Assets need the short-lived SAS token; only the AOI window is transferred.

Earth Search was tried first and dropped: it answered HTTP 502 for most
regions under this session's request rate, while PC stayed responsive.
"""

import argparse
import datetime as dt
import sys
import urllib.parse
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
MANIFEST = C.PROJECT / "data" / "manifests" / "s2_availability.json"
BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL")
_TOKENS = {}


def sas_for(href):
    u = urllib.parse.urlparse(href)
    key = (u.netloc.split(".")[0], u.path.strip("/").split("/")[0])
    if key not in _TOKENS:
        doc = C.get_json("https://planetarycomputer.microsoft.com/api/sas/v1/token/"
                         f"{key[0]}/{key[1]}")
        _TOKENS[key] = doc["token"]
    return _TOKENS[key]


def search(bbox, start, end, limit=500):
    doc = C.post_json(f"{PC}/search", {
        "collections": ["sentinel-2-l2a"], "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": limit})
    return doc.get("features", [])


def region_stats(region, window_days):
    dates = sorted({dt.date.fromisoformat(d) for d in region["event_dates"]})
    day = dates[0]
    feats = search(region["bbox"], day - dt.timedelta(days=window_days),
                   day + dt.timedelta(days=window_days))
    clouds = [float(f["properties"]["eo:cloud_cover"]) for f in feats
              if f["properties"].get("eo:cloud_cover") is not None]
    days = sorted({f["properties"]["datetime"][:10] for f in feats})
    return {"bbox": region["bbox"], "scenes": len(feats),
            "date_min": days[0] if days else None,
            "date_max": days[-1] if days else None,
            "cloud_median": round(sorted(clouds)[len(clouds) // 2], 2) if clouds else None,
            "scenes_under_10pct_cloud": sum(1 for c in clouds if c <= 10)}


def read_test(name, region):
    dates = sorted({dt.date.fromisoformat(d) for d in region["event_dates"]})
    feats = search(region["bbox"], dates[0] - dt.timedelta(days=365),
                   dates[-1] + dt.timedelta(days=365))
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    if not feats:
        return None, "no scenes"
    f = feats[0]
    outdir = C.PROJECT / "data" / "raw" / "s2" / f["id"]
    rec = {"scene": f["id"], "region": name,
           "datetime": f["properties"]["datetime"],
           "cloud_cover": f["properties"].get("eo:cloud_cover"),
           "properties": {k: f["properties"].get(k) for k in
                          ("s2:processing_baseline", "s2:product_uri", "proj:epsg")
                          if k in f["properties"]},
           "bands": {}, "bytes": 0}
    for band in BANDS:
        a = f["assets"].get(band)
        if not a:
            continue
        info = C.window_read(f"{a['href']}?{sas_for(a['href'])}", region["bbox"],
                             outdir / f"{band}_aoi.tif")
        if info is None:
            continue
        rec["bands"][band] = info
        rec["bytes"] += info["bytes"]
    return rec, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=str(C.PROJECT / "configs" / "regions.yaml"))
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--read-test", default="hiroshima")
    args = ap.parse_args(argv)

    regions = yaml.safe_load(Path(args.regions).read_text(encoding="utf-8"))["regions"]
    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
              "catalogue": PC, "collection": "sentinel-2-l2a",
              "regions": {}, "read_test": None, "failures": []}
    if MANIFEST.exists():
        old = __import__("json").loads(MANIFEST.read_text(encoding="utf-8"))
        if old.get("catalogue") == PC:
            report["regions"] = old.get("regions", {})
            report["failures"] = old.get("failures", [])
            report["read_test"] = old.get("read_test")
            print(f"resuming: {len(report['regions'])} regions recorded")

    for name, region in regions.items():
        if name in report["regions"]:
            continue
        try:
            report["regions"][name] = region_stats(region, args.window_days)
            r = report["regions"][name]
            print(f"{name:16s} scenes={r['scenes']:4d} clear={r['scenes_under_10pct_cloud']:3d}")
        except Exception as exc:
            report["failures"].append({"region": name, "stage": "search",
                                       "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name:16s} FAILED {exc}")
        C.write_json(MANIFEST, report)

    if not report["read_test"]:
        try:
            rec, err = read_test(args.read_test, regions[args.read_test])
            if err:
                report["failures"].append({"stage": "read-test", "error": err})
            else:
                report["read_test"] = rec
                print(f"\nread test {rec['scene']}: {rec['bytes']/1e6:.1f} MB "
                      f"over {len(rec['bands'])} bands")
        except Exception as exc:
            report["failures"].append({"stage": "read-test",
                                       "error": f"{type(exc).__name__}: {exc}"})
            print(f"read test FAILED: {exc}")

    C.write_json(MANIFEST, report)
    print(f"\ns2_availability.json written | regions: {len(report['regions'])} | "
          f"failures: {len(report['failures'])}")
    return 0 if report["read_test"] else 1


if __name__ == "__main__":
    sys.exit(main())

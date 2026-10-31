"""Phase 0: Landsat 8/9 Collection 2 Level-2 audit and AOI-window read test.

Catalogue: Planetary Computer (USGS C2 L2 mirror), assets signed with the
short-lived SAS token. Only the AOI window is transferred.
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
MANIFEST = C.PROJECT / "data" / "manifests" / "landsat_availability.json"
SR_BANDS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22")
QA_BANDS = ("qa_pixel", "qa_radsat")
PLATFORMS = ("landsat-8", "landsat-9")


def sas_for(href):
    u = urllib.parse.urlparse(href)
    account = u.netloc.split(".")[0]
    container = u.path.strip("/").split("/")[0]
    r = C.get_json(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/"
                   f"{account}/{container}")
    return r["token"]


def search(bbox, start, end, limit=200):
    doc = C.post_json(f"{PC}/search", {
        "collections": ["landsat-c2-l2"], "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit})
    return [f for f in doc.get("features", [])
            if f["properties"].get("platform") in PLATFORMS]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=str(C.PROJECT / "configs" / "regions.yaml"))
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--read-test", default="hiroshima")
    args = ap.parse_args(argv)

    regions = yaml.safe_load(Path(args.regions).read_text(encoding="utf-8"))["regions"]
    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
              "catalogue": PC, "collection": "landsat-c2-l2",
              "platforms": list(PLATFORMS), "regions": {},
              "read_test": None, "failures": []}

    for name, region in regions.items():
        dates = sorted({dt.date.fromisoformat(d) for d in region["event_dates"]})
        if not dates:
            continue
        try:
            feats = search(region["bbox"], dates[0] - dt.timedelta(days=args.window_days),
                           dates[-1] + dt.timedelta(days=args.window_days))
            clouds = [float(f["properties"]["eo:cloud_cover"]) for f in feats
                      if f["properties"].get("eo:cloud_cover") is not None]
            days = sorted({f["properties"]["datetime"][:10] for f in feats})
            report["regions"][name] = {
                "bbox": region["bbox"], "scenes": len(feats),
                "date_min": days[0] if days else None,
                "date_max": days[-1] if days else None,
                "cloud_median": round(sorted(clouds)[len(clouds) // 2], 2) if clouds else None,
                "scenes_under_10pct_cloud": sum(1 for c in clouds if c <= 10),
            }
            print(f"{name:16s} L8/9 scenes={len(feats):3d} "
                  f"clear={report['regions'][name]['scenes_under_10pct_cloud']:3d}")
        except Exception as exc:
            report["failures"].append({"region": name, "stage": "search",
                                       "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name:16s} FAILED {exc}")

    name = args.read_test
    region = regions[name]
    dates = sorted({dt.date.fromisoformat(d) for d in region["event_dates"]})
    feats = search(region["bbox"], dates[0] - dt.timedelta(days=365),
                   dates[-1] + dt.timedelta(days=365))
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    if not feats:
        report["failures"].append({"stage": "read-test", "error": "no scenes"})
    else:
        f = feats[0]
        token = sas_for(f["assets"]["red"]["href"])
        outdir = C.PROJECT / "data" / "raw" / "landsat" / f["id"]
        rec = {"scene": f["id"], "region": name,
               "platform": f["properties"].get("platform"),
               "datetime": f["properties"]["datetime"],
               "cloud_cover": f["properties"].get("eo:cloud_cover"),
               "collection_number": f["properties"].get("landsat:collection_number"),
               "processing_level": f["properties"].get("landsat:processing_level"),
               "bands": {}, "bytes": 0}
        for band in SR_BANDS + QA_BANDS:
            a = f["assets"].get(band)
            if not a:
                continue
            try:
                info = C.window_read(f"{a['href']}?{token}", region["bbox"],
                                     outdir / f"{band}_aoi.tif")
                if info is None:
                    continue
                rec["bands"][band] = info
                rec["bytes"] += info["bytes"]
            except Exception as exc:
                report["failures"].append({"band": band, "stage": "window-read",
                                           "error": f"{type(exc).__name__}: {exc}"})
        mtl = f["assets"].get("mtl.json")
        if mtl:
            try:
                n, digest = C.download(f"{mtl['href']}?{token}",
                                       outdir / "mtl.json")
                rec["mtl"] = {"bytes": n, "sha256": digest}
                rec["bytes"] += n
            except Exception as exc:
                report["failures"].append({"stage": "mtl",
                                           "error": f"{type(exc).__name__}: {exc}"})
        report["read_test"] = rec
        print(f"\nread test {f['id']}: {rec['bytes']/1e6:.1f} MB over "
              f"{len(rec['bands'])} bands | platform {rec['platform']}")

    C.write_json(MANIFEST, report)
    print(f"landsat_availability.json written | failures: {len(report['failures'])}")
    return 0 if report["read_test"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 0: EnMAP L2A inventory (catalogue only) and access probe.

The catalogue (DLR EOC STAC) is public; the files sit behind the EOC CAS login.
This script counts what exists per region and records whether download access
works, so the report can state the admission result instead of assuming it.
"""

import argparse
import csv
import datetime as dt
import re
import sys
import urllib.parse
from pathlib import Path

import yaml
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
ARCHIVE_START = "2022-04-27"
CSV_OUT = C.PROJECT / "data" / "manifests" / "enmap_inventory.csv"
SUMMARY = C.PROJECT / "data" / "manifests" / "enmap_inventory_summary.json"


def search(bbox, start, end, limit=200):
    doc = C.post_json(f"{BASE}/search", {
        "collections": ["ENMAP_HSI_L2A"], "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit})
    return doc.get("features", [])


def quality_of(props):
    for key in ("enmap:overallQuality", "enmap:overall_quality", "quality"):
        if key in props:
            return props[key]
    return None


def probe_download(feat):
    """One CAS login + one asset HEAD; returns a status record, never raises."""
    out = {"login": "not-attempted", "asset_status": None, "note": ""}
    env = C.load_env()
    if "ENMAP_USER" not in env:
        out["note"] = "no ENMAP_USER in .env"
        return out
    href = feat["assets"]["vnir"]["href"]
    s = requests.Session()
    lp = s.get(href, timeout=90, headers={"User-Agent": C.UA})
    if "download.geoservice.dlr.de" in lp.url and "login" not in lp.url:
        out["login"] = "already-authorised"
        return out
    block = re.search(r"<form[^>]*>(?:(?!</form>).)*name=\"password\".*?</form>",
                      lp.text, re.S)
    if not block:
        out["login"] = "no-login-form"
        out["note"] = f"landed on {lp.url[:80]}"
        return out
    block = block.group(0)
    data = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', block))
    for n in re.findall(r'<input[^>]*name="([^"]+)"', block):
        data.setdefault(n, "")
    action = (re.search(r'<form[^>]*action="([^"]+)"', block) or [None, "login"])[1]
    data.update({"username": env["ENMAP_USER"], "password": env["ENMAP_PASS"]})
    r = s.post(urllib.parse.urljoin(lp.url, action), data=data, timeout=120,
               headers={"User-Agent": C.UA, "Referer": lp.url})
    out["login"] = f"http-{r.status_code}"
    if r.status_code != 200 or "login" in r.url:
        out["note"] = "credentials rejected by the EOC CAS"
        return out
    h = s.head(href, timeout=120, allow_redirects=True)
    out["asset_status"] = h.status_code
    out["asset_type"] = h.headers.get("content-type", "")
    out["asset_bytes"] = int(h.headers.get("content-length", 0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=str(C.PROJECT / "configs" / "regions.yaml"))
    ap.add_argument("--cloud-max", type=float, default=20.0)
    ap.add_argument("--probe", action="store_true", help="also test download access")
    args = ap.parse_args(argv)

    regions = yaml.safe_load(Path(args.regions).read_text(encoding="utf-8"))["regions"]
    today = dt.date.today().isoformat()
    rows, summary = [], {"archive_start": ARCHIVE_START, "cloud_max": args.cloud_max,
                         "per_region": {}, "probe": None,
                         "generated": dt.datetime.now().isoformat(timespec="seconds")}

    for name, region in regions.items():
        try:
            feats = search(region["bbox"], ARCHIVE_START, today)
        except Exception as exc:
            summary["per_region"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{name:16s} FAILED {exc}")
            continue
        good = []
        for f in feats:
            p = f["properties"]
            cc = p.get("eo:cloud_cover")
            row = {"region": name, "scene_id": f["id"],
                   "datetime": p.get("datetime", "")[:19],
                   "cloud_cover": cc, "snow_cover": p.get("eo:snow_cover"),
                   "quality": quality_of(p), "biome": p.get("enmap:biomeType"),
                   "sun_elevation": p.get("view:sun_elevation"),
                   "proj_epsg": p.get("proj:epsg")}
            rows.append(row)
            if cc is not None and float(cc) <= args.cloud_max:
                good.append(row)
        summary["per_region"][name] = {"scenes": len(feats), "usable": len(good),
                                       "date_min": min((r["datetime"][:10] for r in rows
                                                        if r["region"] == name), default=None),
                                       "date_max": max((r["datetime"][:10] for r in rows
                                                        if r["region"] == name), default=None)}
        print(f"{name:16s} EnMAP scenes={len(feats):4d} usable(cloud<={args.cloud_max:g}%)={len(good):4d}")

    usable_regions = [n for n, s in summary["per_region"].items()
                      if s.get("usable", 0) > 0]
    summary["usable_scenes"] = sum(s.get("usable", 0) for s in summary["per_region"].values())
    summary["usable_regions"] = len(usable_regions)
    summary["admission"] = {
        "requires_scenes": 20, "requires_regions": 5,
        "scenes_ok": summary["usable_scenes"] >= 20,
        "regions_ok": len(usable_regions) >= 5,
    }
    summary["admission"]["passed"] = (summary["admission"]["scenes_ok"]
                                       and summary["admission"]["regions_ok"])
    print(f"\nusable scenes {summary['usable_scenes']} across "
          f"{len(usable_regions)} regions | catalogue admission: "
          f"{'PASS' if summary['admission']['passed'] else 'FAIL'}")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["region", "scene_id", "datetime", "cloud_cover",
                            "snow_cover", "quality", "biome", "sun_elevation",
                            "proj_epsg"])
        w.writeheader()
        w.writerows(rows)

    if args.probe and rows:
        all_feats = []
        for name, region in list(regions.items())[:5]:
            try:
                all_feats += search(region["bbox"], ARCHIVE_START, today, limit=5)
            except Exception:
                pass
        if all_feats:
            summary["probe"] = probe_download(all_feats[0])
            print("download probe:", summary["probe"])

    C.write_json(SUMMARY, summary)
    print(f"\n{CSV_OUT.name} rows: {len(rows)} | {SUMMARY.name} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())


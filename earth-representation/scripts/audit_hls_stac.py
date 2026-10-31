"""Phase 0: HLS V2.0 availability audit and small-sample download verification.

Queries the official LP DAAC catalogue (CMR-STAC) for HLSS30_2.0 / HLSL30_2.0
over the configured regions, records counts/dates/cloud cover, then downloads
two S30 and two L30 granules with the Earthdata token and verifies they read.

Writes data/manifests/hls_availability.json and data/raw/hls/<granule>/.
"""

import argparse
import datetime as dt
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

S30, L30 = "HLSS30_2.0", "HLSL30_2.0"
CODE = {S30: "s30", L30: "l30"}
HALF_WINDOW_DAYS = 180
SKIP_ASSET_PREFIXES = ("s3_", "thumbnail", "browse")


def region_dates(region):
    """Query windows: one per event date, collapsed to the first and last date."""
    days = sorted({dt.date.fromisoformat(d) for d in region["event_dates"]})
    if not days:
        return []
    picks = [days[0]] if len(days) == 1 else [days[0], days[-1]]
    return [(d - dt.timedelta(days=HALF_WINDOW_DAYS),
             d + dt.timedelta(days=HALF_WINDOW_DAYS)) for d in picks]


def audit_region(name, region):
    rows = {"bbox": region["bbox"], "event_dates": region["event_dates"],
            "collections": {}}
    collected = {}
    for coll in (S30, L30):
        seen, dates, clouds, tiles = {}, [], [], set()
        for start, end in region_dates(region):
            feats, matched = C.search(coll, region["bbox"], start, end, limit=500)
            for f in feats:
                seen[f["id"]] = f
        for gid, f in seen.items():
            props = f["properties"]
            dates.append(props["datetime"][:10])
            cc = props.get("eo:cloud_cover")
            if cc is not None:
                clouds.append(float(cc))
            tiles.add(gid.split(".")[2])
        rows["collections"][coll] = {
            "granules": len(seen),
            "unique_tiles": sorted(tiles),
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "cloud_cover_median": float(np.median(clouds)) if clouds else None,
            "cloud_cover_p10": float(np.percentile(clouds, 10)) if clouds else None,
            "granule_ids": sorted(seen)[:400],
        }
        collected.update(seen)
    return rows, collected


def aoi_quality(feat, bbox, header, probe_dir):
    """AOI-window Fmask summary: how much is real data, and how much is clear.

    Sorting by catalogue cloud cover is not enough: a granule that barely
    touches the AOI reports 0% cloud because it is mostly fill, and a fully
    covered granule can still be almost entirely cloud.
    """
    import rasterio

    a = feat["assets"].get("Fmask")
    if not a:
        return None
    dst = probe_dir / f"{feat['id']}_fmask_probe.tif"
    info = C.window_read(a["href"], bbox, dst, header=header)
    if info is None:
        return {"fill": 1.0, "clear": 0.0}
    with rasterio.open(dst) as ds:
        arr = ds.read(1)
    fill = float(np.mean((arr & 1) != 0))
    bad = sum(1 << b for b in (0, 1, 2, 3, 4))      # fill, cloud, adj-cloud, shadow, snow
    clear = float(np.mean((arr & bad) == 0))
    return {"fill": fill, "clear": clear}


def pick_for_download(all_feats, code, regions, header, n=2, probe=12):
    """Highest AOI clear fraction, one region each, among least cloudy candidates."""
    cands = [f for f in all_feats.values() if f"hls.{code}." in f["id"].lower()]
    cands.sort(key=lambda f: (f["properties"].get("eo:cloud_cover", 100),
                              f["properties"]["datetime"]))
    probe_dir = C.PROJECT / ".tmp" / "fmask_probe"
    scored = []
    for f in cands[:probe]:
        try:
            q = aoi_quality(f, regions[f["_region"]]["bbox"], header, probe_dir)
        except Exception:
            q = {"fill": 1.0, "clear": 0.0}
        scored.append((q["clear"], f))
        f["_aoi_quality"] = q
        print(f"    probe {f['id'][:28]:28s} clear={q['clear']:.3f} fill={q['fill']:.3f}")
    scored.sort(key=lambda t: -t[0])

    picked, used = [], set()
    for clear, f in scored:
        if clear < 0.10:
            continue
        if f["_region"] not in used:
            picked.append(f)
            used.add(f["_region"])
        if len(picked) == n:
            break
    if len(picked) < n:  # fewer than n regions qualify: reuse regions, best first
        for clear, f in scored:
            if f not in picked and clear >= 0.10:
                picked.append(f)
            if len(picked) == n:
                break
    return picked


def verify_granule(gdir):
    """Read-verification of the AOI windows and product metadata."""
    out = {"raster": {}, "checks": {}, "errors": []}
    tifs = sorted(gdir.glob("*_aoi.tif"))
    for t in tifs:
        try:
            out["raster"][t.name] = {"structure": C.raster_report(t),
                                     "pixels": C.sample_stats(t)}
        except Exception as exc:
            out["errors"].append(f"{t.name}: {type(exc).__name__}: {exc}")

    b04 = gdir / "B04_aoi.tif"
    if b04.exists():
        import rasterio
        with rasterio.open(b04) as ds:
            px = abs(ds.transform.a)
        out["checks"]["crs_is_utm"] = str(rasterio.open(b04).crs).startswith("EPSG:32")
        out["checks"]["pixel_size_m"] = round(px, 3)
        out["checks"]["is_30m"] = 29.0 <= px <= 31.0
    fmask = gdir / "Fmask_aoi.tif"
    if fmask.exists():
        import rasterio
        with rasterio.open(fmask) as ds:
            a = ds.read(1)
        flags = {"fill": 1 << 0, "cloud": 1 << 1, "adj_cloud": 1 << 2,
                 "cloud_shadow": 1 << 3, "snow": 1 << 4, "water": 1 << 5}
        out["checks"]["fmask_flag_fractions"] = {
            k: round(float(np.mean((a & v) != 0)), 4) for k, v in flags.items()}
        bad = sum(1 << b for b in (0, 1, 2, 3, 4))
        out["checks"]["aoi_clear_fraction"] = round(float(np.mean((a & bad) == 0)), 4)
        out["checks"]["fmask_distinct_values"] = int(len(np.unique(a)))
    xml = next((x for x in gdir.glob("*.xml")), None)
    if xml:
        text = xml.read_text(encoding="utf-8", errors="replace")
        out["checks"]["metadata_xml"] = {
            "bytes": xml.stat().st_size,
            "scale_factor": (re.search(r"SCALE_FACTOR[^0-9]*([0-9.eE+-]+)", text) or [None, None])[1],
            "add_offset": (re.search(r"ADD_OFFSET[^0-9-]*(-?[0-9.eE+-]+)", text) or [None, None])[1],
            "has_fmask_desc": "Fmask" in text,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=str(C.PROJECT / "configs" / "regions.yaml"))
    ap.add_argument("--download", type=int, default=2, help="granules per collection")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args(argv)

    regions = yaml.safe_load(Path(args.regions).read_text(encoding="utf-8"))["regions"]
    env = C.load_env()
    token = C.earthdata_token(env)
    headers = {"Authorization": f"Bearer {token}"}

    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
              "collections": [S30, L30], "window_days": HALF_WINDOW_DAYS,
              "regions": {}, "downloads": [], "failures": []}

    all_feats = {}
    for name, region in regions.items():
        try:
            stats, feats = audit_region(name, region)
            report["regions"][name] = stats
            for gid, f in feats.items():
                f["_region"] = name
                all_feats[gid] = f
            s = stats["collections"][S30]["granules"]
            l = stats["collections"][L30]["granules"]
            print(f"{name:16s} S30={s:4d} L30={l:4d}")
        except Exception as exc:
            report["failures"].append({"region": name, "stage": "search",
                                       "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name:16s} SEARCH FAILED {exc}")

    total = Counter()
    for name, r in report["regions"].items():
        for coll, s in r["collections"].items():
            total[coll] += s["granules"]
    report["totals"] = dict(total)
    print("totals:", dict(total))

    if not args.no_download:
        hdr = C.header_file(token, C.PROJECT / ".tmp" / "edl_header.txt")
        for coll in (S30, L30):
            picked = pick_for_download(all_feats, CODE[coll], regions, hdr,
                                       n=args.download)
            for f in picked:
                gid = f["id"]
                gdir = C.PROJECT / "data" / "raw" / "hls" / gid
                bbox = regions[f["_region"]]["bbox"]
                rec = {"granule": gid, "collection": coll, "region": f["_region"],
                       "datetime": f["properties"]["datetime"],
                       "cloud_cover": f["properties"].get("eo:cloud_cover"),
                       "aoi_quality_probe": f.get("_aoi_quality"),
                       "aoi_bbox": bbox, "assets": [], "bytes": 0}
                try:
                    for band in ("B04", "Fmask", "SZA", "SAA"):
                        a = f["assets"].get(band)
                        if not a:
                            continue
                        info = C.window_read(a["href"], bbox,
                                             gdir / f"{band}_aoi.tif", header=hdr)
                        if info is None:
                            rec.setdefault("skipped", []).append(
                                f"{band}: AOI outside scene")
                            continue
                        rec["assets"].append({"band": band, **info})
                        rec["bytes"] += info["bytes"]
                    xml = f["assets"].get("metadata")
                    if xml and xml.get("href", "").startswith("https://"):
                        dst = gdir / xml["href"].rsplit("/", 1)[-1]
                        n, digest = C.download(xml["href"], dst, headers=headers)
                        rec["assets"].append({"band": "metadata", "bytes": n,
                                              "sha256": digest})
                        rec["bytes"] += n
                    rec["verification"] = verify_granule(gdir)
                    print(f"  ok {gid} {rec['bytes']/1e6:.2f} MB "
                          f"({len(rec['assets'])} assets, window)")
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    report["failures"].append({"granule": gid, "stage": "window-read",
                                               "error": rec["error"]})
                    print(f"  FAIL {gid}: {exc}")
                report["downloads"].append(rec)

    C.write_json(C.PROJECT / "data" / "manifests" / "hls_availability.json", report)
    ok = sum(1 for d in report["downloads"] if "error" not in d)
    print(f"\nhls_availability.json written | granules downloaded ok: {ok} | "
          f"failures: {len(report['failures'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

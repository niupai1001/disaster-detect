"""Download EnMAP with a browser session cookie instead of a login flow.

When the account is logged in interactively (OTP, delegated identity, whatever),
the resulting cookie is enough: the download service only checks the session.
"""

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cookie", required=True,
                    help="raw Cookie header copied from the browser")
    ap.add_argument("--bbox", nargs=4, type=float, default=[11.0, 44.0, 11.6, 44.6])
    ap.add_argument("--start", default="2023-04-01")
    ap.add_argument("--end", default="2023-07-01")
    ap.add_argument("--assets", nargs="*", default=["vnir", "swir"])
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data" / "enmap"))
    args = ap.parse_args(argv)

    s = requests.Session()
    for part in args.cookie.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            s.cookies.set(k.strip(), v.strip())

    feats = requests.post(f"{STAC}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": args.bbox,
        "datetime": f"{args.start}T00:00:00Z/{args.end}T23:59:59Z",
        "limit": args.limit}, timeout=120, headers=UA).json().get("features", [])
    print("scenes:", len(feats))
    if not feats:
        return 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for feat in feats:
        for key in args.assets:
            asset = feat["assets"].get(key)
            if not asset:
                continue
            url = asset["href"]
            with s.get(url, headers={**UA, "Range": "bytes=0-1023"},
                       timeout=120, stream=True) as probe:
                ctype = probe.headers.get("content-type", "")
                print(f"{key}: {probe.status_code} {ctype}")
                if "text/html" in ctype:
                    continue
            with s.get(url, headers=UA, stream=True, timeout=600) as r:
                if r.status_code != 200 or "text/html" in r.headers.get("content-type", ""):
                    print("  download refused")
                    continue
                dst = outdir / url.rsplit("/", 1)[-1]
                tmp = dst.with_suffix(".part")
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(4 << 20):
                        fh.write(chunk)
                        n += len(chunk)
                tmp.replace(dst)
                print(f"  saved {dst.name} {n/1e6:.1f} MB")
                saved += 1
    print("saved files:", saved)
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

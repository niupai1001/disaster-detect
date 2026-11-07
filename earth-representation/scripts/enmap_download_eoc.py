"""Download EnMAP L2A products via the DLR EOC login that actually works.

Account lemon123 authenticates at sso.eoc.dlr.de and holds the EnMAP Access
Service subscription, so the plain CAS form is enough - no CODE-DE delegation
and no OTP on this route.
"""

import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"


def env():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def login(session, href, user, password):
    lp = session.get(href, headers=UA, timeout=120)
    if "login" not in lp.url and "sso" not in lp.url:
        return
    # parse with the shared helper: a hand-rolled regex here dropped hidden
    # fields whose attributes are ordered value-before-name, which silently
    # left the session unauthorised
    from scripts.enmap_download import forms as _forms
    eoc = next((f for f in _forms(lp.text)
                if "password" in f["fields"] and f["fields"].get("client_name") is None), None)
    if eoc is None:
        raise RuntimeError("EOC login form not found")
    fields = dict(eoc["fields"])
    fields.setdefault("_eventId", "submit")
    fields.update({"username": user, "password": password})
    r = session.post(urllib.parse.urljoin(lp.url, eoc["action"] or "login"), data=fields,
                     headers={**UA, "Referer": lp.url}, timeout=120)
    if "login" in r.url:
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        raise RuntimeError(f"login failed: {text[:160]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--bbox", nargs=4, type=float, default=[11.0, 44.0, 11.6, 44.6])
    ap.add_argument("--start", default="2023-04-01")
    ap.add_argument("--end", default="2023-07-01")
    ap.add_argument("--assets", nargs="*", default=["vnir", "swir"])
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--head-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "enmap"))
    args = ap.parse_args(argv)

    e = env()
    user = args.user or e.get("ENMAP_USER")
    password = args.password or e.get("ENMAP_PASS")

    feats = requests.post(f"{STAC}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": args.bbox,
        "datetime": f"{args.start}T00:00:00Z/{args.end}T23:59:59Z",
        "limit": args.scenes}, timeout=120, headers=UA).json().get("features", [])
    print(f"scenes: {len(feats)}")
    if not feats:
        return 1

    s = requests.Session()
    s.trust_env = False
    first = feats[0]["assets"][args.assets[0]]["href"]
    login(s, first, user, password)
    print("login: ok")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for feat in feats:
        for key in args.assets:
            asset = feat["assets"].get(key)
            if not asset:
                continue
            url = asset["href"]
            name = url.rsplit("/", 1)[-1]
            h = s.head(url, headers=UA, timeout=120, allow_redirects=True)
            size = int(h.headers.get("content-length", 0))
            print(f"  {key:5s} {name[:40]:42s} {size/1e6:8.1f} MB {h.headers.get('content-type','')}")
            if args.head_only or not size:
                continue
            dst = outdir / name
            if dst.exists() and dst.stat().st_size == size:
                print("    already present")
                continue
            tmp = dst.with_suffix(".part")
            n = 0
            with s.get(url, headers=UA, stream=True, timeout=1800) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(8 << 20):
                        fh.write(chunk)
                        n += len(chunk)
            tmp.replace(dst)
            print(f"    saved {n/1e6:.1f} MB")
            rows.append({"scene": feat["id"], "asset": key, "file": str(dst),
                         "bytes": n})

    if rows:
        (outdir / "manifest.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

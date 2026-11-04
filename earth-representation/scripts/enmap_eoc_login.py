"""Log in with the DLR EOC account (the one that carries the EnMAP subscription).

The CODE-DE button goes to CloudFerro, whose accounts are separate and have no
EnMAP entitlement - that is what produced the 401 after a valid OTP. The EOC
form on the same page authenticates against the DLR identity that owns the
subscriptions shown in "Manage Access Permissions".
"""

import argparse
import re
import sys
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from scripts.enmap_download import forms  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
SERVICE = "https://download.geoservice.dlr.de/ENMAP/files/"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--otp", default=None)
    args = ap.parse_args(argv)

    feat = requests.post(f"{STAC}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
        "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
        timeout=120, headers=UA).json()["features"][0]
    href = feat["assets"]["vnir"]["href"]

    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
    print("landed:", lp.url[:80])

    # the plain EOC form (fm1), not the CODE-DE delegation
    eoc = next((f for f in forms(lp.text) if "password" in f["fields"]
                and f["fields"].get("client_name") is None), None)
    if eoc is None:
        print("EOC login form not found")
        return 1
    data = dict(eoc["fields"])
    data.update({"username": args.user, "password": args.password})
    data.setdefault("_eventId", "submit")
    r = s.post(urllib.parse.urljoin(lp.url, eoc["action"] or "login"), data=data,
               headers={**UA, "Referer": lp.url}, timeout=120)
    print("eoc login ->", r.status_code, r.url[:90])

    otp = next((f for f in forms(r.text) if "otp" in f["fields"]), None)
    if otp is not None:
        if not args.otp:
            text = re.sub(r"<[^>]+>", " ", r.text)
            print("OTP required;", re.sub(r"\s+", " ", text)[:200])
            return 1
        d = dict(otp["fields"])
        d.update({"otp": args.otp, "login": "Sign In"})
        r = s.post(urllib.parse.urljoin(r.url, otp["action"]), data=d,
                   headers={**UA, "Referer": r.url}, timeout=120)
        print("otp ->", r.status_code, r.url[:90])

    if "login" in r.url:
        err = re.search(r'id="input-error"[^>]*>(.*?)</span>', r.text, re.S)
        msg = re.sub(r"<[^>]+>", " ", err.group(1)).strip() if err else "no message"
        print("still on login page:", msg[:120])
        return 1

    root = s.get(SERVICE, headers=UA, timeout=120)
    print("service:", root.status_code, root.url[:80])

    a = s.get(href, headers={**UA, "Range": "bytes=0-2047"}, timeout=120)
    print("asset:", a.status_code, a.headers.get("content-type"),
          a.content[:4].hex(), len(a.content), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

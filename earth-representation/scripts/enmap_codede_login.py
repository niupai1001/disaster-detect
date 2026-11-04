"""Log in to the EnMAP download service through the CODE-DE (EO-Lab) route.

Two hops: the CAS page delegates to the CloudFerro Keycloak realm, and the
Keycloak form needs the username/password. After that the asset URL returns the
file instead of the login page.
"""

import argparse
import re
import sys
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}


def env():
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def forms(html):
    out = []
    for m in re.finditer(r"<form[^>]*>(.*?)</form>", html, re.S | re.I):
        block = m.group(0)
        action = (re.search(r'action="([^"]+)"', block) or [None, ""])[1]
        fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', block))
        for n in re.findall(r'<input[^>]*name="([^"]+)"', block):
            fields.setdefault(n, "")
        out.append({"action": action, "fields": fields})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--download", default=None, help="save this band to the given path")
    args = ap.parse_args(argv)

    e = env()
    user = args.user or e.get("CODEDE_USER") or e.get("ENMAP_USER")
    password = args.password or e.get("CODEDE_PASS") or e.get("ENMAP_PASS")

    BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
    feat = requests.post(f"{BASE}/search", json={
        "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
        "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
        timeout=120, headers=UA).json()["features"][0]
    href = feat["assets"]["vnir"]["href"]
    print("scene:", feat["id"])

    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
    if "login" not in lp.url:
        print("already authorised; asset reachable")
        return 0

    # hop 1: CAS delegates to the CODE-DE identity provider
    delegated = next((f for f in forms(lp.text)
                      if f["fields"].get("client_name") == "codede3_enmap-dl"), None)
    if delegated is None:
        print("no CODE-DE form on the login page")
        return 1
    r = s.post(urllib.parse.urljoin(lp.url, delegated["action"]),
               data=delegated["fields"], headers={**UA, "Referer": lp.url},
               timeout=120)
    print("delegation ->", r.status_code, r.url[:90])

    # hop 2: Keycloak form
    if "cloudferro" in r.url or "openid-connect" in r.url:
        kc = next((f for f in forms(r.text) if "password" in f["fields"]), None)
        if kc is None:
            print("no Keycloak password form; page head:",
                  re.sub(r"\s+", " ", r.text)[:200])
            return 1
        data = dict(kc["fields"])
        data["username"] = user
        data["password"] = password
        r2 = s.post(urllib.parse.urljoin(r.url, kc["action"]), data=data,
                    headers={**UA, "Referer": r.url}, timeout=120)
        print("keycloak ->", r2.status_code, r2.url[:90])
        err = re.search(r"(Invalid[^<]{0,80}|error[^<]{0,60})", r2.text)
        if err:
            print("message:", err.group(0)[:100])

    # verify: the asset must now come back as a file
    h = s.get(href, headers={**UA, "Range": "bytes=0-1023"}, timeout=120,
              stream=True)
    print("asset after login:", h.status_code, h.headers.get("content-type"),
          h.headers.get("content-range"))
    if args.download and h.status_code in (200, 206):
        Path(args.download).parent.mkdir(parents=True, exist_ok=True)
        with open(args.download, "wb") as fh:
            for chunk in h.iter_content(1 << 20):
                fh.write(chunk)
        print("saved", args.download, Path(args.download).stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

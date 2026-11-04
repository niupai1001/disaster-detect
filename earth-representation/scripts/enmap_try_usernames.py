"""Try the plausible username spellings against the CODE-DE realm.

Keycloak answered "Invalid username or password" for the address as written, so
the point is to find which form the account actually has - not to guess at
random.
"""

import re
import urllib.parse

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
PASSWORD = "Lemon123!Lemon23"
CANDIDATES = ["260415@bjfu.edu.cn", "2260415@bjfu.edu.cn", "260415", "2260415"]

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
feat = requests.post(f"{BASE}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["vnir"]["href"]


def attempt(user):
    s = requests.Session()
    lp = s.get(href, headers=UA, timeout=120)
    blk = re.search(r'<form[^>]*id="formcodede3_enmap-dl"[^>]*>(.*?)</form>', lp.text, re.S).group(0)
    fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', blk))
    act = re.search(r'action="([^"]+)"', blk).group(1)
    r = s.post(urllib.parse.urljoin(lp.url, act), data=fields,
               headers={**UA, "Referer": lp.url}, timeout=120)
    m = re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', r.text, re.S)
    if not m:
        return "no-keycloak-form", None
    action = m.group(1).replace("&amp;", "&")
    block = re.search(r'<form[^>]*id="kc-form-login"[^>]*>(.*?)</form>', r.text, re.S).group(1)
    data = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', block))
    for n in re.findall(r'<input[^>]*name="([^"]+)"', block):
        data.setdefault(n, "")
    data.update({"username": user, "password": PASSWORD, "login": "Sign In"})
    r2 = s.post(action, data=data, headers={**UA, "Referer": r.url}, timeout=120)
    msg = re.search(r'id="input-error"[^>]*>(.*?)</span>', r2.text, re.S)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", msg.group(1))).strip() if msg else ""
    ok = "login-actions/authenticate" not in r2.url
    return ("ok" if ok else (text or "still on login page")), r2.url[:60]


for user in CANDIDATES:
    status, url = attempt(user)
    print(f"{user:24s} -> {status} | {url}")

"""Complete the Keycloak form post and show the server's actual error text."""

import re
import urllib.parse

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
USER = "260415@bjfu.edu.cn"
PASSWORD = "Lemon123!Lemon23"

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
feat = requests.post(f"{BASE}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["vnir"]["href"]

s = requests.Session()
lp = s.get(href, headers=UA, timeout=120)
blk = re.search(r'<form[^>]*id="formcodede3_enmap-dl"[^>]*>(.*?)</form>', lp.text, re.S | re.I).group(0)
fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', blk))
act = re.search(r'action="([^"]+)"', blk).group(1)
r = s.post(urllib.parse.urljoin(lp.url, act), data=fields,
           headers={**UA, "Referer": lp.url}, timeout=120)

m = re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', r.text, re.S)
action = m.group(1).replace("&amp;", "&")
block = re.search(r'<form[^>]*id="kc-form-login"[^>]*>(.*?)</form>', r.text, re.S).group(1)
data = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', block))
for n in re.findall(r'<input[^>]*name="([^"]+)"', block):
    data.setdefault(n, "")
print("kc fields:", sorted(data))
data["username"] = USER
data["password"] = PASSWORD
data["login"] = "Sign In"

r2 = s.post(action, data=data, headers={**UA, "Referer": r.url,
                                        "Content-Type": "application/x-www-form-urlencoded"},
            timeout=120)
print("post ->", r2.status_code, "|", r2.url[:80])

# the message lives in an alert/error element; print its text
for pat in (r'id="input-error"[^>]*>(.*?)</span>',
            r'class="[^"]*alert[^"]*"[^>]*>(.*?)</div>',
            r'class="[^"]*error[^"]*"[^>]*>(.*?)</div>'):
    for mm in re.finditer(pat, r2.text, re.S):
        txt = re.sub(r"<[^>]+>", " ", mm.group(1))
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            print("message:", txt[:160])

"""What does the CODE-DE page show right after posting 2260415@...?

No 'invalid credentials' this time, so the account may be asking for a required
action (verify email, update password, accept terms). Print what the page says.
"""

import re
import urllib.parse

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
USER = "2260415@bjfu.edu.cn"
PASSWORD = "Lemon123!Lemon23"

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
feat = requests.post(f"{BASE}/search", json={
    "collections": ["ENMAP_HSI_L2A"], "bbox": [11.0, 44.0, 11.6, 44.6],
    "datetime": "2023-04-01T00:00:00Z/2023-07-01T23:59:59Z", "limit": 1},
    timeout=120, headers=UA).json()["features"][0]
href = feat["assets"]["vnir"]["href"]

s = requests.Session()
lp = s.get(href, headers=UA, timeout=120)
blk = re.search(r'<form[^>]*id="formcodede3_enmap-dl"[^>]*>(.*?)</form>', lp.text, re.S).group(0)
fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', blk))
act = re.search(r'action="([^"]+)"', blk).group(1)
r = s.post(urllib.parse.urljoin(lp.url, act), data=fields,
           headers={**UA, "Referer": lp.url}, timeout=120)

action = re.search(r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"', r.text, re.S).group(1)
action = action.replace("&amp;", "&")
block = re.search(r'<form[^>]*id="kc-form-login"[^>]*>(.*?)</form>', r.text, re.S).group(1)
data = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', block))
for n in re.findall(r'<input[^>]*name="([^"]+)"', block):
    data.setdefault(n, "")
data.update({"username": USER, "password": PASSWORD, "login": "Sign In"})
r2 = s.post(action, data=data, headers={**UA, "Referer": r.url}, timeout=120)

print("status:", r2.status_code)
print("url:", r2.url[:100])
text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r2.text, flags=re.S)
text = re.sub(r"<[^>]+>", " ", text)
text = re.sub(r"\s+", " ", text).strip()
print("page text:", text[:600])
print("forms:", re.findall(r'<form[^>]*id="([^"]*)"[^>]*action="([^"]*)"', r2.text, re.S)[:3])

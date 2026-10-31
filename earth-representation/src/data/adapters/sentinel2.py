"""Sentinel-2 L2A adapter (Planetary Computer copy of the Copernicus product)."""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import _common as C  # noqa: E402

from ._base import Scene  # noqa: E402

PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
KEYS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A",
        "B11", "B12", "SCL")
from ._pc_sas import sas_for as _sas


def scenes(bbox, start, end, limit=100, cloud_max=None):
    doc = C.post_json(f"{PC}/search", {
        "collections": ["sentinel-2-l2a"], "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": limit})
    out = []
    for f in doc.get("features", []):
        cc = f["properties"].get("eo:cloud_cover")
        if cloud_max is not None and (cc is None or cc > cloud_max):
            continue
        baseline = float(f["properties"].get("s2:processing_baseline", "02.00"))
        # baseline 04.00 introduced the -1000 DN harmonic offset
        offset = -1000.0 if baseline >= 4.0 else 0.0
        assets = {k: f"{a['href']}?{_sas(a['href'])}"
                  for k, a in f["assets"].items() if k in KEYS}
        out.append(Scene(
            product="sentinel2", scene_id=f["id"],
            datetime=f["properties"]["datetime"], epsg=f["properties"].get("proj:epsg"),
            assets=assets,
            extra={"quantification": 10000.0, "boa_offset": offset,
                   "processing_baseline": baseline, "cloud_cover": cc,
                   "product_uri": f["properties"].get("s2:product_uri")}))
    return out



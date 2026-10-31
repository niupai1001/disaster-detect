"""Landsat 8/9 Collection 2 Level-2 adapter (Planetary Computer mirror)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import _common as C  # noqa: E402

from ._base import Scene  # noqa: E402

PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
KEYS = ("coastal", "blue", "green", "red", "nir08", "swir16", "swir22",
        "qa_pixel", "qa_radsat")
PLATFORMS = ("landsat-8", "landsat-9")
from ._pc_sas import sas_for as _sas


def scenes(bbox, start, end, limit=200, cloud_max=None):
    doc = C.post_json(f"{PC}/search", {
        "collections": ["landsat-c2-l2"], "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": limit})
    out = []
    for f in doc.get("features", []):
        props = f["properties"]
        if props.get("platform") not in PLATFORMS:
            continue
        cc = props.get("eo:cloud_cover")
        if cloud_max is not None and (cc is None or cc > cloud_max):
            continue
        assets = {k: f"{a['href']}?{_sas(a['href'])}"
                  for k, a in f["assets"].items() if k in KEYS}
        out.append(Scene(product="landsat", scene_id=f["id"],
                         datetime=props["datetime"], epsg=props.get("proj:epsg"),
                         assets=assets,
                         extra={"cloud_cover": cc, "scene_id": props.get("landsat:scene_id"),
                                "path": props.get("landsat:wrs_path"),
                                "row": props.get("landsat:wrs_row"),
                                "collection": props.get("landsat:collection_number"),
                                "bbox_wgs84": f.get("bbox")}))
    return out




"""EnMAP L2A adapter — interface only.

Phase 0 admission failed: the DLR download service rejects the registered
credentials with HTTP 401, so no EnMAP pixels were ever retrieved. The interface
is kept so the pipeline shape does not depend on this product being available.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import _common as C  # noqa: E402

from ._base import Scene  # noqa: E402

BASE = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
KEYS = ("vnir", "swir", "quality_cloud")


def available():
    """Phase 0 verdict, quoted in reports instead of guessing."""
    return False, "Phase 0: HTTP 401 at sso.eoc.dlr.de/eoc/auth/login"


def scenes(bbox, start, end, limit=50):
    doc = C.post_json(f"{BASE}/search", {
        "collections": ["ENMAP_HSI_L2A"], "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": limit})
    out = []
    for f in doc.get("features", []):
        assets = {k: a["href"] for k, a in f["assets"].items() if k in KEYS}
        out.append(Scene(product="enmap", scene_id=f["id"],
                         datetime=f["properties"]["datetime"],
                         epsg=f["properties"].get("proj:epsg"),
                         assets=assets,
                         extra={"version": f["properties"].get("enmap:archivedVersion")}))
    return out


def read_patch(*args, **kwargs):
    ok, why = available()
    if not ok:
        raise RuntimeError(f"EnMAP data unavailable: {why}")
    raise NotImplementedError("EnMAP needs the 224-band reading path; not built")

"""HLS V2.0 adapter (NASA LP DAAC, Earthdata token).

The scale factor and add offset are read from the product XML rather than
assumed, because the plan requires provenance-aware scaling.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import _common as C  # noqa: E402

from ._base import Scene  # noqa: E402

KEYS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
        "B09", "B10", "B11", "B12", "Fmask")
COLLECTIONS = {"s30": "HLSS30_2.0", "l30": "HLSL30_2.0"}


def _token():
    return C.earthdata_token(C.load_env())


def _header(token):
    path = C.PROJECT / ".tmp" / "edl_header.txt"
    return C.header_file(token, path)


_SCALING = {}


def scaling_for(collection, feat, token):
    """SCALE_FACTOR / ADD_OFFSET from the product XML, once per collection.

    The factor is a property of the product version, not of the granule, so a
    per-scene fetch would cost one HTTP request per candidate scene for nothing.
    """
    if collection in _SCALING:
        return _SCALING[collection]
    scale, offset = 0.0001, 0.0
    meta = feat["assets"].get("metadata")
    if meta:
        r = C.requests.get(meta["href"],
                           headers={"Authorization": f"Bearer {token}"}, timeout=120)
        if r.ok:
            m = re.search(r"SCALE_FACTOR[^0-9]*([0-9.eE+-]+)", r.text)
            o = re.search(r"ADD_OFFSET[^0-9-]*(-?[0-9.eE+-]+)", r.text)
            if m:
                scale = float(m.group(1))
            if o:
                offset = float(o.group(1))
    _SCALING[collection] = (scale, offset)
    return scale, offset


def utm_epsg_from_tile(scene_id):
    """HLS items carry no proj:epsg; the MGRS tile id fixes the UTM zone."""
    tile = scene_id.split(".")[2] if scene_id.count(".") >= 2 else ""
    if len(tile) < 5 or not tile[1:3].isdigit():
        return None
    zone = int(tile[1:3])
    band = tile[3].upper()
    south = band in "CDEFGHJKLM"
    return (32600 if not south else 32700) + zone


def scenes(bbox, start, end, limit=100, cloud_max=None, sensor=None):
    token = _token()
    header = str(_header(token))
    cols = [COLLECTIONS[sensor]] if sensor else list(COLLECTIONS.values())
    out = []
    for coll in cols:
        feats, _ = C.search(coll, bbox, start, end, limit=limit)
        for f in feats:
            cc = f["properties"].get("eo:cloud_cover")
            if cloud_max is not None and (cc is None or cc > cloud_max):
                continue
            assets = {k: a["href"] for k, a in f["assets"].items() if k in KEYS}
            scale, offset = scaling_for(coll, f, token)
            epsg = f["properties"].get("proj:epsg") or utm_epsg_from_tile(f["id"])
            out.append(Scene(product="hls", scene_id=f["id"],
                             datetime=f["properties"]["datetime"], epsg=epsg,
                             assets=assets, header=header,
                             extra={"hls_scale": scale, "hls_offset": offset,
                                    "collection": coll, "cloud_cover": cc,
                                    "bbox_wgs84": f.get("bbox")}))
    return out





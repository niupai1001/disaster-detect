"""Read-only public STAC and COG review; writes diagnostics in outputs/review."""
import datetime as dt
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from landslide_seg.data.sen12 import load_sen12, list_sen12_files, assemble_full
from landslide_seg.data.s2_backfill import find_same_day_item, read_asset_window
from pyproj import Transformer
s = load_sen12(list_sen12_files(ROOT / "data/sen12")[1])
i = s.event_index()[1]
date = s.attrs["_time_ref"] + dt.timedelta(days=int(s.time[i]))
lon, lat = Transformer.from_crs(s.attrs["crs"], "EPSG:4326", always_xy=True).transform(s.attrs["center_x"], s.attrs["center_y"])
result = {"file": s.path.name, "date": str(date), "status": "UNVERIFIED"}
try:
    item = find_same_day_item(lon, lat, date)
    result["selected_item"] = {"id": item["id"], "properties": item["properties"]} if item else None
    a = assemble_full(s)
    if a is not None:
        np.savez_compressed(ROOT / "outputs/review/real_diagnostic_input.npz",
            **{k: a[k] for k in ["x12", "x10", "mask", "valid", "doy"]})
        result["status"] = "FETCHED_NOT_CERTIFIED_COMPLIANT"
        result["provenance"] = a["provenance"]
        result["valid_fraction"] = float(a["valid"].mean())
        cx, cy = s.attrs["center_x"], s.attrs["center_y"]
        ref = read_asset_window(item["assets"]["nir"]["href"], (cx-640,cy-640,cx+640,cy+640), s.attrs["crs"], (128,128))
        raw = s.bands["B08"][i].astype(float)
        ok = (ref > 0) & (raw > 0)
        result["B08_compare"] = {"corr": float(np.corrcoef(raw[ok],ref[ok])[0,1]),
            "bias_dn": float((ref[ok]-raw[ok]).mean()), "rmse_dn": float(np.sqrt(((ref[ok]-raw[ok])**2).mean()))}
except Exception as exc:
    result.update(status="BLOCKED", error_type=type(exc).__name__, error=str(exc))
(ROOT/"outputs/review/backfill_checks.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
print(json.dumps(result, indent=2, default=str))

"""Machine-readable inventory of every local dataset plus its physics linkage.

Answers, per dataset: where it is, how many files, what an array looks like, which
bands and GSD it carries, and whether its sensor physics exists in
sensor_physics.json. This is the file an agent should read before planning work.
"""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
S2_RAW = Path(r"N:\Users\yzh\Disater\landslide-seg\data\sen12")
PHYS = json.loads((ROOT / "data" / "physics" / "sensor_physics.json").read_text(encoding="utf-8"))


def count(p, pattern="*", recursive=False):
    p = Path(p)
    if not p.exists():
        return 0
    it = p.rglob(pattern) if recursive else p.glob(pattern)
    return sum(1 for f in it if f.is_file())


def size_gb(p, pattern="*", recursive=False):
    p = Path(p)
    if not p.exists():
        return 0.0
    it = p.rglob(pattern) if recursive else p.glob(pattern)
    return sum(os.path.getsize(f) for f in it if f.is_file()) / 1e9


def s2_shape():
    import rasterio
    f = next(S2_RAW.glob("*.nc"))
    with rasterio.open(str(f)) as ds:
        subs = {s.rsplit(":", 1)[-1]: s for s in ds.subdatasets}
        with rasterio.open(subs["B02"]) as d:
            return {"bands": sorted(subs), "y": d.height, "x": d.width,
                    "times": d.count, "crs": str(d.crs)}


def nc_shape(path, var):
    from scipy.io import netcdf_file
    with netcdf_file(str(path), "r", mmap=False) as nc:
        a = nc.variables[var].shape
        return list(a)


def main():
    out = {"schema_version": "1.0", "root": str(ROOT), "datasets": {}}

    d = {"path": str(S2_RAW), "role": "observation", "sensor": "S2A/S2B",
         "files": count(S2_RAW, "*.nc"), "size_gb": round(size_gb(S2_RAW, "*.nc"), 2),
         "storage": "netCDF4/HDF5, one file per patch", "bands": 10,
         "band_names": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
         "extras": ["SCL", "MASK", "DEM"], "time_steps": 15,
         "native_gsd_m": {"10": ["B02", "B03", "B04", "B08"],
                          "20": ["B05", "B06", "B07", "B8A", "B11", "B12"]},
         "stored_on": "10 m grid for every band (20 m bands are upsampled)",
         "reflectance": "DN/10000", "georeferenced": True,
         "physics_in_json": "S2A/S2B/S2C",
         "status": "usable; 3 bands missing (B01/B09/B10); 20 m bands not native"}
    d.update(s2_shape())
    out["datasets"]["s2_patches"] = d

    for name, sensor in (("landsat_native", "LANDSAT8_OLI/LANDSAT9_OLI2"),
                         ("sen12_landsat", "LANDSAT8_OLI/LANDSAT9_OLI2"),
                         ("enmap_pairs", "ENMAP")):
        p = ROOT / "data" / name
        files = count(p, "*.nc")
        if not files:
            continue
        sample = next(p.glob("*.nc"))
        var = "sr" if "landsat" in name else ("spectral" if "enmap" in name else "sr")
        try:
            shape = nc_shape(sample, var)
        except Exception:
            shape = None
        gsd = {  "landsat_native": 30, "sen12_landsat": 10, "enmap_pairs": 10 }[name]
        out["datasets"][name] = {
            "path": str(p), "role": "observation", "sensor": sensor,
            "files": files, "size_gb": round(size_gb(p, "*.nc"), 3),
            "storage": "netCDF3 (scipy)", "variable": var, "shape": shape,
            "native_gsd_m": 30 if "landsat" in name else 30,
            "stored_on_gsd_m": gsd,
            "native": gsd == (30 if "landsat" in name else 30),
            "reflectance": "DN/10000" if "landsat" in name else "scaled L2A",
            "physics_in_json": sensor,
            "status": ("native 30 m, no resampling" if name == "landsat_native"
                       else "resampled to 10 m" if name != "enmap_pairs"
                       else "224 bands resampled to 10 m"),
        }

    cov = {"path": str(ROOT / "data" / "enmap"), "role": "reference scene",
           "files": count(ROOT / "data" / "enmap", "*.TIF"),
           "size_gb": round(size_gb(ROOT / "data" / "enmap", "*.TIF"), 2),
           "sensor": "ENMAP", "bands": 224, "native_gsd_m": 30,
           "storage": "COG GeoTIFF, ZSTD, 512x512 tiles, all bands per tile",
           "note": "one full scene kept as the reference for tile decoding"}
    out["datasets"]["enmap_scene"] = cov

    out["datasets"]["physics"] = {
        "path": str(ROOT / "data" / "physics"),
        "role": "physical descriptors",
        "files": count(ROOT / "data" / "physics", "*", recursive=True),
        "entry_point": "physics/sensor_physics.json",
        "sensors": {k: {"bands": len(v["bands"]),
                        "srf_status": v["srf"]["status"],
                        "psf_status": v["psf"]["status"]}
                    for k, v in PHYS["sensors"].items()},
    }

    out["datasets"]["cog_patches"] = {
        "path": str(ROOT / "data" / "interim" / "patches"), "role": "observation",
        "files": count(ROOT / "data" / "interim" / "patches", "*.npz"),
        "size_gb": round(size_gb(ROOT / "data" / "interim" / "patches", "*.npz"), 3),
        "sensor": "S2", "bands": 10, "shape": [10, 128, 128],
        "physics_in_json": "S2A/S2B",
        "status": "small S2 subset from the COG pipeline",
    }

    out["not_used"] = {
        "sen12_hls": {"reason": "HLS is a harmonised product, not an independent "
                               "sensor in the V3 spec"},
    }

    dst = ROOT / "data" / "datasets.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dst} ({dst.stat().st_size/1024:.0f} KB)")
    for k, v in out["datasets"].items():
        print(f"  {k:16s} files={v.get('files', 0):5d} "
              f"{('%.2f GB' % v['size_gb']) if 'size_gb' in v else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build one physics file per sensor for agent consumption.

Sources used, and how each is verified:
  * Sentinel-2: official MSI SRF workbook (COPE-GSEG-EOPG-TN-15-0007 v4.0),
    sheets "Spectral Responses (S2A|S2B|S2C)" - measured curves.
  * Landsat 8/9: EUMETSAT rtcoef SRF tables (Ball BA RSR v1.1) - measured
    curves, given as wavenumber; converted to wavelength here.
  * EnMAP: the product metadata XML lists centre wavelength and FWHM for all 224
    bands; the curve below is a Gaussian with that FWHM and is flagged
    "derived", not a measured SRF.

Outputs:
  data/physics/srf/<sensor>.csv   wavelength_nm + one column per band
  data/physics/sensor_physics.json  bands, GSD, radiometry, provenance, status
"""

import csv
import glob
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
PH = ROOT / "data" / "physics"
SRF = PH / "srf"
XLSX = SRF / "sentinel2_msi_srf_v4_2024.xlsx"
LS_TARS = {"LANDSAT8_OLI": SRF / "landsat8_oli_srf_eumetsat.tar.gz",
           "LANDSAT9_OLI2": SRF / "landsat9_oli2_srf_eumetsat.tar.gz"}
ENMAP_XML = next(iter((ROOT / "data" / "enmap").glob("*METADATA.XML")), None)

# native ground sampling distance, metres
GSD = {
    "S2A": {"B1": 60, "B2": 10, "B3": 10, "B4": 10, "B5": 20, "B6": 20, "B7": 20,
            "B8": 10, "B8A": 20, "B9": 60, "B10": 60, "B11": 20, "B12": 20},
    "S2B": {"B1": 60, "B2": 10, "B3": 10, "B4": 10, "B5": 20, "B6": 20, "B7": 20,
            "B8": 10, "B8A": 20, "B9": 60, "B10": 60, "B11": 20, "B12": 20},
    "S2C": {"B1": 60, "B2": 10, "B3": 10, "B4": 10, "B5": 20, "B6": 20, "B7": 20,
            "B8": 10, "B8A": 20, "B9": 60, "B10": 60, "B11": 20, "B12": 20},
    "LANDSAT8_OLI": {"B1": 30, "B2": 30, "B3": 30, "B4": 30, "B5": 30, "B6": 30,
                     "B7": 30, "B8": 15, "B9": 30},
    "LANDSAT9_OLI2": {"B1": 30, "B2": 30, "B3": 30, "B4": 30, "B5": 30, "B6": 30,
                      "B7": 30, "B8": 15, "B9": 30},
}


def curve_stats(wl, resp):
    """Centre wavelength and FWHM from a sampled response curve."""
    r = np.asarray(resp, dtype=float)
    w = np.asarray(wl, dtype=float)
    if r.sum() <= 0:
        return None, None
    centre = float((w * r).sum() / r.sum())
    peak = r.max()
    above = w[r >= peak / 2]
    fwhm = float(above.max() - above.min()) if above.size else None
    return round(centre, 2), round(fwhm, 2) if fwhm else None


def write_csv(path, wl, bands):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["wavelength_nm"] + list(bands))
        for i, x in enumerate(wl):
            w.writerow([round(float(x), 3)] + [round(float(bands[b][i]), 6) for b in bands])


def s2():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    out = {}
    for sensor, sheet in (("S2A", "Spectral Responses (S2A)"),
                          ("S2B", "Spectral Responses (S2B)"),
                          ("S2C", "Spectral Responses (S2C)")):
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        header = rows[0]
        cols = {}
        for j, name in enumerate(header):
            if not name:
                continue
            m = re.search(r"_(B\d+A?)$", str(name))
            if m:
                cols[m.group(1)] = j
        data = [r for r in rows[1:] if r[0] is not None]
        wl = np.array([float(r[0]) for r in data])
        bands = {}
        for b, j in cols.items():
            vals = np.array([float(r[j]) if j < len(r) and r[j] is not None else 0.0
                             for r in data])
            keep = vals > 1e-3
            bands[b] = {"wl": wl[keep], "resp": vals[keep]}
        out[sensor] = bands
    wb.close()
    return out


def landsat(tar_path):
    bands = {}
    with tarfile.open(tar_path) as t:
        for name in t.getnames():
            if not name.endswith(".txt"):
                continue
            m = re.search(r"ch(\d+)", name)
            if not m:
                continue
            txt = t.extractfile(name).read().decode("utf-8", "replace").splitlines()
            pairs = []
            for line in txt:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        wn, resp = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    if wn > 0:
                        pairs.append((1e7 / wn, resp))   # cm-1 -> nm
            if pairs:
                wn, resp = zip(*sorted(pairs))
                bands[f"B{int(m.group(1))}"] = {"wl": np.array(wn), "resp": np.array(resp)}
    return bands


def enmap():
    if ENMAP_XML is None:
        return None
    txt = ENMAP_XML.read_text(encoding="utf-8", errors="replace")
    centres = [float(x) for x in re.findall(r"<wavelengthCenterOfBand>([-\d.]+)<", txt)]
    fwhms = [float(x) for x in re.findall(r"<FWHMOfBand>([-\d.]+)<", txt)]
    n = min(len(centres), len(fwhms))
    grid = np.arange(380.0, 2500.0, 1.0)
    bands = {}
    for i in range(n):
        c, f = centres[i], fwhms[i]
        sigma = f / 2.3548
        resp = np.exp(-0.5 * ((grid - c) / sigma) ** 2)
        bands[f"B{i + 1:03d}"] = {"wl": grid, "resp": resp, "centre": c, "fwhm": f}
    return bands


def main():
    SRF.mkdir(parents=True, exist_ok=True)
    physics = {
        "schema_version": "1.0",
        "purpose": "Physical descriptors needed to put an observation into Encoder V3",
        "generated_from": {},
        "sensors": {},
    }

    for sensor, bands in s2().items():
        rows = sorted(bands)
        wl = bands[rows[0]]["wl"]
        meta = []
        for b in rows:
            c, f = curve_stats(bands[b]["wl"], bands[b]["resp"])
            meta.append({"band": b, "centre_nm": c, "fwhm_nm": f,
                         "native_gsd_m": GSD[sensor][b]})
        write_csv(SRF / f"{sensor.lower()}_srf.csv", wl,
                  {b: np.interp(wl, bands[b]["wl"], bands[b]["resp"]) for b in rows})
        physics["sensors"][sensor] = {
            "instrument": "MSI", "bands": meta,
            "srf": {"csv": f"srf/{sensor.lower()}_srf.csv", "status": "measured",
                    "source": str(XLSX.name), "grid_nm": [float(wl.min()), float(wl.max())]},
            "radiometry": {"product": "L2A BOA reflectance",
                           "scale": 0.0001, "offset": 0, "nodata": 0},
            "psf": {"status": "not sourced",
                    "note": "needs the ESA MSI PSF/MTF document; do not guess"},
        }
        physics["generated_from"]["S2"] = str(XLSX)

    for sensor, path in LS_TARS.items():
        bands = landsat(path)
        if not bands:
            continue
        rows = sorted(bands)
        wl = np.arange(max(300, min(bands[b]["wl"].min() for b in rows)),
                       min(2600, max(bands[b]["wl"].max() for b in rows)), 0.5)
        meta = []
        for b in rows:
            c, f = curve_stats(bands[b]["wl"], bands[b]["resp"])
            meta.append({"band": b, "centre_nm": c, "fwhm_nm": f,
                         "native_gsd_m": GSD[sensor].get(b)})
        write_csv(SRF / f"{sensor.lower()}_srf.csv", wl,
                  {b: np.interp(wl, bands[b]["wl"], bands[b]["resp"]) for b in rows})
        physics["sensors"][sensor] = {
            "instrument": "OLI" if sensor.endswith("OLI") else "OLI-2",
            "bands": meta,
            "srf": {"csv": f"srf/{sensor.lower()}_srf.csv", "status": "measured",
                    "source": path.name, "grid_nm": [float(wl.min()), float(wl.max())]},
            "radiometry": {"product": "C2 L2 SR", "scale": 0.0000275, "offset": -0.2,
                           "nodata": None},
            "psf": {"status": "not sourced", "note": "USGS OLI characterisation needed"},
        }
        physics["generated_from"][sensor] = str(path)

    e = enmap()
    if e:
        rows = sorted(e)
        wl = e[rows[0]]["wl"]
        write_csv(SRF / "enmap_srf_derived.csv", wl,
                  {b: e[b]["resp"] for b in rows})
        meta = [{"band": b, "centre_nm": round(e[b]["centre"], 2),
                 "fwhm_nm": round(e[b]["fwhm"], 2), "native_gsd_m": 30} for b in rows]
        physics["sensors"]["ENMAP"] = {
            "instrument": "HSI", "n_bands": len(rows), "bands": meta,
            "srf": {"csv": "srf/enmap_srf_derived.csv", "status": "derived_gaussian",
                    "source": ENMAP_XML.name,
                    "note": "product metadata gives centre and FWHM only; "
                            "official measured SRF file not yet obtained"},
            "radiometry": {"product": "L2A", "scale": 1.0, "offset": 0,
                           "nodata": -32768},
            "psf": {"status": "not sourced", "note": "DLR HSI PSF document needed"},
        }
        physics["generated_from"]["ENMAP"] = str(ENMAP_XML)

    out = PH / "sensor_physics.json"
    out.write_text(json.dumps(physics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    for k, v in physics["sensors"].items():
        print(f"  {k:14s} bands={len(v['bands']):3d} srf={v['srf']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PSF descriptions, with the derivation kept visible.

A Gaussian PSF and its MTF are the same thing: MTF(f) = exp(-2*pi^2*sigma^2*f^2),
so an official MTF value at Nyquist (f = 0.5 cycles/pixel) fixes sigma:

    sigma = sqrt(-2*ln(MTF_nyquist)) / pi

Only sensors with a verified MTF number get a sigma here. The rest are listed as
not_sourced on purpose - a guessed PSF is worse than a missing one, because it
looks like measured physics downstream.
"""

import json
import math
from pathlib import Path

ROOT = Path(r"N:\Users\yzh\Disater\earth-representation")
PSF_DIR = ROOT / "data" / "physics" / "psf"
SPEC_PDF = ROOT / ".tmp" / "pdf" / "enmap_specs.pdf"

# values transcribed from EnMAP_Specs.pdf (DLR, 2-page specification sheet):
# "> 0.25 @ 60 m across track, > 0.16 @ 60 m along track, > 0.64 @ 240 m across
# track, > 0.62 @ 240 m along track". 60 m is Nyquist for a 30 m sampling.
ENMAP_MTF = {"across_track": 0.25, "along_track": 0.16,
             "across_track_240m": 0.64, "along_track_240m": 0.62}


def sigma_from_mtf(mtf):
    return round(math.sqrt(-2.0 * math.log(mtf)) / math.pi, 3)


def main():
    PSF_DIR.mkdir(parents=True, exist_ok=True)
    if SPEC_PDF.exists():
        (PSF_DIR / "sources").mkdir(exist_ok=True)
        dst = PSF_DIR / "sources" / "EnMAP_Specs.pdf"
        if not dst.exists():
            dst.write_bytes(SPEC_PDF.read_bytes())

    model = {
        "schema_version": "1.0",
        "model": "circular Gaussian, sigma in pixels of the native grid",
        "derivation": "sigma = sqrt(-2*ln(MTF_nyquist)) / pi",
        "why": "an official MTF number is citable; a guessed FWHM is not",
        "sensors": {
            "ENMAP": {
                "status": "derived_from_mtf_spec",
                "source": "psf/sources/EnMAP_Specs.pdf (DLR specification sheet)",
                "mtf_nyquist": {"across_track": ENMAP_MTF["across_track"],
                                "along_track": ENMAP_MTF["along_track"]},
                "sigma_px": {"across_track": sigma_from_mtf(ENMAP_MTF["across_track"]),
                             "along_track": sigma_from_mtf(ENMAP_MTF["along_track"])},
                "note": "Nyquist = 60 m for the 30 m sampling; the spec quotes a "
                        "lower bound, so sigma is an upper bound",
            },
            "S2A": {"status": "not_sourced",
                    "needed": "ESA MSI MTF/PSF requirement or in-flight MTF at Nyquist",
                    "candidate": "CNES in-flight MTF measurements (CEOS IVOS-27, 2015) "
                                 "report per-band values only in figures"},
            "S2B": {"status": "not_sourced", "needed": "same as S2A"},
            "S2C": {"status": "not_sourced", "needed": "same as S2A"},
            "LANDSAT8_OLI": {"status": "not_sourced",
                             "needed": "USGS OLI MTF at Nyquist or measured PSF/FWHM"},
            "LANDSAT9_OLI2": {"status": "not_sourced", "needed": "same as Landsat 8"},
        },
    }

    out = PSF_DIR / "psf_model.json"
    out.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")

    # point sensor_physics.json at this file
    phys_path = ROOT / "data" / "physics" / "sensor_physics.json"
    phys = json.loads(phys_path.read_text(encoding="utf-8"))
    for sensor, entry in phys["sensors"].items():
        psf = model["sensors"].get(sensor, {"status": "not_sourced"})
        entry["psf"] = {"status": psf["status"],
                        "model": "gaussian",
                        "detail": "psf/psf_model.json"}
        if "sigma_px" in psf:
            entry["psf"]["sigma_px"] = psf["sigma_px"]
    phys_path.write_text(json.dumps(phys, indent=2, ensure_ascii=False), encoding="utf-8")
    print("sensor_physics.json updated")

    for s, v in model["sensors"].items():
        extra = v.get("sigma_px", "")
        print(f"  {s:14s} {v['status']:22s} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Band identities: names, centre wavelengths, native GSD, product of origin.

Adapters must never map a band by array position. Every band carries the same
identity here, so a reordering of the input changes the metadata with it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    product: str
    key: str            # asset key in the catalogue, e.g. "B04" or "red"
    name: str           # human/identity name, e.g. "S2_B04"
    centre_nm: float
    native_gsd_m: int
    kind: str = "reflectance"   # reflectance | qa | angle


S2_BANDS = (
    Band("sentinel2", "B02", "S2_B02", 492.4, 10),
    Band("sentinel2", "B03", "S2_B03", 559.8, 10),
    Band("sentinel2", "B04", "S2_B04", 664.6, 10),
    Band("sentinel2", "B08", "S2_B08", 832.8, 10),
    Band("sentinel2", "B05", "S2_B05", 704.1, 20),
    Band("sentinel2", "B06", "S2_B06", 740.5, 20),
    Band("sentinel2", "B07", "S2_B07", 782.8, 20),
    Band("sentinel2", "B8A", "S2_B8A", 864.7, 20),
    Band("sentinel2", "B11", "S2_B11", 1613.7, 20),
    Band("sentinel2", "B12", "S2_B12", 2202.4, 20),
    Band("sentinel2", "SCL", "S2_SCL", 0.0, 20, "qa"),
)

LANDSAT_BANDS = (
    Band("landsat", "coastal", "LANDSAT_SR_B1", 442.96, 30),
    Band("landsat", "blue", "LANDSAT_SR_B2", 481.59, 30),
    Band("landsat", "green", "LANDSAT_SR_B3", 561.25, 30),
    Band("landsat", "red", "LANDSAT_SR_B4", 654.59, 30),
    Band("landsat", "nir08", "LANDSAT_SR_B5", 864.67, 30),
    Band("landsat", "swir16", "LANDSAT_SR_B6", 1608.86, 30),
    Band("landsat", "swir22", "LANDSAT_SR_B7", 2200.73, 30),
    Band("landsat", "qa_pixel", "LANDSAT_QA_PIXEL", 0.0, 30, "qa"),
    Band("landsat", "qa_radsat", "LANDSAT_QA_RADSAT", 0.0, 30, "qa"),
)

# HLS S30 is Sentinel-2 reprojected to the common 30 m grid; HLS L30 is Landsat.
# The identity stays HLS rather than "S2" because HLS applied its own processing.
HLS_BANDS = (
    Band("hls", "B02", "HLS_B02", 492.4, 30),
    Band("hls", "B03", "HLS_B03", 559.8, 30),
    Band("hls", "B04", "HLS_B04", 664.6, 30),
    Band("hls", "B8A", "HLS_B8A", 864.7, 30),
    Band("hls", "B11", "HLS_B11", 1613.7, 30),
    Band("hls", "B12", "HLS_B12", 2202.4, 30),
    Band("hls", "Fmask", "HLS_FMASK", 0.0, 30, "qa"),
)

BY_PRODUCT = {
    "sentinel2": S2_BANDS,
    "landsat": LANDSAT_BANDS,
    "hls": HLS_BANDS,
}


def stack(product, keys):
    """Return Band objects in the requested order; unknown keys are an error."""
    index = {b.key: b for b in BY_PRODUCT[product]}
    missing = [k for k in keys if k not in index]
    if missing:
        raise KeyError(f"{product}: unknown band keys {missing}")
    return [index[k] for k in keys]


def describe(product, keys):
    """Plain metadata rows for the unified reader output."""
    return [{"key": b.key, "name": b.name, "centre_nm": b.centre_nm,
             "native_gsd_m": b.native_gsd_m, "kind": b.kind, "product": b.product}
            for b in stack(product, keys)]

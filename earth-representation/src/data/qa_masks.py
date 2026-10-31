"""QA decoding for the three optical products, and the one validity rule.

Each product has its own bit semantics; the pipeline only cares about
"usable for loss / training", so everything is reduced to a boolean mask plus
the named layers that a report may need.
"""

import numpy as np

# Sentinel-2 Scene Classification Layer: only these classes are bad for training.
S2_SCL_BAD = {
    "nodata": 0,
    "saturated": 1,
    "cloud_shadow": 3,
    "cloud_medium": 8,
    "cloud_high": 9,
    "thin_cirrus": 10,
    "snow": 11,
}

# HLS Fmask bit positions (V2.0).
HLS_FMASK_BITS = {"fill": 0, "cloud": 1, "adjacent_cloud": 2,
                  "cloud_shadow": 3, "snow": 4, "water": 5}

# Landsat C2 QA_PIXEL bit positions.
LANDSAT_QA_BITS = {"fill": 0, "dilated_cloud": 1, "cirrus": 2, "cloud": 3,
                   "cloud_shadow": 4, "snow": 5, "clear": 6, "water": 7}


def s2_scl_masks(scl):
    """Split an SCL array into named boolean masks (True = that class present)."""
    out = {name: (scl == value) for name, value in S2_SCL_BAD.items()}
    out["valid"] = ~np.isin(scl, list(S2_SCL_BAD.values()))
    return out


def hls_fmask_masks(fmask):
    out = {name: ((fmask >> bit) & 1).astype(bool)
           for name, bit in HLS_FMASK_BITS.items()}
    bad = np.zeros(fmask.shape, dtype=bool)
    for name in ("fill", "cloud", "adjacent_cloud", "cloud_shadow", "snow"):
        bad |= out[name]
    out["valid"] = ~bad
    return out


def landsat_qa_masks(qa_pixel, qa_radsat=None):
    qa = qa_pixel.astype(np.uint16)
    out = {name: ((qa >> bit) & 1).astype(bool) for name, bit in LANDSAT_QA_BITS.items()}
    bad = np.zeros(qa.shape, dtype=bool)
    for name in ("fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow", "snow"):
        bad |= out[name]
    if qa_radsat is not None:
        out["saturated"] = qa_radsat.astype(np.uint16) != 0
        bad |= out["saturated"]
    out["valid"] = ~bad
    return out


DISPATCH = {
    "sentinel2": lambda layers: s2_scl_masks(layers["SCL"]),
    "hls": lambda layers: hls_fmask_masks(layers["Fmask"]),
    "landsat": lambda layers: landsat_qa_masks(layers["qa_pixel"], layers.get("qa_radsat")),
}


def valid_mask(product, layers):
    """Boolean [H,W] validity for a product, from its own QA layer."""
    if product not in DISPATCH:
        raise KeyError(f"no QA rule for {product}")
    return DISPATCH[product](layers)["valid"]


def lookalike_rule(product):
    """One line describing the rule, for report text."""
    return {
        "sentinel2": "SCL 0/1/3/8/9/10/11 excluded",
        "hls": "Fmask bits fill|cloud|adjacent_cloud|shadow|snow excluded",
        "landsat": "QA_PIXEL fill|dilated_cloud|cirrus|cloud|shadow|snow excluded, QA_RADSAT saturated excluded",
    }[product]

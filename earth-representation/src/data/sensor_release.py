"""Read a frozen two-sensor release without adding members during training."""
import json
from pathlib import Path

import numpy as np

from .triples import S2_NAMES, LANDSAT_NAMES, CENTRE, GSD


def load_release(directory, route):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with np.load(directory / "arrays.npz") as stored:
        arrays = {k:stored[k] for k in stored.files}
    c0 = np.array([CENTRE[k] for k in S2_NAMES], np.float32)
    g0 = np.array([GSD[k] for k in S2_NAMES], np.float32)
    c1 = np.array([CENTRE[k] for k in LANDSAT_NAMES] + [0]*3, np.float32)
    g1 = np.array([GSD[k] for k in LANDSAT_NAMES] + [1]*3, np.float32)
    items = []
    for i, sid in enumerate(manifest["train_ids"]):
        first = arrays["s2"][i]
        if route == "single":
            second, centre, gsd = first, np.r_[c0, c0], np.r_[g0, g0]
            mask = np.ones(20, bool)
        else:
            second = np.pad(arrays["landsat"][i], ((0,3),(0,0),(0,0)))
            centre, gsd = np.r_[c0,c1], np.r_[g0,g1]
            mask = np.array([True]*17+[False]*3)
        items.append(dict(image=np.concatenate([first,second]),
            valid_mask=arrays["valid"][i], centre=centre, gsd=gsd,
            band_mask=mask, bands=S2_NAMES + (S2_NAMES if route=="single" else LANDSAT_NAMES+["padding"]*3),
            path=sid))
    validation = [dict(image=arrays["validation"][i], valid_mask=arrays["validation_valid"][i],
        centre=c0, gsd=g0, bands=S2_NAMES, path=sid)
        for i,sid in enumerate(manifest["validation_ids"])]
    return items, validation, manifest

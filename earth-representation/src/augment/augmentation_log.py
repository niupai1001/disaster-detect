"""Record the parameters of every augmentation so a view is reproducible."""

import json
from pathlib import Path


def record(path, step, view, mode, params, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"step": step, "view": view, "mode": mode,
           "params": params, "seed": seed}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


MODES = ("standard", "generic", "physical", "physical_band", "physical_band_radiometry")

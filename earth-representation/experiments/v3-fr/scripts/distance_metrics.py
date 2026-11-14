"""Native A masking distance and B1 support evidence from fixed-panel arrays."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def run(campaign):
    rows = []
    root = campaign/"s6_component_trials"
    for path in sorted(root.glob("*/val_233/raw_predictions.npz")):
        arm = path.parents[1].name
        with np.load(path) as raw:
            for key in sorted(k for k in raw.files if k.endswith("_target_mask") and "_A_" in k):
                prefix = key.removesuffix("target_mask")
                observation,band = prefix.removesuffix("_").split("_A_",1)
                keep = raw[prefix+"visible_mask"]
                target = raw[key]
                affine = raw[prefix+"affine"]
                if not keep.any():
                    continue
                distance = distance_transform_edt(~keep,sampling=(-affine[4],affine[0]))[target]
                truth = raw[prefix+"truth"][target]
                main = raw[prefix+"main_prediction"][target]
                total = raw[prefix+"total_prediction"][target]
                for lo,hi in ((0,40),(40,80),(80,160),(160,np.inf)):
                    select = (distance>lo)&(distance<=hi)
                    rows.append(dict(arm=arm,observation=observation,band=band,distance_low_m=lo,
                                     distance_high_m=hi if np.isfinite(hi) else None,
                                     targets=int(select.sum()),
                                     mse=float(np.mean((total[select]-truth[select])**2)) if select.any() else None,
                                     main_mse=float(np.mean((main[select]-truth[select])**2)) if select.any() else None))
    assert rows
    with (campaign/"summary/fixed_panel_A_distance.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    run(parser.parse_args().campaign)

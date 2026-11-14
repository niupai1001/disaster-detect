"""Export the immutable full-scene frame for the fixed visual comparison."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from run_v3_fr import bind_runtime,read,write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release",type=Path)
    parser.add_argument("campaign",type=Path)
    args = parser.parse_args()
    bind_runtime(args.release)
    from encoder_v3.latent_response.types import NativeGrid
    from encoder_v3.shared_mae.data import load_data
    from encoder_v3.shared_mae.geometry import metric_geometry
    sets,_ = load_data()
    panel = read(args.campaign/"data_panel.json")
    ids = set(panel["fixed_val"]) | set(panel["historic_display"])
    rows = {}
    for r in sets["VAL"].rows:
        if r["id"] in ids:
            grids = [NativeGrid(tuple(b["transform"]),tuple(b["shape"]),b["crs"]) for b in r["bands"]]
            geometry = metric_geometry([SimpleNamespace(grid=g) for g in grids],40.)
            rows[r["id"]] = dict(origin=geometry.origin,extent=geometry.extent,
                                 shape=geometry.shape,crs=grids[0].crs)
    assert ids==rows.keys()
    write(args.campaign/"fixed_scene_geometry.json",rows)


if __name__=="__main__":
    main()

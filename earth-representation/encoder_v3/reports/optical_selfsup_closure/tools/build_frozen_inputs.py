"""Build or verify the frozen E0 inputs: sampling recipe and fixed VAL panel.

The recipe stores no per-update tensor. The panel stores explicit cases; both
are byte-hashed into config_optical_selfsup_v1.json.

    python reports/optical_selfsup_closure/tools/build_frozen_inputs.py --config encoder_v3/config_optical_selfsup_v1.json --write
    python reports/optical_selfsup_closure/tools/build_frozen_inputs.py --config encoder_v3/config_optical_selfsup_v1.json --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from encoder_v3.data import NativeOpticalDataset  # noqa: E402
from encoder_v3.formal_panels import build_validation_panel_document  # noqa: E402
from encoder_v3.formal_schedule import FAMILIES, _mask_rule, _rectangle_in, build_optical_selfsup_schedule  # noqa: E402
from encoder_v3.prepare_pilot import bounds  # noqa: E402

PANEL_STATUS = "ENGINEERING_FROZEN_RESEARCH_APPROVAL_PENDING"
SELECTION_RULE = ("for each source family: VAL members inside the training native-pixel cap with the richest band count "
                  "(S2=10, Landsat=7), smallest native side then id, up to two per family; "
                  "A = frozen rectangle rule at P; B1 single = the two bands with the lowest and highest measured "
                  "SRF-weighted centroid wavelength; B1 multi = cyclic consecutive windows of length k over that order "
                  "for every k in 2..C-1; each case leaves at least one visible band")


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(document):
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_recipe(config, train):
    sampling = config["sampling"]
    rule = sampling["pool_rule"]
    return build_optical_selfsup_schedule(train, updates=int(sampling["updates"]), seed=int(sampling["seed"]),
                                          slots_per_source=int(sampling["slots_per_source"]),
                                          mask_rule=sampling["mask_rule"], p_values=tuple(sampling["P_values"]),
                                          max_native_pixels=int(rule["max_native_pixels"]))


def select_members(validation, *, per_family=2, max_native_pixels=16384):
    """Richest native band inventory inside the same native-pixel cap as training."""
    chosen = []
    for family, sensors in (("S2", ("S2",)), ("Landsat", ("L8", "L9"))):
        members = [row for row in validation.rows if row["sensor"] in sensors
                   and max(band["shape"][0] * band["shape"][1] for band in row["bands"]) <= max_native_pixels]
        if not members:
            raise ValueError(f"No VAL members inside the native-pixel cap for {family}")
        band_count = max(len(row["bands"]) for row in members)
        pool = [row for row in members if len(row["bands"]) == band_count]
        pool.sort(key=lambda row: (max(max(band["shape"]) for band in row["bands"]), row["id"]))
        if not pool:
            raise ValueError(f"Not enough fixed VAL members for {family}")
        chosen.extend(pool[:per_family])
    return chosen


def band_order(observation):
    """Order native bands by their measured SRF-weighted centroid wavelength."""
    centroids = {}
    for band in observation.bands:
        area = float(torch.trapezoid(band.response, band.wavelengths_nm))
        centroids[band.record_id] = float(torch.trapezoid(band.wavelengths_nm * band.response, band.wavelengths_nm)) / area
    return sorted(observation.bands, key=lambda band: (centroids[band.record_id], band.record_id))


def build_panel(config, validation, *, members=2):
    rule = _mask_rule(config["sampling"]["mask_rule"])
    seed = int(config["validation"]["panel_seed"])
    cases = []
    cap = int(config["sampling"]["pool_rule"]["max_native_pixels"])
    indices = {row["id"]: index for index, row in enumerate(validation.rows)}
    for row in select_members(validation, per_family=members, max_native_pixels=cap):
        left, bottom, _, _ = bounds(row)
        observation = validation[indices[row["id"]]]
        ordered = band_order(observation)
        keys = [band.record_id for band in ordered]
        band_count = len(keys)
        for P in (60.0, 120.0):
            _, _, right, top = bounds(row)
            rectangle = _rectangle_in(left, bottom, right, top, P, seed=seed, update=0, identity=row["id"],
                                      stream="A_rectangle_optical_validation_v1", rule=rule)
            cases.append(dict(panel_id=f"{row['id']}-P{int(P)}-A", source_id=row["id"], kind="A", P_m=P,
                              origin_xy=[left, bottom], rectangle=rectangle))
            for label, band in (("single-low", keys[0]), ("single-high", keys[-1])):
                cases.append(dict(panel_id=f"{row['id']}-P{int(P)}-{label}", source_id=row["id"], kind="B1", P_m=P,
                                  origin_xy=[left, bottom], hidden_band_ids=[band]))
            for count in range(2, band_count):
                start = (count - 1) % band_count
                hidden = [keys[(start + offset) % band_count] for offset in range(count)]
                cases.append(dict(panel_id=f"{row['id']}-P{int(P)}-multi{count}", source_id=row["id"], kind="B1",
                                  P_m=P, origin_xy=[left, bottom], hidden_band_ids=hidden))
    document = build_validation_panel_document(validation, cases, status=PANEL_STATUS)
    document["builder"].update(selection_rule=SELECTION_RULE, panel_seed=seed, max_native_pixels=cap,
                               members=[row["id"] for row in select_members(validation, per_family=members, max_native_pixels=cap)])
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build/verify the frozen E0 recipe and validation panel")
    parser.add_argument("--config", type=Path, default=ROOT / "encoder_v3" / "config_optical_selfsup_v1.json")
    parser.add_argument("--write", action="store_true", help="write the frozen files and bind their hashes into the config")
    parser.add_argument("--verify", action="store_true", help="rebuild and compare against the config hashes")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data = config["data"]
    train = NativeOpticalDataset(Path(data["release"]), Path(data["contracts"]), role="TRAIN")
    validation = NativeOpticalDataset(Path(data["release"]), Path(data["contracts"]), role="VAL")
    recipe = build_recipe(config, train)
    panel = build_panel(config, validation)
    recipe_bytes, panel_bytes = canonical(recipe), canonical(panel)
    recipe_path = args.config.parent / config["sampling"]["recipe"]
    panel_path = args.config.parent / config["validation"]["panel"]
    report = dict(status="OK", recipe=dict(path=str(recipe_path), sha256=sha256_bytes(recipe_bytes),
                                           updates=recipe["updates"], pool=recipe["pool"]["members"]),
                  panel=dict(path=str(panel_path), sha256=sha256_bytes(panel_bytes), cases=len(panel["panels"]),
                             observations=len({case["source_id"] for case in panel["panels"]})))
    if args.write:
        for path, payload in ((recipe_path, recipe_bytes), (panel_path, panel_bytes)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        config["sampling"]["recipe_sha256"] = report["recipe"]["sha256"]
        config["validation"]["panel_sha256"] = report["panel"]["sha256"]
        args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["status"] = "WRITTEN"
    if args.verify:
        expected_recipe = config["sampling"]["recipe_sha256"]
        expected_panel = config["validation"]["panel_sha256"]
        actual_recipe = hashlib.sha256(recipe_path.read_bytes()).hexdigest() if recipe_path.exists() else None
        actual_panel = hashlib.sha256(panel_path.read_bytes()).hexdigest() if panel_path.exists() else None
        report["verify"] = dict(recipe_matches=actual_recipe == expected_recipe == report["recipe"]["sha256"],
                                panel_matches=actual_panel == expected_panel == report["panel"]["sha256"],
                                recorded_recipe_sha256=expected_recipe, recorded_panel_sha256=expected_panel)
        if not all((report["verify"]["recipe_matches"], report["verify"]["panel_matches"])):
            report["status"] = "VERIFY_FAILED"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in ("OK", "WRITTEN") else 4


if __name__ == "__main__":
    raise SystemExit(main())

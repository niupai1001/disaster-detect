"""Supplementary original optical products; no HR/SAR or interpolated inputs."""
import collections
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
import yaml

from .native_data import optical_from_four_view

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"observation_model/outputs/phase0"


def intersects(a,b,margin=0):
    return a[0]<b[2]+margin and b[0]<a[2]+margin and a[1]<b[3]+margin and b[1]<a[3]+margin


def protected_boxes():
    regions=yaml.safe_load((ROOT/"configs/regions.yaml").read_text())["regions"]
    cfg=json.loads((ROOT/"observation_model/config.json").read_text())
    guards=[dict(source=k,bbox=regions[k]["bbox"]) for k in cfg["protected_regions"]]
    p=ROOT/"outputs/verification/worldcover_data/manifest.json"
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            if r["split"] in ["val","test"]:guards.append(dict(source="WorldCover_"+r["sample_id"],bbox=r["bounds_wgs84"]))
    return guards


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--rebuild",action="store_true");args=parser.parse_args()
    accepted=OUT/"release_local_v1/release.json"
    if accepted.exists() and json.loads(accepted.read_text()).get("status")=="ACCEPTED_LOCAL_NATIVE_FOR_MASKED":
        raise ValueError("native inputs are frozen; do not overwrite an accepted release")
    base=ROOT.parent/"landslide-seg/data/use_multisensor/four_view_pilot_v1"
    sites=json.loads((base/"manifest.json").read_text())
    active=ROOT.parent/"landslide-seg/outputs/use_data_integration/active_release/manifest.jsonl"
    holdout=set()
    for line in active.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        if r.get("site_id") and r.get("role") in ["PRETRAIN_VAL","GUARD"]:holdout.add(r["site_id"])
    guards=protected_boxes();cache=OUT/"native_local";cache.mkdir(parents=True,exist_ok=True)
    observations=[];failures=[]
    for i,site in enumerate(sites):
        sid=site["site_id"]
        ring=site["footprint_wgs84"]["coordinates"][0]
        bounds=[min(p[0] for p in ring),min(p[1] for p in ring),max(p[0] for p in ring),max(p[1] for p in ring)]
        hits=[g["source"] for g in guards if intersects(bounds,g["bbox"],.1)]
        if hits:
            failures.append(dict(site_id=sid,reason="protected_geography",guards=hits));continue
        left,bottom,right,top=site["bbox_projected"]
        # Center a1200m native window, aligned to the S2 60m tile lattice.
        left=left+60*round(((right-left)-1200)/120)
        top=top-60*round(((top-bottom)-1200)/120)
        bbox=[left,top-1200,left+1200,top]
        for sensor in ["S2","L8"]:
            path=cache/(sid+"_"+sensor+".npz")
            try:
                if not path.exists() or args.rebuild:
                    arrays,meta=optical_from_four_view(base,site,sensor,bbox)
                    if min(m["valid_fraction"] for m in meta["bands"])<.5:raise ValueError("native joint band QA coverage below50%")
                    np.savez_compressed(path,**arrays)
                    meta.update(role_constraint="heldout" if sid in holdout else "candidate",geographic_bbox=bounds,source_family="original_four_view_L2",cache=str(path),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_manifest=str(base/"manifest.json"))
                    path.with_suffix(".json").write_text(json.dumps(meta,indent=2))
                meta=json.loads(path.with_suffix(".json").read_text())
                observations.append(meta)
            except Exception as exc:failures.append(dict(site_id=sid,sensor=sensor,reason=str(exc)))
        if (i+1)%20==0:print(json.dumps(dict(sites_done=i+1,native_observations=len(observations),exclusions=len(failures))),flush=True)
    (OUT/"native_local_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in observations))
    (OUT/"native_local_exclusions.json").write_text(json.dumps(failures,indent=2))
    counts=dict(collections.Counter(r["sensor"] for r in observations))
    summary=dict(status="native_local_cached_pending_full_release",counts=counts,sites=len({r["source_site"] for r in observations}),exclusions=len(failures),exclusion_reasons=dict(collections.Counter(r["reason"] for r in failures)))
    (OUT/"native_local_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()

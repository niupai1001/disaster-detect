"""Version local native masks without changing radiometry, grids or frozen roles."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import ROOT, rows
from .native_data import aerosol_valid, native_window
from .admit_local import pair_quality


def main():
    folder=ROOT/"observation_model/outputs/phase0/release_local_v2"
    cache=ROOT/"observation_model/outputs/phase0/native_aerosol_v2"
    if (folder/"release.json").exists():
        raise FileExistsError("new release already frozen; inspect it rather than overwrite")
    folder.mkdir(parents=True,exist_ok=True);cache.mkdir(exist_ok=True)
    accepted=[];excluded=[]
    for old in rows():
        row=json.loads(json.dumps(old))
        if row["sensor"]=="L8":
            source=Path(row["bands"][0]["source"]).parent/"qa_aerosol.tif"
            if not source.exists():
                excluded.append(dict(id=row["observation_id"],reason="missing_local_aerosol_QA_no_download"));continue
            qa,qv,transform,crs=native_window(source,row["bbox"],30,bbox_crs=row["crs"])
            assert np.allclose(list(transform)[:6],row["bands"][0]["transform"],rtol=0,atol=1e-6) and crs==row["crs"]
            with np.load(row["cache"]) as z:arrays={k:z[k].copy() for k in z.files}
            for band in row["bands"]:
                name=band["name"];arrays[name+"_valid"] &= qv&aerosol_valid(qa)
                band["valid_fraction"]=float(arrays[name+"_valid"].mean())
            if min(b["valid_fraction"] for b in row["bands"])<.5:
                excluded.append(dict(id=row["observation_id"],reason="same_single_source_minimum_50percent_after_aerosol_QA"));continue
            row["parent_cache"]=row["cache"]
            row["cache"]=str(cache/(row["observation_id"]+".npz"))
            row["aerosol_quality"]=dict(source=str(source),policy="exclude fill or high aerosol with invalid retrieval or interpolation")
            np.savez_compressed(row["cache"],**arrays)
            row["cache_sha256"]=hashlib.sha256(Path(row["cache"]).read_bytes()).hexdigest()
            Path(row["cache"]).with_suffix(".json").write_text(json.dumps(row,indent=2))
        accepted.append(row)
    by_site=collections.defaultdict(dict)
    for row in accepted:by_site[row["source_site"]][row["sensor"]]=row
    pairs=[]
    for site,sensors in by_site.items():
        if set(sensors)!={"S2","L8"}:continue
        a,b=sensors["S2"],sensors["L8"]
        assert a["split"]==b["split"] and a["group"]==b["group"]
        pairs.append(dict(site_id=site,split=a["split"],group=a["group"],s2_id=a["observation_id"],landsat_id=b["observation_id"],s2_scene=a["product_id"],landsat_scene=b["product_id"],**pair_quality(a,b)))
    for name,records in [("single_masked.jsonl",accepted),("cross30.jsonl",pairs)]:
        (folder/name).write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records))
    (folder/"exclusions.json").write_text(json.dumps(excluded,indent=2))
    summary=dict(status="MASK_REPAIR_BUILT_PENDING_ACCEPTANCE",download_status="USER_PAUSED",parent_release="release_local_v1",roles="inherited unchanged; no resplitting",single_source_min_valid_fraction=.5,
                 observations=dict(collections.Counter(r["split"]+"/"+r["sensor"] for r in accepted)),
                 strict_pairs=dict(collections.Counter(r["split"] for r in pairs if r["strict_eligible"])),
                 broad_pairs=dict(collections.Counter(r["split"] for r in pairs if r["broad_eligible"])),
                 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.jsonl")},
                 boundary="same calibrated reflectance and native grids; updated masks change evaluation population; refit TRAIN statistics and all controls; cross all17 support acceptance still required")
    (folder/"release.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=="__main__":main()

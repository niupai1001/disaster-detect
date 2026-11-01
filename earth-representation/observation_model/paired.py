"""Real S2/Landsat labels on the observed Landsat30m grid, local data only."""
import collections
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import ROOT,RELEASE,NativeSet
from .operators import resample_band

CROSS_RELEASE=Path(os.environ.get("OBSERVATION_CROSS_RELEASE",ROOT/"observation_model/outputs/phase0/cross_local_v1"))


def target_cube(data,sensor,ids,phase):
    values=[];valid=[]
    for band in data.metadata[sensor]:
        name=band["name"]
        value,coverage=resample_band(data.value[sensor][name][ids],data.valid[sensor][name][ids],band["native_gsd_m"],data.phase[sensor][name][ids],30,phase)
        values.append(value);valid.append(coverage>=.8)
    return torch.stack(values,1),torch.stack(valid,1)


class NativePairs:
    def __init__(self,data,records=None):
        self.data=data
        if records is None:records=[json.loads(s) for s in (CROSS_RELEASE/"pairs.jsonl").read_text().splitlines()]
        self.records=[r for r in records if r["split"]==data.split]
        maps={s:{r["observation_id"]:i for i,r in enumerate(rows)} for s,rows in data.records.items()}
        self.indices={s:torch.tensor([maps[s][r["s2_id" if s=="S2" else "landsat_id"]] for r in self.records],device=data.device) for s in ["S2","L8"]}
        self.phase=data.phase["L8"][data.metadata["L8"][0]["name"]][self.indices["L8"]]
        self.target={};self.valid={}
        for sensor in ["S2","L8"]:
            self.target[sensor],self.valid[sensor]=target_cube(data,sensor,self.indices[sensor],self.phase)
        joint=self.valid["S2"].all(1)&self.valid["L8"].all(1)
        self.joint=joint
        for sensor in self.valid:self.valid[sensor] &= joint[:,None]


def freeze_local_pairs():
    existing=CROSS_RELEASE/"release.json"
    if existing.exists():return json.loads(existing.read_text())
    config=json.loads((ROOT/"observation_model/config.json").read_text())
    assert not config["download_enabled"] and config["cross_execution"]["required_by_latest_user"]
    source=[json.loads(s) for s in (RELEASE/"cross30.jsonl").read_text().splitlines()]
    source=[r for r in source if r["strict_eligible"]]
    accepted=[];excluded=[]
    for split in ["train","val","test"]:
        data=NativeSet(split);pairs=NativePairs(data,source)
        for i,row in enumerate(pairs.records):
            quality=float(pairs.joint[i].float().mean())
            if quality<.8:
                excluded.append(dict(site_id=row["site_id"],reason="all17native_bands_joint_under80percent",fraction=quality));continue
            accepted.append(dict(**row,all17_joint_valid_fraction=quality,target_quantity="actual_native_L8_SR_and_area_aggregated_S2_SR",source_quantity="actual_native_S2_or_L8_SR",hls_used_as_independent_truth=False))
    counts=collections.Counter(r["split"] for r in accepted)
    if any(counts[s]<1 for s in ["train","val","test"]):raise ValueError("no reliable local pair in a required split; never manufacture target truth")
    CROSS_RELEASE.mkdir(parents=True,exist_ok=True)
    manifest=CROSS_RELEASE/"pairs.jsonl";manifest.write_text("".join(json.dumps(r)+"\n" for r in accepted))
    (CROSS_RELEASE/"exclusions.json").write_text(json.dumps(excluded,indent=2))
    report=dict(status="ACCEPTED_EXPLORATORY_REAL_CROSS30",latest_user_requires_execution=True,quality_thresholds_unchanged=True,original_multiregion_gate=False,pairs=dict(counts),regions={s:len({r["group"] for r in accepted if r["split"]==s}) for s in counts},independent_acquisition_pairs={s:len({(r["s2_scene"],r["landsat_scene"]) for r in accepted if r["split"]==s}) for s in counts},manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),masked_release=str(RELEASE),download_status="USER_PAUSED",boundary="small existing qualified set; exploratory bidirectional observation prediction, not broad generalization; no added downloads or weakened radiometry/QA/time/registration thresholds")
    existing.write_text(json.dumps(report,indent=2));return report


if __name__=="__main__":
    if torch.cuda.is_available():torch.cuda.init()
    print(json.dumps(freeze_local_pairs()))

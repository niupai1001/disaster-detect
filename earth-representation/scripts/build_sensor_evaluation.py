"""Freeze QA-selected local DEV observations for retrieval only."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scripts.build_sensor_release import admit

def main():
    out=Path("outputs/verification/sensor_evaluation");out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(l) for l in Path("outputs/verification/sensor_eval_observations.jsonl").read_text().splitlines() if l.strip()]
    accepted=[];excluded=Counter()
    for meta in rows:
        pair,reason=admit(meta["source_uri"],meta)
        if pair is None:excluded[reason]+=1
        else:accepted.append(pair)
    manifest=dict(status="accepted_geographic_DEV_only" if len(accepted)>=4 else "insufficient_strict_pairs",ids=[x[3]["sample_id"] for x in accepted],rows=[x[3] for x in accepted],excluded=dict(excluded),selection="QA only among existing local15 time steps; not labels or model scores; separate from pretraining and SSL checkpoint selection")
    if accepted:
        np.savez_compressed(out/"arrays.npz",s2=np.stack([x[0] for x in accepted]),landsat=np.stack([x[1] for x in accepted]),valid=np.stack([x[2] for x in accepted]))
        manifest["arrays_sha256"]=hashlib.sha256((out/"arrays.npz").read_bytes()).hexdigest()
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(n=len(accepted),excluded=dict(excluded))))
if __name__=="__main__":main()

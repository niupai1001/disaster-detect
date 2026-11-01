"""Record actual budgets and repair legacy boundary RNG in separate copies."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/verification"
TAGS=["verified_s2_"+x for x in ["standard","generic","physical"]]+["corrective_scale_"+x for x in ["uniform","random","selective"]]+["sensor_"+x for x in ["single","mixed","paired"]]


def records(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def attempted_exposure(train,n,batch_size=32):
    total=0;updates=0;per_epoch=(n+batch_size-1)//batch_size
    for sid in {r["session_id"] for r in train}:
        rows=[r for r in train if r["session_id"]==sid]
        first,last=rows[0],rows[-1]
        first_batch=(n%batch_size or batch_size) if first["global_step"]%per_epoch==0 else batch_size
        total+=last["metrics"]["exposure"]-first["metrics"]["exposure"]+first_batch
        updates+=last["global_step"]-first["global_step"]+1
    return int(total),updates


def main():
    rows=[]
    for tag in TAGS:
        meta=json.loads((OUT/"runs"/(tag+".provenance.json")).read_text())
        log=records(OUT/"tvlp"/tag/"metrics.jsonl")
        train=[r for r in log if r["phase"]=="train"]
        val=[r for r in log if r["phase"]=="val"]
        states=[]
        for suffix in ["", ".best_unlabeled"]:
            path=OUT/"runs"/(tag+suffix+".pt")
            state=torch.load(path,map_location="cpu",weights_only=False)
            row={k:state.get(k) for k in ["step","epoch","exposure","cursor","stale","best","stop_reason"]}
            row.update(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),lr=state["optimizer"]["param_groups"][0]["lr"])
            if state.get("sampling_resume_version",1)<2 and state["cursor"]==0:
                fixed=path.with_name(path.stem+".resume_fixed.pt")
                if not fixed.exists():
                    rng=np.random.default_rng();rng.bit_generator.state=state["epoch_rng"]
                    rng.permutation(len(meta["members"]))
                    state["epoch_rng"]=rng.bit_generator.state
                    state["sampling_resume_version"]=2
                    torch.save(state,fixed)
                copy=torch.load(fixed,map_location="cpu",weights_only=False)
                same=all(torch.equal(v,copy[k][n]) for k in ["student","teacher"] for n,v in state[k].items())
                if not same:raise ValueError("migration changed weights")
                row.update(resume_path=str(fixed.relative_to(ROOT)),model_teacher_identical=True)
                del copy
            else:row["resume_path"]=row["path"]
            states.append(row)
            del state
        # Sum session spans: excludes initialization; not GPU kernel time or idle interruption.
        sessions=sorted({r["session_id"] for r in train})
        seconds=sum((datetime.fromisoformat(t[-1]["timestamp"])-datetime.fromisoformat(t[0]["timestamp"])).total_seconds() for sid in sessions if len(t:=[r for r in train if r["session_id"]==sid])>1)
        common_step=1210 if tag.startswith("sensor_") else 3100
        common=next((r["metrics"]["loss"] for r in val if r["global_step"]==common_step),None)
        attempted,attempted_updates=attempted_exposure(train,len(meta["members"]))
        rows.append(dict(tag=tag,checkpoints=states,train_locations=len(meta["members"]),validation_locations=len(meta["validation_members"]),
            final_train_loss=train[-1]["metrics"]["loss"],absolute_min_recorded_val_loss=min(r["metrics"]["loss"] for r in val),
            common_step=common_step,common_step_fixed_view_dino_loss=common,active_logged_seconds=seconds,
            attempted_location_exposure=attempted,attempted_updates=attempted_updates,
            rolled_back_location_exposure=attempted-states[0]["exposure"],
            locations_per_second=attempted/seconds,peak_ssl_gpu_mib=None,
            distinct_sensor_observation_exposure=states[0]["exposure"]*(2 if tag in ["sensor_mixed","sensor_paired"] else 1),
            warnings=["best checkpoint selected by >=0.5% relative improvement, not absolute minimum", "SSL GPU peak was not recorded by frozen releases; no retrospective value invented", "common-step results are fixed-view unlabeled loss, not equal-budget downstream checkpoints"],
            actual_config=meta["arguments"],runtime_contract=meta.get("runtime_contract"),source_manifest=str((OUT/"runs"/(tag+".provenance.json")).relative_to(ROOT))))
    (OUT/"training_acceptance.json").write_text(json.dumps(dict(status="actual_completed_runs",runs=rows),indent=2))
    print(json.dumps([{k:r[k] for k in ["tag","common_step","common_step_fixed_view_dino_loss"]} for r in rows]))


if __name__=="__main__":main()

"""Freeze unlabeled selections and inference exports before any final TEST evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .data import ROOT,RELEASE

OUT=Path(os.environ.get("OBSERVATION_OUTPUT_ROOT",ROOT/"observation_model/outputs"))


def durable_save(value,path):
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def completed_runs():
    names=["masked_transformer","masked_cnn","masked_random","cross_masked_control","cross_cross30"]
    path=OUT/"after_cross_status.json"
    if not path.exists():return None
    status=json.loads(path.read_text())["status"]
    if status=="P4_training_complete_pending_final_evaluation":names += ["observation_none","observation_fixed"]
    elif status!="P4_SKIPPED_NEGATIVE_VALIDATION_GATE":return None
    runs={}
    for name in names:
        path=OUT/"runs"/name/"training_result.json"
        if not path.exists():return None
        result=json.loads(path.read_text())
        if result["status"]!="training_complete_pending_evaluation":return None
        runs[name]=result
    return runs


def freeze(runs):
    output=OUT/"final";output.mkdir(exist_ok=True)
    manifest=output/"selection.json"
    if manifest.exists():return json.loads(manifest.read_text())
    common={}
    for prefix in ["masked_","cross_","observation_"]:
        members=[r for name,r in runs.items() if name.startswith(prefix)]
        if members:common[prefix]=min(r["step"] for r in members)//1000*1000
    points=[];random_equivalence=[]
    reference_path=OUT/"reference_initial/random_encoder.pt"
    reference=torch.load(reference_path,map_location="cpu",weights_only=False) if reference_path.exists() else None
    for name,result in runs.items():
        folder=OUT/"runs"/name;prefix=next(k for k in common if name.startswith(k));variant="cnn" if name=="masked_cnn" else "random" if name=="masked_random" else "transformer"
        for role,filename in [("best","best_unlabeled.pt"),("last","last.pt"),("common",f"step{common[prefix]}.pt")]:
            source=folder/filename
            state=torch.load(source,map_location="cpu",weights_only=False)
            if name=="masked_random" and reference is not None:
                assert state["statistics"]==reference["statistics"],"random reference normalization changed"
                assert all(torch.equal(value,state["model"][key]) for key,value in reference["model"].items() if key.startswith(("tokenizer.","encoder."))),"precomputed random encoder differs from selected checkpoint"
                random_equivalence.append(dict(role=role,step=state["step"],reference=str(reference_path),tokenizer_and_encoder_bitwise_equal=True,statistics_equal=True))
            digest=hashlib.sha256()
            for key,value in state["model"].items():digest.update(key.encode());digest.update(value.contiguous().numpy().tobytes())
            weight_hash=digest.hexdigest()
            destination=output/"export"/name/(role+".pt");destination.parent.mkdir(parents=True,exist_ok=True)
            payload=dict(model=state["model"],statistics=state["statistics"],variant=variant,step=state["step"],
                         resolved_config=state.get("resolved_config") or json.loads((folder/"resolved_config.json").read_text()),
                         source=str(source),source_model_sha256=weight_hash,role=role,selection="absolute-minimum unlabeled VAL for best; final state for last; shared stage milestone for common")
            durable_save(payload,destination)
            points.append(dict(run=name,role=role,step=state["step"],variant=variant,checkpoint=str(destination),original_checkpoint=str(source),
                               training_configuration=str(folder/"resolved_config.json"),
                               model_sha256=weight_hash,export_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    result=dict(status="SELECTION_FROZEN_BEFORE_FINAL_TEST",runs=runs,points=points,common_stage_steps=common,precomputed_random_encoder_equivalence=random_equivalence,
                data_release=str(RELEASE),statistics_sha256=hashlib.sha256((RELEASE/"statistics.json").read_bytes()).hexdigest(),
                physics_gate=json.loads((OUT/"phase4/gate.json").read_text()),
                selection="pretraining choices use unlabeled validation only; auxiliary labels cannot select encoder checkpoints",
                comparison_boundary="independent schedules/early stopping; common steps and independent best reported separately; stages inherit fixed ancestors",
                remaining="final VAL/TEST comparisons, selected frozen-encoder probes, costs, report and shared-ledger update remain required")
    temporary=manifest.with_suffix(".tmp");temporary.write_text(json.dumps(result,indent=2));temporary.replace(manifest)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watch",action="store_true");args=parser.parse_args()
    while True:
        runs=completed_runs()
        if runs is not None:
            result=freeze(runs);print(json.dumps(dict(status=result["status"],points=len(result["points"]))),flush=True);return
        if not args.watch:print("Waiting for all mandatory runs and the conditional P4 decision; no TEST opened");return
        time.sleep(15)


if __name__=="__main__":main()

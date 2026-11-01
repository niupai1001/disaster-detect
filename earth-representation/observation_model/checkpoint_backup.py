"""Persist complete checkpoint bytes without touching active trainer code or GPU state."""
import argparse
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import time

import torch

from .data import ROOT

OUT=Path(os.environ.get("OBSERVATION_OUTPUT_ROOT",ROOT/"observation_model/outputs"))
RUNS=["masked_transformer","masked_cnn","masked_random","cross_masked_control","cross_cross30","observation_none","observation_fixed"]


def snapshot(name):
    source=OUT/"runs"/name/"last.pt"
    if not source.exists():return None
    folder=OUT/"recovery_guard"/name;folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/"snapshot.json";stamp=source.stat().st_mtime_ns
    if metadata.exists() and json.loads(metadata.read_text()).get("source_mtime_ns")==stamp:return None
    raw=source.read_bytes()
    state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=False)
    required=["model","optimizer","scheduler","sampler","torch_rng","cuda_rng","step","statistics"]
    if not all(k in state for k in required):raise ValueError("incomplete training state")
    temporary=folder/"complete.tmp"
    with temporary.open("wb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    destination=folder/"complete.pt"
    if destination.exists():destination.replace(folder/"previous.pt")
    temporary.replace(destination)
    result=dict(run=name,step=state["step"],bytes=len(raw),source=str(source),source_mtime_ns=stamp,
                backup=str(destination),created_at=datetime.now(timezone.utc).isoformat(),
                scope="original checkpoint bytes; full CPU load verified before fsync+atomic replacement; previous backup retained; no GPU initialization")
    temporary=metadata.with_suffix(".tmp");temporary.write_text(json.dumps(result,indent=2));temporary.replace(metadata)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watch",action="store_true");args=parser.parse_args()
    failures={}
    while True:
        for name in RUNS:
            source=OUT/"runs"/name/"last.pt"
            if not source.exists():continue
            stamp=source.stat().st_mtime_ns
            if failures.get(name)==stamp:continue
            try:
                result=snapshot(name)
                if result:print(json.dumps(result),flush=True)
            except Exception as exc:
                failures[name]=stamp
                error=dict(run=name,error=type(exc).__name__+": "+str(exc),action="previous valid backup retained; retry only when source changes",timestamp=datetime.now(timezone.utc).isoformat())
                folder=OUT/"recovery_guard";folder.mkdir(exist_ok=True)
                with (folder/"errors.jsonl").open("a") as stream:stream.write(json.dumps(error)+"\n")
                print(json.dumps(error),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=="__main__":main()

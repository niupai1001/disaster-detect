"""Corrected, budget-limited main runs; mandatory cross uses common repaired10k."""

from .execution_gate import require_training_authorization
if __name__ == "__main__": require_training_authorization()
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"observation_model/outputs/repair_v2"
SOURCE=OUT/"source/earth-representation"


def worker(gpu,name,initial=None):
    directory=OUT/"runs"/name;directory.mkdir(parents=True,exist_ok=True)
    result=directory/"training_result.json"
    if result.exists() and json.loads(result.read_text()).get("status")=="training_complete_pending_evaluation":return dict(name=name,reused=True,returncode=0)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1",PYTHONUTF8="1",OBSERVATION_PROJECT_ROOT=str(ROOT),
        OBSERVATION_RELEASE=str(ROOT/"observation_model/outputs/phase0/release_local_v2"),
        OBSERVATION_CROSS_RELEASE=str(ROOT/"observation_model/outputs/phase0/cross_local_v2"),OBSERVATION_CONFIG=str(OUT/"config.json"))
    if name.startswith("masked_"):
        command=[sys.executable,"-u","-X","utf8","-m","observation_model.train_masked","--variant",name.removeprefix("masked_"),"--coordinate-mode","explicit_global"]
    else:
        command=[sys.executable,"-u","-X","utf8","-m","observation_model.train_cross","--mode",name.removeprefix("cross_"),"--initial",str(initial)]
    command += ["--out",str(directory),"--tag","obs_v2_"+name,"--tv-root",str(OUT/"tvlp")]
    if (directory/"last.pt").exists():command += ["--resume",str(directory/"last.pt")]
    logs=OUT/"logs";logs.mkdir(exist_ok=True)
    with (logs/(name+".log")).open("a",encoding="utf-8") as stream:
        process=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT)
        status=dict(status="running",name=name,pid=process.pid,gpu=gpu,command=command,source=str(SOURCE))
        (directory/"process.json").write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        code=process.wait()
    status.update(status="terminal",returncode=code);(directory/"process.json").write_text(json.dumps(status,indent=2))
    if code:raise RuntimeError(f"{name} failed with {code}; inspect its preserved log")
    return status


def cnn_then_random():
    return [worker(1,"masked_cnn"),worker(1,"masked_random")]


def cross_when_ready():
    point=OUT/"runs/masked_transformer/step10000.pt"
    while not point.exists():
        process=OUT/"runs/masked_transformer/process.json"
        if process.exists():
            state=json.loads(process.read_text())
            if state.get("status")=="terminal":raise RuntimeError("repaired Transformer stopped before cross initialization")
        time.sleep(10)
    # Milestones are saved synchronously by the trainer. Validate after the write finishes.
    import torch
    while True:
        try:
            state=torch.load(point,map_location="cpu",weights_only=False)
            assert state["step"]==10000
            break
        except (EOFError,RuntimeError):time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,0,"cross_masked_control",point),pool.submit(worker,1,"cross_cross30",point)]
        return [job.result() for job in jobs]


def main():
    acceptance=json.loads((OUT/"acceptance/acceptance.json").read_text())
    if acceptance["status"]!="PASSED_REPAIRED_ENGINEERING":raise ValueError("repaired engineering not accepted")
    lock=OUT/"queue_lock.json"
    with lock.open("x") as stream:json.dump(dict(pid=os.getpid(),source=str(SOURCE)),stream)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(worker,0,"masked_transformer"),pool.submit(cnn_then_random),pool.submit(cross_when_ready)]
            results=[job.result() for job in jobs]
        (OUT/"queue_result.json").write_text(json.dumps(dict(status="P2_P3_TERMINAL_PENDING_EVALUATION",results=results),indent=2))
    finally:lock.unlink(missing_ok=True)


if __name__=="__main__":main()

"""Two GPUs, three independently scheduled single-seed masked prediction runs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"observation_model/outputs"
RELEASE=OUT/"releases/masked_v1/earth-representation"
PYTHON=ROOT.parent/"landslide-seg/.venv-use-independent/Scripts/python.exe"


def worker(gpu,variants):
    results=[];environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu),OBSERVATION_PROJECT_ROOT=str(ROOT),OBSERVATION_RELEASE=str(OUT/"phase0/release_local_v1"),PYTHONUTF8="1")
    for variant in variants:
        directory=OUT/"runs"/("masked_"+variant);tag="obs_masked_"+variant
        done=directory/"training_result.json"
        if done.exists() and json.loads(done.read_text()).get("status")=="training_complete_pending_evaluation":
            results.append(dict(variant=variant,returncode=0,reused_complete=True));continue
        command=[str(PYTHON),"-u","-X","utf8","-m","observation_model.train_masked","--variant",variant,"--tag",tag,"--out",str(directory),"--tv-root",str(OUT/"tvlp")]
        if (directory/"last.pt").exists():command += ["--resume",str(directory/"last.pt")]
        logs=OUT/"logs";logs.mkdir(exist_ok=True)
        with (logs/(tag+".console.log")).open("a",encoding="utf-8") as stream:
            process=subprocess.Popen(command,cwd=RELEASE,env=environment,stdout=stream,stderr=subprocess.STDOUT)
            status=dict(status="running",gpu=gpu,pid=process.pid,variant=variant,command=command,source_release=str(RELEASE),data_release=environment["OBSERVATION_RELEASE"],results=results)
            (OUT/f"masked_gpu{gpu}_status.json").write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
            code=process.wait()
        results.append(dict(variant=variant,returncode=code))
        # Independent variants still run if a sibling fails; root inspects actual failure.
    (OUT/f"masked_gpu{gpu}_status.json").write_text(json.dumps(dict(status="terminal",results=results),indent=2))
    return results


def main():
    if not (OUT/"phase1/acceptance.json").exists():raise ValueError("real100/resume acceptance missing")
    config=json.loads((ROOT/"observation_model/config.json").read_text())
    if config["download_enabled"]:raise ValueError("user download pause must remain in force")
    lock=OUT/"masked_queue_lock.json"
    with lock.open("x") as stream:json.dump(dict(pid=os.getpid(),entrypoint=__file__),stream)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            jobs=[executor.submit(worker,0,["transformer"]),executor.submit(worker,1,["cnn","random"])]
            results=[job.result() for job in jobs]
        (OUT/"masked_queue_status.json").write_text(json.dumps(dict(status="training_terminal_pending_research_evaluation",results=results),indent=2))
    finally:lock.unlink(missing_ok=True)


if __name__=="__main__":main()

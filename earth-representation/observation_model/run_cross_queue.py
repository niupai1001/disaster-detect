"""Prioritize required cross-source runs from the common10k masked milestone."""

from .execution_gate import require_training_authorization
if __name__ == "__main__": require_training_authorization()
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"observation_model/outputs"
PYTHON=ROOT.parent/"landslide-seg/.venv-use-independent/Scripts/python.exe"


def worker(gpu,mode,source,release):
    out=OUT/"runs"/("cross_"+mode);tag="obs_cross_"+mode
    completed=out/"training_result.json"
    if completed.exists() and json.loads(completed.read_text())["status"]=="training_complete_pending_evaluation":return dict(mode=mode,returncode=0,reused=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OBSERVATION_PROJECT_ROOT=str(ROOT),OBSERVATION_RELEASE=str(OUT/"phase0/release_local_v1"),PYTHONUTF8="1")
    command=[str(PYTHON),"-u","-X","utf8","-m","observation_model.train_cross","--mode",mode,"--initial",str(source),"--out",str(out),"--tag",tag,"--tv-root",str(OUT/"tvlp")]
    if (out/"last.pt").exists():command += ["--resume",str(out/"last.pt")]
    with (OUT/"logs"/(tag+".console.log")).open("a",encoding="utf-8") as stream:
        process=subprocess.Popen(command,cwd=release,env=env,stdout=stream,stderr=subprocess.STDOUT)
        status=dict(status="running",gpu=gpu,pid=process.pid,mode=mode,command=command,initial=str(source),source_release=str(release))
        (OUT/f"cross_gpu{gpu}_status.json").write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        code=process.wait()
    result=dict(mode=mode,returncode=code)
    (OUT/f"cross_gpu{gpu}_status.json").write_text(json.dumps(dict(status="terminal",**result),indent=2));return result


def main():
    config=json.loads((ROOT/"observation_model/config.json").read_text())
    if config["download_enabled"]:raise ValueError("downloads must remain paused")
    for name in ["cross4","cross_control4"]:
        r=json.loads((OUT/"phase1"/name/"training_result.json").read_text())
        if r["status"]!="engineering_complete":raise ValueError("cross engineering incomplete")
    lock=OUT/"cross_queue_lock.json"
    with lock.open("x") as stream:json.dump(dict(pid=os.getpid(),entrypoint=__file__),stream)
    try:
        milestone=OUT/"runs/masked_transformer/step10000.pt"
        state=dict(status="waiting_for_common_masked_step10000",pid=os.getpid(),milestone=str(milestone),reason="latest user prioritizes cross-source; other masked runs continue unchanged",download_enabled=False)
        (OUT/"cross_queue_status.json").write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
        previous_size=None
        while True:
            if milestone.exists():
                size=milestone.stat().st_size
                if size>0 and size==previous_size:break
                previous_size=size
            terminal=OUT/"masked_gpu0_status.json"
            if terminal.exists():
                r=json.loads(terminal.read_text())
                if r.get("status")=="terminal" and any(x.get("returncode") for x in r.get("results",[])):raise RuntimeError("masked source failed before cross initial milestone; root must inspect and recover")
            time.sleep(10)
        initial=OUT/"cross_initial";initial.mkdir(exist_ok=True);source=initial/"masked_step10000.pt"
        if not source.exists():shutil.copyfile(milestone,source)
        (initial/"provenance.json").write_text(json.dumps(dict(policy="fixed predeclared shared10000-update milestone, no test-based selection; masked independent runs continue",source=str(milestone),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),step=10000),indent=2))
        release=OUT/"releases/cross_v1/earth-representation";package=release/"observation_model"
        if not (release.parent/"source_manifest.json").exists():
            package.mkdir(parents=True,exist_ok=True)
            for p in (ROOT/"observation_model").glob("*.py"):shutil.copyfile(p,package/p.name)
            shutil.copyfile(ROOT/"observation_model/config.json",package/"config.json")
            writer=release/"src/ssl";writer.mkdir(parents=True,exist_ok=True)
            (writer.parent/"__init__.py").write_text("");(writer/"__init__.py").write_text("");shutil.copyfile(ROOT/"src/ssl/tvlp_writer.py",writer/"tvlp_writer.py")
            hashes={str(p.relative_to(release)):hashlib.sha256(p.read_bytes()).hexdigest() for p in release.rglob("*") if p.is_file()}
            (release.parent/"source_manifest.json").write_text(json.dumps(dict(files=hashes,original_project=str(ROOT),data_release=str(OUT/"phase0/cross_local_v1"),initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2))
        (OUT/"cross_queue_status.json").write_text(json.dumps(dict(status="cross_training_running",initial=str(source),source_release=str(release)),indent=2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            jobs=[executor.submit(worker,0,"masked_control",source,release),executor.submit(worker,1,"cross30",source,release)]
            results=[job.result() for job in jobs]
        (OUT/"cross_queue_status.json").write_text(json.dumps(dict(status="cross_training_terminal_pending_evaluation",results=results),indent=2))
    finally:lock.unlink(missing_ok=True)


if __name__=="__main__":main()

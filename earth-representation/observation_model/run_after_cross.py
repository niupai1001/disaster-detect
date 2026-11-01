"""After mandatory P3, evaluate real pairs and conditionally run exactly two P4 arms."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .compare import compare
from .data import ROOT,RELEASE

OUT=Path(os.environ.get("OBSERVATION_OUTPUT_ROOT",ROOT/"observation_model/outputs"))
PYTHON=ROOT.parent/"landslide-seg/.venv-use-independent/Scripts/python.exe"
EVALUATION_SOURCE=Path(os.environ.get("OBSERVATION_EVALUATION_SOURCE",OUT/"releases/probe_v1/earth-representation"))
STATUS=OUT/"after_cross_status.json"


def write_status(status,**details):
    value=dict(status=status,pid=os.getpid(),**details)
    temporary=STATUS.with_suffix(".tmp");temporary.write_text(json.dumps(value,indent=2));temporary.replace(STATUS)
    print(json.dumps(value),flush=True)


def environment(gpu):
    value=os.environ.copy();value.update(CUDA_VISIBLE_DEVICES=str(gpu),OBSERVATION_PROJECT_ROOT=str(ROOT),OBSERVATION_RELEASE=str(RELEASE),PYTHONUTF8="1")
    return value


def snapshot(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError("frozen P3 checkpoint differs; preserve both versions before continuing")
    else:
        temporary=destination.with_suffix(".tmp");shutil.copyfile(source,temporary);temporary.replace(destination)
    return digest


def evaluate(kind,checkpoint,name):
    output=OUT/"phase3/gate_evaluation"/(name+".json")
    if output.exists():return json.loads(output.read_text())
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[str(PYTHON),"-u","-X","utf8","-m","observation_model.evaluate","--mode","cross","--split","val","--kind",kind,"--out",str(output)]
    if checkpoint is not None:command += ["--checkpoint",str(checkpoint)]
    with (OUT/"logs"/("gate_"+name+".console.log")).open("a",encoding="utf-8") as stream:
        subprocess.run(command,cwd=EVALUATION_SOURCE,env=environment(0),stdout=stream,stderr=subprocess.STDOUT,check=True)
    return json.loads(output.read_text())


def physical_score(result):
    directions={}
    for band in result["bands"]:directions.setdefault(band["direction"],[]).append(band["rmse"])
    return sum(sum(values)/len(values) for values in directions.values())/len(directions)


def freeze_observation_source(initial_hash):
    technical=json.loads(Path(os.environ.get("OBSERVATION_P4_ACCEPTANCE",OUT/"phase1/observation_acceptance.json")).read_text())
    for name,digest in technical["source_sha256"].items():
        if hashlib.sha256((ROOT/"observation_model"/name).read_bytes()).hexdigest()!=digest:raise ValueError("P4 training source changed after its actual engineering acceptance: "+name)
    release=OUT/"releases/observation_v1/earth-representation";manifest=release.parent/"source_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["initial_sha256"]!=initial_hash:raise ValueError("P4 source already bound to another initial checkpoint")
        return release
    package=release/"observation_model";package.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/"observation_model").glob("*.py"):shutil.copyfile(path,package/path.name)
    shutil.copyfile(ROOT/"observation_model/config.json",package/"config.json")
    writer=release/"src/ssl";writer.mkdir(parents=True,exist_ok=True)
    (writer.parent/"__init__.py").write_text("");(writer/"__init__.py").write_text("");shutil.copyfile(ROOT/"src/ssl/tvlp_writer.py",writer/"tvlp_writer.py")
    hashes={str(p.relative_to(release)):hashlib.sha256(p.read_bytes()).hexdigest() for p in release.rglob("*") if p.is_file()}
    manifest.write_text(json.dumps(dict(files=hashes,initial_sha256=initial_hash,original_project=str(ROOT),data_release=str(RELEASE)),indent=2))
    return release


def train_arm(mode,gpu,initial,gate,release):
    output=OUT/"runs"/("observation_"+mode);tag="obs_observation_"+mode
    result=output/"training_result.json"
    if result.exists():
        previous=json.loads(result.read_text())
        if previous["status"]=="training_complete_pending_evaluation":return dict(mode=mode,returncode=0,reused=True)
    command=[str(PYTHON),"-u","-X","utf8","-m","observation_model.train_cross","--mode","cross30","--observation-mode",mode,
             "--initial",str(initial),"--gate",str(gate),"--out",str(output),"--tag",tag,"--tv-root",str(OUT/"tvlp")]
    if (output/"last.pt").exists():command += ["--resume",str(output/"last.pt")]
    with (OUT/"logs"/(tag+".console.log")).open("a",encoding="utf-8") as stream:
        process=subprocess.Popen(command,cwd=release,env=environment(gpu),stdout=stream,stderr=subprocess.STDOUT)
        (OUT/("observation_gpu"+str(gpu)+"_status.json")).write_text(json.dumps(dict(status="running",pid=process.pid,gpu=gpu,mode=mode,command=command,source_release=str(release)),indent=2))
        code=process.wait()
    value=dict(mode=mode,returncode=code)
    (OUT/("observation_gpu"+str(gpu)+"_status.json")).write_text(json.dumps(dict(status="terminal",**value),indent=2));return value


def main():
    lock=OUT/"after_cross_lock.json"
    with lock.open("x") as stream:json.dump(dict(pid=os.getpid(),entrypoint=__file__),stream)
    try:
        config=json.loads(Path(os.environ.get("OBSERVATION_CONFIG",ROOT/"observation_model/config.json")).read_text())
        if config["download_enabled"]:raise ValueError("downloads remain paused")
        write_status("waiting_for_required_P3_completion",source="cross_queue_status.json")
        while True:
            results=[OUT/"runs"/name/"training_result.json" for name in ["cross_masked_control","cross_cross30"]]
            if all(p.exists() and json.loads(p.read_text())["status"]=="training_complete_pending_evaluation" for p in results):break
            queue_path=OUT/"cross_queue_status.json"
            queue=json.loads(queue_path.read_text()) if queue_path.exists() else {}
            if queue.get("status")=="cross_training_terminal_pending_evaluation" and any(r["returncode"] for r in queue["results"]):
                raise RuntimeError("required cross training failed; inspect and restore it before gate evaluation")
            time.sleep(10)
        write_status("evaluating_P3_best_on_fixed_VAL_only")
        folder=OUT/"phase3/selection";folder.mkdir(exist_ok=True,parents=True)
        candidate=folder/"cross_best.pt";control=folder/"masked_control_best.pt"
        initial_hash=snapshot(OUT/"runs/cross_cross30/best_unlabeled.pt",candidate)
        snapshot(OUT/"runs/cross_masked_control/best_unlabeled.pt",control)
        trial=evaluate("transformer",candidate,"cross_best")
        reference=evaluate("transformer",control,"masked_control_best")
        baselines={kind:evaluate(kind,OUT/"baselines/cross_mlp/best.pt" if kind=="mlp" else None,"simple_"+kind) for kind in ["mean","interpolation","ridge","mlp"]}
        selected=min(baselines,key=lambda kind:physical_score(baselines[kind]))
        statistics=json.loads((RELEASE/"statistics.json").read_text())["bands"]
        contrasts=dict(versus_matched_control=compare(trial,reference,"cross30",statistics),versus_best_simple=compare(trial,baselines[selected],"cross30",statistics))
        passed=all(r["extension_gate_passed"] for r in contrasts.values())
        gate=dict(status="PASS_P3_BIDIRECTIONAL" if passed else "SKIP_P4_NO_VERIFIED_BIDIRECTIONAL_GAIN",initial=str(candidate),initial_sha256=initial_hash,
                  selection="P3 absolute-minimum direct cross30 unlabeled validation checkpoints; no TEST used",comparisons=contrasts,
                  simple_model_selection={kind:physical_score(value) for kind,value in baselines.items()},selected_simple=selected,
                  boundary="exploratory strict pairs; few independent acquisitions; paired bootstrap is not multi-seed evidence or broad geographic validation")
        gate_file=OUT/"phase4/gate.json";gate_file.parent.mkdir(exist_ok=True);gate_file.write_text(json.dumps(gate,indent=2))
        if not passed:
            write_status("P4_SKIPPED_NEGATIVE_VALIDATION_GATE",gate=str(gate_file),next="freeze/evaluate all completed pretrained encoders; preserve negative result")
            return
        technical=json.loads(Path(os.environ.get("OBSERVATION_P4_ACCEPTANCE",OUT/"phase1/observation_acceptance.json")).read_text())
        if technical["status"]!="passed":raise ValueError("observation training not technically accepted")
        release=freeze_observation_source(initial_hash)
        write_status("P4_training_running",gate=str(gate_file),initial=str(candidate),source_release=str(release))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            jobs=[executor.submit(train_arm,mode,gpu,candidate,gate_file,release) for gpu,mode in [(0,"none"),(1,"fixed")]]
            results=[job.result() for job in jobs]
        if any(r["returncode"] for r in results):raise RuntimeError("observation arm failed; preserve result and restore its exact checkpoint")
        write_status("P4_training_complete_pending_final_evaluation",results=results,gate=str(gate_file))
    except BaseException as exc:
        write_status("failed_requires_root_recovery",error=type(exc).__name__+": "+str(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=="__main__":main()

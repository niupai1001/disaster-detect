"""Complete predefined evaluation and frozen probes after unlabeled checkpoint selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch

from .data import ROOT,RELEASE
from .compare import compare

OUT=Path(os.environ.get("OBSERVATION_OUTPUT_ROOT",ROOT/"observation_model/outputs"))
PYTHON=ROOT.parent/"landslide-seg/.venv-use-independent/Scripts/python.exe"


def validation_gates(index,points):
    statistics=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    result={}
    for mode,mask_set in [("masked","singleton"),("cross","cross30")]:
        eligible={r["name"]:json.loads(Path(r["result"]).read_text()) for r in index if r["mode"]==mode and r["split"]=="val" and r["pair_set"]=="strict" and r["route"]=="direct"}
        def physical(value):
            groups={}
            for band in value["bands"]:
                if band["mask_set"]==mask_set:groups.setdefault(band["direction"],[]).append(band["rmse"])
            return sum(sum(v)/len(v) for v in groups.values())/len(groups)
        controls={kind:eligible["simple_"+kind] for kind in ["mean","interpolation","ridge","mlp"]}
        selected=min(controls,key=lambda k:physical(controls[k]))
        contrasts={p["run"]:compare(eligible[p["run"]+"_best"],controls[selected],mask_set,statistics) for p in points if p["role"]=="best"}
        result[mode]=dict(selected_simple=selected,simple_physical_scores={k:physical(v) for k,v in controls.items()},comparisons=contrasts,selection="VAL physical RMSE only; no TEST used")
    result["boundary"]="final validation evidence; mandatory cross is required by user even if masked recovery fails its extension gate; independent seed0 schedules are not equal-budget causal ablations"
    return result


def source_release():
    release=OUT/"releases/final_eval_v1/earth-representation";manifest=release.parent/"source_manifest.json"
    if manifest.exists():return release
    package=release/"observation_model";package.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/"observation_model").glob("*.py"):shutil.copyfile(path,package/path.name)
    shutil.copyfile(ROOT/"observation_model/config.json",package/"config.json")
    writer=release/"src/ssl";writer.mkdir(parents=True,exist_ok=True)
    (writer.parent/"__init__.py").write_text("");(writer/"__init__.py").write_text("");shutil.copyfile(ROOT/"src/ssl/tvlp_writer.py",writer/"tvlp_writer.py")
    manifest.write_text(json.dumps(dict(files={str(p.relative_to(release)):hashlib.sha256(p.read_bytes()).hexdigest() for p in release.rglob("*") if p.is_file()},
                                        original_project=str(ROOT),data_release=str(RELEASE),selection=str(OUT/"final/selection.json")),indent=2))
    return release


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watch",action="store_true");args=parser.parse_args()
    final=OUT/"final";selection=final/"selection.json"
    if not args.watch and not selection.exists():print("Final selection not frozen; no TEST opened");return
    lock=OUT/"final_evaluation_lock.json"
    with lock.open("x") as stream:json.dump(dict(pid=os.getpid()),stream)
    def status(stage,**details):
        (OUT/"final_evaluation_status.json").write_text(json.dumps(dict(status=stage,pid=os.getpid(),**details),indent=2))
        print(json.dumps(dict(status=stage,**details)),flush=True)
    try:
        status("waiting_for_frozen_unlabeled_selection")
        while not selection.exists():time.sleep(15)
        manifest=json.loads(selection.read_text())
        if manifest["status"]!="SELECTION_FROZEN_BEFORE_FINAL_TEST":raise ValueError("pretraining selection not accepted")
        source=source_release();environment=os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="1",OBSERVATION_PROJECT_ROOT=str(ROOT),OBSERVATION_RELEASE=str(RELEASE),PYTHONUTF8="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        def execute(module,arguments,name):
            with (OUT/"logs"/("final_"+name+".console.log")).open("a",encoding="utf-8") as stream:
                subprocess.run([str(PYTHON),"-u","-X","utf8","-m",module,*arguments],cwd=source,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        evaluated={};index=[];point_steps={p["checkpoint"]:p["step"] for p in manifest["points"]}
        def evaluate(kind,checkpoint,name,mode,split,pair_set="strict",route="direct",weight_hash=None):
            path=final/"evaluation"/split/(name+"_"+mode+"_"+pair_set+"_"+route+".json");path.parent.mkdir(parents=True,exist_ok=True)
            key=(weight_hash,kind,mode,split,pair_set,route) if weight_hash else None
            if not path.exists():
                if key is not None and key in evaluated:
                    result=json.loads(evaluated[key].read_text());result["equivalent_weights_reused_from"]=str(evaluated[key]);result["predictor"]["checkpoint"]=str(checkpoint)
                    result["predictor"]["step"]=point_steps[str(checkpoint)]
                    path.write_text(json.dumps(result,indent=2))
                else:
                    arguments=["--mode",mode,"--split",split,"--kind",kind,"--pair-set",pair_set,"--route",route,"--out",str(path)]
                    if checkpoint:arguments += ["--checkpoint",str(checkpoint)]
                    status("evaluating",name=name,mode=mode,split=split,pair_set=pair_set,route=route)
                    execute("observation_model.evaluate",arguments,name+"_"+mode+"_"+split+"_"+pair_set+"_"+route)
            if key is not None:evaluated[key]=path
            index.append(dict(name=name,kind=kind,mode=mode,split=split,pair_set=pair_set,route=route,result=str(path)))
        for point in manifest["points"]:
            for split in ["val","test"]:
                for mode in ["masked","cross"]:
                    evaluate(point["variant"],point["checkpoint"],point["run"]+"_"+point["role"],mode,split,weight_hash=point["model_sha256"])
                if point["role"]=="best":
                    for pairs in ["broad","all_local"]:evaluate(point["variant"],point["checkpoint"],point["run"]+"_best","cross",split,pairs,weight_hash=point["model_sha256"])
                if point["run"].startswith("observation_"):
                    evaluate(point["variant"],point["checkpoint"],point["run"]+"_"+point["role"],"cross",split,route="observed-fine30",weight_hash=point["model_sha256"])
        for mode in ["masked","cross"]:
            for kind in ["mean","interpolation","ridge","mlp"]:
                checkpoint=OUT/"baselines"/(mode+"_mlp")/"best.pt" if kind=="mlp" else None
                for split in ["val","test"]:evaluate(kind,checkpoint,"simple_"+kind,mode,split)
        (final/"evaluation_index.json").write_text(json.dumps(index,indent=2))
        (final/"validation_gates.json").write_text(json.dumps(validation_gates(index,manifest["points"]),indent=2))
        probes=[]
        for point in manifest["points"]:
            if point["role"]!="best":continue
            if point["run"]=="masked_random" and not os.environ.get("OBSERVATION_OUTPUT_ROOT"):
                from .model import make_model
                from .frozen_features import model_hash
                initial=make_model("random");weights=torch.load(point["checkpoint"],map_location="cpu",weights_only=False)
                assert weights["statistics"]==json.loads((RELEASE/"statistics.json").read_text())["bands"]
                assert all(torch.equal(value,weights["model"][key]) for key,value in initial.state_dict().items() if key.startswith(("encoder.","tokenizer.")))
                initial_hash=model_hash(initial)
                for dataset in ["worldcover","landslide"]:
                    reference=OUT/"frozen_features"/dataset/"random/metadata.json"
                    assert json.loads(reference.read_text())["encoder_state_sha256_before"]==initial_hash
                    for task in ["classification","segmentation"] if dataset=="worldcover" else ["segmentation"]:
                        path=OUT/"probes"/dataset/"random"/task/"result.json"
                        if not path.exists():raise ValueError("verified random reference probe missing")
                        probes.append(dict(run=point["run"],dataset=dataset,task=task,result=str(path),reused_reference=True,
                                           equivalence="tokenizer and encoder tensors exactly equal seed0 reference; trained reconstruction decoder not used",checkpoint=point["checkpoint"]))
                continue
            for dataset in ["worldcover","landslide"]:
                features=OUT/"frozen_features"/dataset/point["run"];head=OUT/"probes"/dataset/point["run"]
                status("extracting_frozen_features",run=point["run"],dataset=dataset)
                if not (features/"metadata.json").exists():
                    execute("observation_model.frozen_features",["--dataset",dataset,"--kind",point["variant"],"--checkpoint",point["checkpoint"],"--out",str(features)],point["run"]+"_"+dataset+"_features")
                tasks=["classification","segmentation"] if dataset=="worldcover" else ["segmentation"]
                for task in tasks:
                    if not (head/task/"result.json").exists():
                        status("training_frozen_readout",run=point["run"],dataset=dataset,task=task)
                        execute("observation_model.frozen_probe",["--features",str(features),"--out",str(head),"--task",task,"--max-updates","2000"],point["run"]+"_"+dataset+"_"+task)
                    probes.append(dict(run=point["run"],dataset=dataset,task=task,result=str(head/task/"result.json")))
        (final/"probe_index.json").write_text(json.dumps(probes,indent=2))
        status("evaluation_and_probes_complete_pending_report_and_final_audit",evaluation_count=len(index),probe_count=len(probes))
    except BaseException as exc:
        status("failed_requires_root_recovery",error=type(exc).__name__+": "+str(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=="__main__":main()

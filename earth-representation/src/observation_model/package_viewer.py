"""Render fixed real cases and validate completed logs, preserving damaged originals."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import ROOT

OUT=Path(os.environ.get("OBSERVATION_OUTPUT_ROOT",ROOT/"observation_model/outputs"))
VIEWER=Path("C:/Users/yzh/Downloads/training_viewer_v1.1_spec_and_examples/training_viewer_v1/.venv/Scripts/training-viewer.exe")


def render_cases(directory,destination):
    paths=sorted(directory.glob("*.npz"))
    if not paths:raise ValueError("fixed real case arrays missing: "+str(directory))
    fig,axes=plt.subplots(len(paths),5,figsize=(15,3.2*len(paths)),squeeze=False,constrained_layout=True)
    for i,path in enumerate(paths):
        with np.load(path) as z:
            meta=json.loads(str(z["metadata"].item()));visible=z["visible_input"];visible_valid=z["visible_valid"].astype(bool)
            target=z["target"];prediction=z["prediction"];valid=z["valid"].astype(bool)
        lo,hi=np.quantile(target[valid],[.01,.99]);hi=max(hi,lo+1e-6)
        error=np.abs(prediction.astype(np.float64)-target);maximum=max(float(np.quantile(error[valid],.99)),1e-6)
        axes[i,0].imshow(np.ma.masked_where(~visible_valid,visible),cmap="gray")
        axes[i,0].set_title("Visible "+meta["visible_input_band"])
        truth=axes[i,1].imshow(np.ma.masked_where(~valid,target),vmin=lo,vmax=hi,cmap="viridis");axes[i,1].set_title("Real target "+meta["band"])
        axes[i,2].imshow(np.ma.masked_where(~valid,prediction),vmin=lo,vmax=hi,cmap="viridis");axes[i,2].set_title("Model estimate")
        picture=axes[i,3].imshow(np.ma.masked_where(~valid,error),vmin=0,vmax=maximum,cmap="magma")
        axes[i,3].set_title(f"Absolute error; RMSE={np.sqrt(np.mean(error[valid]**2)):.4f}")
        axes[i,4].imshow(valid,cmap="gray",vmin=0,vmax=1);axes[i,4].set_title("Measured target QA")
        fig.colorbar(truth,ax=axes[i,1:3],shrink=.7,label="Reflectance; shared display scale")
        fig.colorbar(picture,ax=axes[i,3],shrink=.7,label="Absolute reflectance error")
        for ax in axes[i]:ax.tick_params(labelsize=7);ax.set_xlabel("Column on each raster's own pixel grid",fontsize=7)
    prefix="Engineering checkpoint only; not a trained-model result. " if "phase1" in directory.parts else "Selected checkpoint. "
    fig.suptitle(prefix+"Fixed first cases per direction/mask; not selected by model quality. Display limits do not alter metrics.")
    destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=140);plt.close(fig)
    return [str(p) for p in paths]


def readable_log(source):
    valid={};damaged=[]
    for filename in ["metrics.jsonl","events.jsonl"]:
        rows=[]
        for i,line in enumerate((source/filename).read_bytes().splitlines()):
            try:json.loads(line)
            except (json.JSONDecodeError,UnicodeDecodeError):
                damaged.append(dict(file=filename,line=i+1,bytes=len(line),sha256=hashlib.sha256(line).hexdigest()));continue
            rows.append(line)
        valid[filename]=rows
    if not damaged:return source,[]
    target=OUT/"tvlp_reviewed"/(source.name+"_reviewed_logs")
    if (target/"run.json").exists():return target,damaged
    target.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/"run.json").read_text());meta["run_id"]=target.name;meta["name"]+=" / recovered log copy"
    meta["config"]["log_recovery"]=dict(original_run=str(source),removed_unreadable_records=damaged,unknown_values="not imputed",original_files_preserved=True)
    (target/"run.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    for name,rows in valid.items():(target/name).write_bytes(b"\n".join(rows)+b"\n")
    if (source/"artifacts").exists():shutil.copytree(source/"artifacts",target/"artifacts",dirs_exist_ok=True)
    return target,damaged


def package():
    final=OUT/"final";index=json.loads((final/"evaluation_index.json").read_text());selection=json.loads((final/"selection.json").read_text());results=[]
    for run in selection["runs"]:
        source=OUT/"tvlp"/(os.environ.get("OBSERVATION_RUN_PREFIX","obs_")+run)
        if not source.exists():source=OUT/"tvlp"/("obs_"+run)
        target,damaged=readable_log(source)
        events=[json.loads(s) for s in (target/"events.jsonl").read_text().splitlines()]
        existing={r.get("path") for r in events if r["type"]=="artifact"};session=events[-1]["session_id"]
        for mode in ["masked","cross"]:
            row=next(r for r in index if r["name"]==run+"_best" and r["mode"]==mode and r["split"]=="test" and r["route"]=="direct" and r["pair_set"]=="strict")
            path=Path(row["result"]);directory=path.parent/(path.stem+"_cases");relative="artifacts/fixed_"+mode+"_test.png"
            if relative not in existing:
                arrays=render_cases(directory,target/relative)
                event=dict(type="artifact",timestamp=datetime.now(timezone.utc).isoformat(),session_id=session,path=relative,artifact_type="image",tag="fixed-real-"+mode,phase="test",role="comparison",
                           metadata=dict(generated_after_training=True,selection="first cases per direction/mask, never model-quality selection",source_arrays=arrays,kind="model estimate against real measured target"))
                with (target/"events.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(event)+"\n")
            published=final/"figures"/(run+"_fixed_"+mode+"_test.png");published.parent.mkdir(exist_ok=True)
            shutil.copyfile(target/relative,published)
        response=subprocess.run([str(VIEWER),"validate",str(target),"--json"],capture_output=True,text=True)
        validation=json.loads(response.stdout)
        results.append(dict(run=run,directory=str(target),damaged_original_records=damaged,validation=validation))
    passed=all(r["validation"]["error_count"]==0 and r["validation"]["warning_count"]==0 for r in results)
    report=dict(status="passed" if passed else "requires_review",runs=results,viewer_url="http://127.0.0.1:8766/",reviewed_root=str(OUT/"tvlp_reviewed"),
                boundary="original damaged logs retained; unreadable records excluded only in an explicitly named copy, never reconstructed")
    (final/"viewer_validation.json").write_text(json.dumps(report,indent=2))
    if not passed:raise ValueError("completed Viewer protocol validation has errors or warnings; inspect actual records")
    print(json.dumps(dict(status=report["status"],runs=len(results))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watch",action="store_true");args=parser.parse_args()
    while True:
        status=OUT/"final_evaluation_status.json"
        try:ready=status.exists() and json.loads(status.read_text())["status"]=="evaluation_and_probes_complete_pending_report_and_final_audit"
        except json.JSONDecodeError:ready=False
        if ready:package();return
        if not args.watch:print("Required final evaluations not complete; no final Viewer acceptance claimed");return
        time.sleep(30)


if __name__=="__main__":main()

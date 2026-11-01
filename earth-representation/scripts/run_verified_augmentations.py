"""Authorized single-seed independent runs; frozen source, no seed sweep."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/verification"
env=os.environ.copy()
env.update(EARTH_S2_ROOT=str(ROOT.parent/"landslide-seg/data/sen12"),EARTH_MANIFEST_ROOT=str(ROOT.parent/"landslide-seg/manifests"),EARTH_PROTOCOL_MANIFEST=str(ROOT.parent/"landslide-seg/manifests/v4r1/pool_100.jsonl"),CUDA_VISIBLE_DEVICES="0")
state={"status":"running","scope":"strength-matched ordinary blur/resize versus low-pass/area; seed0; semantic evaluation pending","runs":[]}
assert json.loads((OUT/"augmentation_calibration.json").read_text())["status"]=="accepted"
for mode in ["generic","physical"]:
    tag="verified_s2_"+mode
    command=[sys.executable,"-u","-X","utf8","-m","scripts.train_dino_official","--augmentation-calibration",str(OUT/"augmentation_calibration.json"),"--adaptive","--source","landslide","--n-patches","2223","--batch-size","32","--steps","50000","--max-epochs","400","--seed","0","--aug-mode",mode,"--tag",tag,"--out",str(OUT/"runs"),"--device","cuda","--tv-root",str(OUT/"tvlp")]
    checkpoint=OUT/"runs"/(tag+".pt")
    if checkpoint.exists():command += ["--resume",str(checkpoint)]
    state["current"]=tag;state["command"]=command
    (OUT/"augmentation_queue_status.json").write_text(json.dumps(state,indent=2))
    with (OUT/(tag+".console.log")).open("a",encoding="utf-8") as log:
        code=subprocess.run(command,cwd=OUT/"augmentation_release/earth-representation",env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    state["runs"].append({"tag":tag,"returncode":code})
    if code:
        state["status"]="failed";break
else:state["status"]="training_complete_pending_acceptance"
(OUT/"augmentation_queue_status.json").write_text(json.dumps(state,indent=2))

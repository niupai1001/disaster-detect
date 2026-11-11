"""Materialize the user-authorized additional ten epochs only after passed acceptance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parent, acceptance):
    parent,acceptance=Path(parent).resolve(),Path(acceptance).resolve()
    state=json.loads((parent/"formal_run/status.json").read_text())
    report=json.loads(acceptance.read_text(encoding="utf-8"))
    checkpoint=parent/"formal_run/last.pt"
    if (state["status"]!="COMPLETE" or report["status"]!="PASS" or not all(report["checks"].values())
            or not all(report["checks"].get(key) is True for key in ("A_improved","B1_improved","train_full_coverage",
                        "validation_full_coverage","checkpoint_binding","real_recovery_gate","tests_pass","optimizer_journal"))):
        raise ValueError("Completed parent and all accepted checks required")
    if report["run_id"]!=state["viewer"]["run_id"] or report["final_checkpoint_sha256"]!=sha(checkpoint):
        raise ValueError("Acceptance does not bind the committed parent")
    if report["source_manifest_sha256"]!=sha(parent/"source_manifest.json"):
        raise ValueError("Parent source manifest changed")
    document=copy.deepcopy(json.loads((parent/"config.json").read_text(encoding="utf-8")))
    previous_epochs=document["training"]["epochs"]
    document["training"]["epochs"]=10
    document["authorization"]=dict(source="user_in_current_chat",
        instruction="本轮后再续训10轮（累计11轮）；采用完整覆盖、有限值、恢复及A/B1完整VAL均改善的验收条件，通过后自动启动。",
        condition="All checks in the bound acceptance report passed",
        known_limits="Ten additional epochs are authorized; no automatic extension beyond this budget.")
    document["continuation"]=dict(parent_release=str(parent),checkpoint_sha256=sha(checkpoint),
        source_run_id=state["viewer"]["run_id"],acceptance_report=str(acceptance),acceptance_sha256=sha(acceptance),
        start_epoch=state.get("sampling_start_epoch",0)+previous_epochs)
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--acceptance",type=Path,required=True)
    parser.add_argument("--output-config",type=Path,required=True)
    args=parser.parse_args()
    if args.output_config.exists():
        parser.error("A new continuation configuration is required")
    document=prepare(args.parent,args.acceptance)
    args.output_config.parent.mkdir(parents=True,exist_ok=True)
    args.output_config.write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ADDITIONAL_TEN_EPOCHS_CONFIGURED",args.output_config)


if __name__=="__main__":
    main()

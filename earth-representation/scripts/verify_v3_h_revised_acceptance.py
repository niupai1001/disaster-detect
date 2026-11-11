"""Assess completed V3-H-R training; never launches training."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def assess(initial, final, updates, train_ids, val_ids, state, integrity):
    trained=[member["id"] for update in updates for member in update["members"]]
    before={member["id"]:member for member in initial["members"]}
    after={member["id"]:member for member in final["members"]}
    checks=dict(integrity)
    checks.update(
        completed=state.get("status")=="COMPLETE",
        train_full_coverage=Counter(trained)==Counter(train_ids),
        validation_full_coverage=(Counter(m["id"] for m in final["members"])==Counter(val_ids) and
                                  Counter(m["id"] for m in initial["members"])==Counter(val_ids)),
        contiguous_updates=[r["cursor"] for r in updates]==list(range(1,state["total_batches"]+1)),
        completed_budget=state["cursor"]==state["total_batches"]==state["spent_updates"],
        finite_training=all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) and
                            all(math.isfinite(v["loss"]) and all(math.isfinite(x) for x in v["channel_mse"].values())
                                for m in r["members"] for v in m["tasks"].values()) for r in updates))
    matched=before.keys()==after.keys()
    channel_changes=[]
    for identifier in before.keys() & after.keys():
        a,b=before[identifier],after[identifier]
        matched=matched and a["tasks"].keys()==b["tasks"].keys()
        for task in a["tasks"].keys() & b["tasks"].keys():
            x,y=a["tasks"][task],b["tasks"][task]
            matched=matched and x["targets"]==y["targets"] and x["channel_mse"].keys()==y["channel_mse"].keys()
            for channel in x["channel_mse"].keys() & y["channel_mse"].keys():
                channel_changes.append(dict(id=identifier,task=task,channel=channel,
                    initial=x["channel_mse"][channel],final=y["channel_mse"][channel],
                    difference=y["channel_mse"][channel]-x["channel_mse"][channel]))
    checks["same_evaluation_target_coverage"]=bool(matched)
    checks["finite_validation"]=all(math.isfinite(x[key]) for x in channel_changes for key in ("initial","final"))
    tasks={}
    for task in ("A","B1"):
        a=[r["tasks"][task]["loss"] for r in initial["members"] if task in r["tasks"]]
        b=[r["tasks"][task]["loss"] for r in final["members"] if task in r["tasks"]]
        first=sum(a)/len(a) if a else None
        last=sum(b)/len(b) if b else None
        valid=bool(a and b and len(a)==len(b) and math.isfinite(first) and math.isfinite(last))
        checks[task+"_improved"]=valid and last<first
        checks[task+"_summary_matches"]=valid and math.isclose(
            last,final["tasks"][task]["mse"],rel_tol=1e-9,abs_tol=1e-12) and math.isclose(
            first,initial["tasks"][task]["mse"],rel_tol=1e-9,abs_tol=1e-12)
        tasks[task]=dict(initial=first,final=last,eligible=len(b),
                        relative_change=None if not valid or first==0 else last/first-1)
    return dict(status="PASS" if all(checks.values()) else "FAIL",checks=checks,tasks=tasks,
                failed_checks=[k for k,v in checks.items() if not v],channel_changes=channel_changes)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path,required=True)
    parser.add_argument("--initial-evaluation",type=Path,required=True)
    parser.add_argument("--tests",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    release=args.release.resolve()
    run=release/"formal_run"
    state=json.loads((run/"status.json").read_text())
    if state["status"]!="COMPLETE":
        print("PENDING",state["status"],state["cursor"],flush=True)
        return
    if args.output.exists():
        parser.error("New assessment output required")
    manifest=json.loads((release/"source_manifest.json").read_text())
    source_intact=all(sha(release/"runtime"/name)==value for name,value in manifest.items())
    if not source_intact:
        raise ValueError("Frozen source changed")
    sys.path.insert(0,str(release/"runtime/src"))
    import torch
    from encoder_v3.shared_mae.data import load_data
    from encoder_v3.hierarchical_revised.train import binding_for
    config=json.loads((release/"config.json").read_text(encoding="utf-8"))
    datasets,resolver=load_data()
    binding=binding_for(config,datasets,resolver)
    final_payload=torch.load(run/"last.pt",map_location="cpu",weights_only=False)
    initial=json.loads(args.initial_evaluation.read_text())
    final=json.loads((run/"validation.json").read_text())
    updates=[json.loads(line) for line in (run/"updates.jsonl").read_text().splitlines()]
    journal=[json.loads(line) for line in (run/"optimizer_attempts.jsonl").read_text().splitlines()]
    test_root=ET.parse(args.tests).getroot()
    suites=[test_root] if test_root.tag=="testsuite" else list(test_root)
    tested=sum(int(s.attrib["tests"]) for s in suites)
    tests_pass=bool(tested>=106 and all(int(s.attrib[k])==0 for s in suites for k in ("errors","failures","skipped")))
    gate=json.loads((release/"preflight.json").read_text())
    weights=hashlib.sha256()
    for name,tensor in sorted(final_payload["model"].items()):
        weights.update(name.encode())
        weights.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    final_parameter_sha=weights.hexdigest()
    integrity=dict(source_intact=source_intact,tests_pass=tests_pass,
        weights_match_commit=(final_parameter_sha==state["final_parameter_sha"]==
                              final_payload["runner"]["final_parameter_sha"]),
        real_parameter_updates=final_parameter_sha!=state["initial_parameter_sha"],
        initial_parameters_match=initial["parameter_sha"]==state["initial_parameter_sha"],
        checkpoint_binding=final_payload["identity"]==binding==initial["identity"]==gate["identity"],
        real_recovery_gate=gate["status"]=="PASS",
        initial_checkpoint_unchanged=(initial["status"]=="COMPLETE" and initial["checkpoint_cursor"]==0 and
                                      initial["checkpoint_sha256"]==sha(run/"initial.pt")),
        checkpoint_committed=(final_payload["runner"]["status"]=="COMPLETE" and
                              final_payload["runner"]["cursor"]==state["cursor"]),
        optimizer_journal=all([r["sequence"] for r in journal if r["event"]==kind]==
                              list(range(1,state["spent_updates"]+1)) for kind in ("intent","applied")))
    result=assess(initial,final,updates,[r["id"] for r in datasets["TRAIN"].rows],
                  [r["id"] for r in datasets["VAL"].rows],state,integrity)
    result.update(run_id=state["viewer"]["run_id"],source_manifest_sha256=sha(release/"source_manifest.json"),
        final_checkpoint_sha256=sha(run/"last.pt"),initial_evaluation_sha256=sha(args.initial_evaluation),
        validation_sha256=sha(run/"validation.json"),tests_sha256=sha(args.tests),
        script_sha256=sha(__file__),criteria="full coverage, finite values, frozen identity, recovery and both full VAL task means improve")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(result["status"],result["tasks"],result["failed_checks"],flush=True)


if __name__=="__main__":
    main()

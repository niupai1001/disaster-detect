"""Compare committed A/B1 task views, masks and Viewer records to frozen artifacts."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys


def data(path):
    return json.loads(path.read_bytes())


def json_lines(path):
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def masks(path):
    for step,record in enumerate(json_lines(path/"updates.jsonl"),1):
        assert record["cursor"]==step
        members = {(m["id"],task):(v["target_hash"],v["visible_hash"],v["targets"])
                   for m in record["members"] for task,v in m["tasks"].items()}
        assert len(members)==2*record["batch_size"]
        yield record,members


def verify(campaign,prior):
    component = campaign/"s6_component_trials"
    base = component/"B0"
    with (prior/"p3_task_interference/T-JOINT/updates.jsonl").open(encoding="utf-8") as stream:
        targets = [{(m["id"],task):(v["target_mask_sha256"],v["targets"])
                    for m in u["members"] for task,v in m["tasks"].items()} for u in map(json.loads,stream)]
    actual = list(masks(base))
    for (_,rows),ref in zip(actual,targets):
        assert {k:(v[0],v[2]) for k,v in rows.items()}==ref
    assert len(actual)<=len(targets)==233
    check = dict(B0=dict(committed_updates=len(actual),target_masks_match_prior=True))
    for group in (component,campaign/"s7_seed_confirmation"):
        for run in sorted(group.glob("*/updates.jsonl")):
            path = run.parent
            if path==base:
                continue
            number = 0
            for record,current in masks(path):
                number = record["cursor"]
                assert actual[number-1][1]==current,(path.name,number)
            check[path.name] = dict(committed_updates=number,masks_and_counts_match_B0=True)
    viewer = campaign/"viewer_index.json"
    if viewer.is_file():
        index = data(viewer)
        for name,meta in index.items():
            run = Path(meta["directory"])
            assert data(run/"run.json")["run_id"]==meta["run_id"]
            events = list(json_lines(run/"events.jsonl"))
            metrics = list(json_lines(run/"metrics.jsonl"))
            train = [r["step"] for r in metrics if "train" in r["values"]]
            assert train==list(range(1,len(train)+1)),name
            check.setdefault(name,{}).update(viewer_metric_records=len(metrics),viewer_event=events[-1]["kind"])
    return check


if __name__=="__main__":
    campaign,prior = map(Path,sys.argv[1:3])
    result = verify(campaign,prior)
    text = json.dumps(result,indent=2,ensure_ascii=False)+"\n"
    (campaign/"comparison_audit.json").write_text(text,encoding="utf-8")
    print(text,flush=True)

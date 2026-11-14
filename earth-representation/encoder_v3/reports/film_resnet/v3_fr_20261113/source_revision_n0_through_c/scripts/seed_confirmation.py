"""Confirm the selected S6 configuration on two additional initialization seeds."""
import argparse
from pathlib import Path
import statistics
import subprocess
import sys

from run_v3_fr import read, write
from screen import METRICS, quality


def run(args):
    source = args.output/"s6_component_trials"
    decision = read(source/"screen.json")
    assert decision["status"] in ("NO_CANDIDATE","CANDIDATE")
    target = args.output/"s7_seed_confirmation"
    target.mkdir(exist_ok=True)
    if decision["status"]=="NO_CANDIDATE":
        write(target/"screen.json",dict(status="NOT_TRIGGERED",reason="No S6 candidate",optimizer_updates=0))
        return
    candidate,base = decision["candidate"],decision["baseline"]
    protocol = dict(candidate=candidate,baseline=base,seeds=[17,29,43],
                    mask_seed=17,data_order_seed=17,updates_per_run=233)
    write(target/"selection.json",protocol)
    for seed in (29,43):
        subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("run_v3_fr.py")),"compare",
                        "--release",str(args.release),"--prior",str(args.prior),"--output",str(args.output),
                        "--seed",str(seed),"--arms",base,candidate],check=True)
    scores = {}
    for arm in (base,candidate):
        scores[arm] = [read(source/arm/"val_233/summary.json")["groups"]] + [
            read(target/f"{arm}_seed{seed}/val_233/summary.json")["groups"] for seed in (29,43)]
    groups = scores[base][0]
    pooled = {arm:{g:{m:statistics.mean(row[g][m] for row in scores[arm]) for m in METRICS}
                   for g in groups} for arm in (base,candidate)}
    changes = {g:{m:[a[g][m]-b[g][m] for a,b in zip(scores[candidate],scores[base],strict=True)]
                  for m in METRICS} for g in groups}
    spread = {g:{m:statistics.stdev(values) for m,values in row.items()} for g,row in changes.items()}
    limits = read(args.output/"acceptance_policy.json")["S6_E"]
    result = quality(pooled[base],pooled[candidate],limits)
    field = "main_texture_gradient_mse"
    best_task = min(("A/all","B1/all"),key=lambda k:result["relative"][k][field]
                    if result["relative"][k][field] is not None else float("inf"))
    matching = sum(v<0 for v in changes[best_task][field])
    # Record each seed's screen so a severe regression can be reviewed explicitly.
    per_seed = [quality(a,b,limits) for a,b in zip(scores[base],scores[candidate],strict=True)]
    write(target/"screen.json",dict(status="REVIEW_REQUIRED" if result["status"]=="PASS_NUMERIC" and matching>=2
          else "NO_CANDIDATE",protocol=protocol,average_screen=result,primary_task=best_task,
          direction_agrees_in_seeds=matching,per_seed=per_seed,seed_deltas=changes,seed_standard_deviations=spread))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release","prior","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())

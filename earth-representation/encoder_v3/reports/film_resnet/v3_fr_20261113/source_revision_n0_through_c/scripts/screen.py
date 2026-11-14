"""Pair complete short runs and apply the preregistered quality screen."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import sys

import numpy as np

from run_v3_fr import read, table, write


ARMS = {"B0":None,"BP":"B0","N0":"BP","N1":"N0","N2":"N1","C":"N2"}
METRICS = ("mse","texture_gradient_mse","main_path_mse","main_texture_gradient_mse",
           "detail_contribution_energy")


def clusters(rows):
    parent = {}
    def root(key):
        while key in parent:
            key = parent[key]
        return key
    for row in rows:
        tags = [(name,row[name]) for name in ("source_group","acquisition_key","source_product")]
        for key in tags[1:]:
            a,b = root(key),root(tags[0])
            if a!=b:
                parent[a] = b
    return {r["observation_id"]:root(("source_group",r["source_group"])) for r in rows}


def paired(base, candidate, base_name, name, grouped):
    fields = ("observation_id","task","sensor","target_count","target_mask_hash","visible_mask_hash")
    data, summary = [],defaultdict(list)
    for a,b in zip(base,candidate,strict=True):
        assert tuple(a[k] for k in fields)==tuple(b[k] for k in fields),a["observation_id"]
        group = grouped[a["observation_id"]]
        differences = {f:float(b[f])-float(a[f]) for f in METRICS}
        row = dict(baseline=base_name,candidate=name,observation=a["observation_id"],task=a["task"],
                   sensor=a["sensor"],cluster=":".join(group)) | {f+"_delta":v for f,v in differences.items()}
        data.append(row)
        for sensor in ("all",a["sensor"]):
            summary[(a["task"],sensor)].append((group,differences))
    report = {}
    for (task,sensor),items in summary.items():
        groups = defaultdict(list)
        for key,delta in items:
            groups[key].append([delta[k] for k in METRICS])
        sums = np.array([np.sum(v,axis=0) for v in groups.values()])
        counts = np.array([len(v) for v in groups.values()])
        ids = np.random.default_rng(17).integers(len(groups),size=(5000,len(groups)))
        boot = sums[ids].sum(1)/counts[ids].sum(1)[:,None]
        intervals = np.quantile(boot,[.025,.975],axis=0).T
        report[f"{task}/{sensor}"] = dict(observations=len(items),clusters=len(groups),metrics={
            field:dict(mean=float(np.mean(vec)),median=float(np.median(vec)),p90=float(np.quantile(vec,.9)),
                       fraction_improved=float(np.mean(vec<0)),bootstrap_95=interval.tolist())
            for field,interval in zip(METRICS,intervals,strict=True)
            for vec in [np.array([r[field] for _,r in items])]})
    return data,report


def quality(base, current, criteria):
    relative = {}
    for group,values in base.items():
        relative[group] = {field:((current[group][field]-ref)/ref
                           if ref>=criteria["relative_denominator_min"] else None)
                           for field in METRICS for ref in [values[field]]}
    failures = [f"{group}: MSE" for group,r in relative.items()
                if r["mse"] is None or r["mse"]>criteria["mse_relative_max_degradation"]]
    a,b = [relative[key]["main_texture_gradient_mse"] for key in ("A/all","B1/all")]
    if (a is None or b is None or min(a,b)>-criteria["texture_relative_improvement"]
        or max(a,b)>criteria["other_texture_relative_max_degradation"]):
        failures.append("Shared-path texture thresholds")
    return dict(status="PASS_NUMERIC" if not failures else "FAIL", failures=failures,relative=relative)


def load_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def paired_bands(base,candidate,base_name,name):
    fields = ("observation_id","task","sensor","band","native_gsd_x","native_gsd_y",
              "target_count","target_mask_hash","visible_mask_hash")
    groups = defaultdict(list)
    for left,right in zip(base,candidate,strict=True):
        assert tuple(left[k] for k in fields)==tuple(right[k] for k in fields)
        record = {k:left[k] for k in fields}
        record.update(baseline=base_name,candidate=name,
                      mse_delta=float(right["mse"])-float(left["mse"]),
                      main_mse_delta=float(right["main_path_mse"])-float(left["main_path_mse"]),
                      signed_gradient_delta=(float(right["signed_gradient_error"])-float(left["signed_gradient_error"])
                                             if left["signed_gradient_error"] else None),
                      main_signed_gradient_delta=(float(right["main_signed_gradient_error"])
                                                  -float(left["main_signed_gradient_error"])
                                                  if left["main_signed_gradient_error"] else None))
        groups[(record["task"],record["sensor"],record["native_gsd_x"],record["native_gsd_y"],
                int(left["same_grid_visible_bands"]))].append(record)
    return [r for v in groups.values() for r in v], [
        dict(baseline=base_name,candidate=name,task=task,sensor=sensor,gsd_x=x,gsd_y=y,
             same_grid_visible_bands=n,bands=len(v),
             mean_mse_delta=float(np.mean([r["mse_delta"] for r in v])),
             mean_main_mse_delta=float(np.mean([r["main_mse_delta"] for r in v])))
        for (task,sensor,x,y,n),v in groups.items()]


def screen(out):
    root = out/"s6_component_trials"
    obs,bands,endings,states = {},{},{},{}
    for arm in ARMS:
        arm_dir = root/arm
        state = read(arm_dir/"status.json")
        assert state["status"]=="COMPLETE" and state["spent_updates"]==233
        states[arm] = state
        endings[arm] = state["validation"][-1]["groups"]
        obs[arm] = load_csv(arm_dir/"val_233/metrics_by_observation.csv")
        bands[arm] = load_csv(arm_dir/"val_233/metrics_by_band.csv")
    grouped = clusters(obs["B0"])
    criteria = read(out/"acceptance_policy.json")["S6_E"]
    effects,rows,band_rows,strata,candidates = {},[],[],[],[]
    for arm,parent in ARMS.items():
        if parent is None:
            continue
        results = {}
        for reference in dict.fromkeys((parent,"B0","BP")):
            if arm==reference:
                continue
            data,summary = paired(obs[reference],obs[arm],reference,arm,grouped)
            rows.extend(data)
            data,grouped_bands = paired_bands(bands[reference],bands[arm],reference,arm)
            band_rows.extend(data)
            strata.extend(grouped_bands)
            results[reference] = dict(paired=summary,screen=quality(endings[reference],endings[arm],criteria))
        effects[arm] = results
        replacement = "B0" if arm=="BP" else "BP"
        if all(results[key]["screen"]["status"]=="PASS_NUMERIC" for key in {parent,replacement}):
            score = min(results[replacement]["screen"]["relative"][key]["main_texture_gradient_mse"]
                        for key in ("A/all","B1/all"))
            candidates.append((score,arm,replacement))
    table(root/"paired_effects.csv",rows)
    table(root/"paired_effects_by_band.csv",band_rows)
    table(root/"paired_effects_by_gsd_and_support.csv",strata)
    costs = []
    for name,s in states.items():
        costs.append(dict(arm=name,seed=s["seed"],updates=s["spent_updates"],wall_seconds=s["wall_seconds"],
                          update_seconds=s["update_seconds"],save_seconds=s["save_seconds"],
                          validation_seconds=sum(v["seconds"] for v in s["validation"]),
                          A_views=s["actual_task_views"]["A"],B1_views=s["actual_task_views"]["B1"],
                          A_targets=s["target_pixel_counts"]["A"],B1_targets=s["target_pixel_counts"]["B1"],
                          peak_primary_bytes=max(s["peak_allocated_bytes"].values()),parameters=s["parameters"],
                          trainable_parameters=s["trainable_parameters"]))
    table(root/"cost_summary.csv",costs)
    chosen = [dict(candidate=arm,baseline=reference,ranking_value=score) for score,arm,reference in sorted(candidates)]
    write(root/"screen.json",dict(status="IMAGE_REVIEW_REQUIRED" if candidates else "NO_CANDIDATE",
                                 effects=effects,numeric_candidates=chosen))
    print("S6 numerical screen",chosen or "NO_CANDIDATE",flush=True)


if __name__ == "__main__":
    screen(Path(sys.argv[1]))

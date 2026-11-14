"""Write a compact endpoint and budget table from the completed run records."""
import json
from pathlib import Path
import sys

from run_v3_fr import table


out = Path(sys.argv[1])
stage = out/"s6_component_trials"
rows = []
for name in ("B0","BP","N0","N1","N2","C"):
    arm = stage/name
    if not (arm/"status.json").exists():
        continue
    state = json.loads((arm/"status.json").read_bytes())
    if state["status"]!="COMPLETE":
        continue
    groups = state["validation"][-1]["groups"]
    a,b = groups["A/all"],groups["B1/all"]
    rows.append(dict(arm=name,seed=state["seed"],updates=state["spent_updates"],
                     A_mse=a["mse"],B1_mse=b["mse"],A_texture=a["texture_gradient_mse"],
                     B1_texture=b["texture_gradient_mse"],B1_main_mse=b["main_path_mse"],
                     B1_main_texture=b["main_texture_gradient_mse"],
                     A_S2_mse=groups["A/S2"]["mse"],A_L8_mse=groups["A/L8"]["mse"],
                     B1_S2_mse=groups["B1/S2"]["mse"],B1_L8_mse=groups["B1/L8"]["mse"],
                     wall_seconds=state["wall_seconds"],update_seconds=state["update_seconds"],
                     validation_seconds=sum(x["seconds"] for x in state["validation"]),
                     peak_bytes=max(state["peak_allocated_bytes"].values()),
                     parameters=state["parameters"],trainable_parameters=state["trainable_parameters"],
                     A_views=state["actual_task_views"]["A"],B1_views=state["actual_task_views"]["B1"],
                     A_targets=state["target_pixel_counts"]["A"],B1_targets=state["target_pixel_counts"]["B1"]))
if rows:
    table(out/"summary/endpoint_metrics_and_cost.csv",rows)

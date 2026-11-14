"""Verify actual per-member exposure and optimizer response in the S5 journals."""
import json
import math
from pathlib import Path
import sys


campaign = Path(sys.argv[1])
root = campaign/"s5_capacity_fit"
report = {}
for arm in ("AE-L","AE-F","OR-D0","OR-D1"):
    path = root/arm
    identity = json.loads((path/"identity.json").read_bytes())
    with (path/"updates.jsonl").open(encoding="utf-8") as stream:
        entries = list(map(json.loads,stream))
    assert len(entries)==400
    for step,entry in enumerate(entries,1):
        assert entry["step"]==step and list(entry["losses"])==identity["fit_ids"]
        assert all(math.isfinite(x) and x>0 for x in
                   (entry["gradient_norm"],entry["parameter_change"],entry["lr"]))
        if arm.startswith("OR-"):
            assert len(entry["latent_gradient_norm"])==6
            assert all(math.isfinite(x) and x>0 for x in entry["latent_gradient_norm"].values())
    nodes = []
    for n in (1,25,100,200,300,399):
        before,after = entries[n-1:n+1]
        nodes.append(dict(update=n,mse_before=before["losses"],mse_after=after["losses"]))
    report[arm] = dict(status="PASS",updates=400,fit_ids=identity["fit_ids"],
                      fit_views_per_member=400,before_after_nodes=nodes)
(root/"optimizer_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")

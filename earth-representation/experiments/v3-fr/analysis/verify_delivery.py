"""Check final report links, every completed endpoint identity and published byte hashes."""
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re
import sys


out = Path(sys.argv[1])
project = Path(__file__).resolve().parents[1]
summary = out / "summary"


def read(path):
    return json.loads(path.read_bytes())


def hash_file(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


screen = read(out/"s6_component_trials/screen.json")
assert screen["status"] == "NO_CANDIDATE"
assert read(out/"s7_seed_confirmation/screen.json")["status"] == "NOT_TRIGGERED"
arms = {"B0","BP","N0","N1","N2","C"}
for name,num in (("metrics_by_observation.csv",766),("metrics_by_band.csv",3133)):
    with (summary/name).open(encoding="utf-8-sig") as stream:
        counts = Counter((r["experiment_id"],int(r["update"])) for r in csv.DictReader(stream))
    assert counts == {(arm,step):num for arm in arms for step in (0,42,116,233)},name

for name in ("paired_effects.csv","paired_effects_by_band.csv"):
    count = 766 if name=="paired_effects.csv" else 3133
    with (out/"s6_component_trials"/name).open(encoding="utf-8-sig") as stream:
        rows = Counter((r["candidate"],r["baseline"]) for r in csv.DictReader(stream))
    assert len(rows)==12 and set(rows.values())=={count},name
for path in (out/"README.md",*(summary/n for n in ("REPORT.md","DECISION.md","reproduce.md"))):
    for name in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
        assert (path.parent/name).exists(),(path.name,name)
audit = read(out/"comparison_audit.json")
assert len(audit)==10 and all(r["viewer_values_match_source"] and r["viewer_event"]=="completed" for r in audit.values())
manifest = read(summary/"artifact_manifest.json")
for item in manifest:
    path = (out if item["scope"]=="campaign" else project)/item["path"]
    assert path.stat().st_size==item["bytes"] and hash_file(path)==item["sha256"],item["path"]
for arm in arms:
    path = out/"s6_component_trials"/arm
    status = read(path/"status.json")
    assert status["spent_updates"] == 233 and status["actual_task_views"]==dict(A=7442,B1=7442)
    assert hash_file(path/"update_233.pt") == status["checkpoint_hash"]
print(f"Validated {len(manifest)} manifest entries, all 24 evaluation nodes, 12 paired comparisons and 10 Viewer runs")

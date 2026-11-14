"""Assemble committed campaign evidence and preserve source/output provenance."""
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


campaign = Path(sys.argv[1])
project = Path(__file__).resolve().parents[1]
summary = campaign/"summary"
summary.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_bytes()) if path.is_file() else None


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def gather_csv(name,nodes):
    dest = summary/name
    rows = 0
    with dest.open("w",encoding="utf-8-sig",newline="") as out:
        writer = None
        for node in nodes:
            path = node/name
            if not path.is_file():
                continue
            with path.open(encoding="utf-8-sig",newline="") as stream:
                reader = csv.DictReader(stream)
                if writer is None:
                    writer = csv.DictWriter(out,reader.fieldnames)
                    writer.writeheader()
                assert writer.fieldnames==reader.fieldnames
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    return rows


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


stages = {}
for stage,files in {
    "S0":["baseline_identity.json","geometry_census_summary.json"],
    "S1":["s1_baseline/baseline_replay.json","s1_baseline/executor_reference.json"],
    "S4":["s4_decoder_execution/decoder_terminal_gate.json","s4_decoder_execution/legacy_detail_gate.json",
          "s4_decoder_execution/capacity.json","s4_decoder_execution/one_step_equivalence.json","s4_decoder_execution/resume_gate.json",
          "s4_decoder_execution/d1_synthetic_patterns.json"],
    "S5":["s5_capacity_fit/fit_gate.json"],
    "S6":["s6_component_trials/screen.json"],
    "S7":["s7_seed_confirmation/screen.json"],
}.items():
    stages[stage] = {f:(report or {}).get("status","PENDING") for f in files for report in [read(campaign/f)]}
for name in ("candidate_unit_tests_final.xml","historical_regression.xml"):
    path = campaign/name
    if path.exists():
        suite = ET.parse(path).find("testsuite")
        stages[name] = {k:int(suite.attrib[k]) for k in ("tests","errors","failures","skipped")}
stages["excluded_attempts"] = {"S6_initial_evaluation":dict(optimizer_updates=0,
    reason="New evaluation initially used float64 quantile selection; corrected to frozen Torch float32."),
    "S6_pre_revision_N0":dict(optimizer_updates=0,reason="Reserved output blocked the superseded initialization before launch.")}
write(summary/"test_report.json",stages)
nodes = sorted(p.parent for stage in ("s6_component_trials","s7_seed_confirmation")
               for p in (campaign/stage).glob("*/val_*/summary.json"))
if nodes:
    for name in ("metrics_by_observation.csv","metrics_by_band.csv"):
        print(name,gather_csv(name,nodes))
manifest = []
for base,kind in ((campaign,"campaign"),(project,"source")):
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.name in ("artifact_manifest.json","viewer_index.tmp") or "__pycache__" in p.parts:
            continue
        path = p.relative_to(base)
        manifest.append(dict(scope=kind,path=path.as_posix(),sha256=sha(p),bytes=p.stat().st_size,
                    phase=path.parts[0],purpose={"pt":"checkpoint","npz":"floating prediction / geometry",
                    "py":"source","csv":"metrics or trace","png":"figure","md":"document"}.get(p.suffix[1:],"evidence"),
                    retention="local"))
write(summary/"artifact_manifest.json",manifest)
print("Manifest",len(manifest))

"""Read-only rehash of completed runs; does not train or overwrite old evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'outputs/use_v2'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

prior = read(base/'audit/final_acceptance_v1.json')
rows=[]
errors=[]
checked=0
for run in prior['runs']:
    folder=base/'runs'/run['job']
    s=read(folder/'summary.json')
    local=[]
    if not s['complete'] or s['formal_updates'] != run['formal_updates']:
        local.append('completion/budget mismatch')
    if abs(s['best_score']-run['best'])>0.000051:
        local.append('published rounded score mismatch')
    for name,digest in s['checkpoints'].items():
        p=folder/name
        if not p.exists() or sha(p)!=digest: local.append('checkpoint mismatch '+name)
        checked+=1
    curve=[json.loads(x) for x in (folder/'curve.jsonl').read_text().splitlines() if x.strip()]
    frozen={x['frozen_sha256'] for x in curve if 'frozen_sha256' in x}
    if len(frozen)>1: local.append('frozen hash changed in curve')
    rows.append(dict(job=run['job'],updates=s['formal_updates'],score=s['best_score'],trainable_parameters=s['trainable_parameters'],curve_records=len(curve),frozen_hashes=sorted(frozen),summary_sha256=sha(folder/'summary.json'),problems=local))
    errors.extend([run['job']+': '+x for x in local])
    print(run['job'], 'checked',flush=True)
result=dict(artifact_checks_passed=not errors,run_count=len(rows),checkpoint_files_rehashed=checked,formal_updates=sum(x['updates'] for x in rows),problems=errors,runs=rows,limits=['No new full-dataset inference or training','Recorded frozen hashes checked; optimizer membership not independently replayed','Research success is not established by artifact integrity'])
dest=base/'closeout_review';dest.mkdir(exist_ok=True)
(dest/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False))

"""Aggregate existing metric-v2 counts on fixed Italy cohorts; no inference/training."""
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ('s2_10band_v2_clean', 's2_10band_v3_p20')
OUT = ROOT / 'outputs/segformer/review_next_stage'

def records(version):
    return [json.loads(line) for line in (ROOT/'manifests'/f'{version}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

def aggregate(rows):
    fields=('tp','fp','fn','tn','n_gt_objects_evaluable','n_matched_evaluable','tiny_n_evaluable','tiny_n_matched_evaluable')
    s={k:sum(int(r[k]) for r in rows) for k in fields}
    def div(a,b): return a/b if b else None
    return dict(n=len(rows),**s,iou=div(s['tp'],s['tp']+s['fp']+s['fn']),dice=div(2*s['tp'],2*s['tp']+s['fp']+s['fn']),precision=div(s['tp'],s['tp']+s['fp']),recall=div(s['tp'],s['tp']+s['fn']),object_recall=div(s['n_matched_evaluable'],s['n_gt_objects_evaluable']),tiny_recall=div(s['tiny_n_matched_evaluable'],s['tiny_n_evaluable']))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old={r['sample_id'] for r in records(VERSIONS[0]) if r['split']=='test'}
    full={r['sample_id'] for r in records(VERSIONS[1]) if r['split']=='test'}
    assert old<=full and len(old)==382 and len(full)==2795
    runs=[]
    for ver in VERSIONS:
        for exp in ('A0','A1','B0','B1'):
            for seed in (1234,2026,3407):
                path=ROOT/'outputs/segformer'/ver/exp/str(seed)/'eval_v2/per_tile_test.csv'
                with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
                assert len({r['sample_id'] for r in rows})==len(rows)
                expected=old if ver==VERSIONS[0] else full
                assert {r['sample_id'] for r in rows}==expected
                cohorts={'old382':old}
                if ver==VERSIONS[1]:cohorts.update(new2413=full-old,full2795=full)
                for name,ids in cohorts.items():
                    sub=[r for r in rows if r['sample_id'] in ids]
                    assert len(sub)==len(ids)
                    runs.append(dict(version=ver,experiment=exp,seed=seed,cohort=name,**aggregate(sub)))
    summary=[]
    for ver in VERSIONS:
        for exp in ('A0','A1','B0','B1'):
            for cohort in ('old382','new2413','full2795'):
                rs=[r for r in runs if (r['version'],r['experiment'],r['cohort'])==(ver,exp,cohort)]
                if not rs:continue
                d=dict(version=ver,experiment=exp,cohort=cohort,n=rs[0]['n'])
                for k in ('iou','dice','precision','recall','object_recall','tiny_recall'):
                    vals=[r[k] for r in rs if r[k] is not None]
                    d[k]=mean(vals) if vals else None
                    d[k+'_std']=pstdev(vals) if vals else None
                summary.append(d)
    (OUT/'fixed_cohort_comparison.json').write_text(json.dumps({'source':'existing metric-v2 per-tile confusion counts; no new inference','runs':runs,'summary':summary},indent=2),encoding='utf-8')
    for r in summary:print(r['version'],r['experiment'],r['cohort'],round(r['iou'],5))

if __name__=='__main__':main()

"""Bounded research lookup; --refresh-stages renders document sections from one ledger."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets/research'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def stages():
    return read(ASSETS/'stage_ledger.json')


def refresh():
    data=stages()
    lines=['| 阶段 | 数据划分 | 实验目标 | 相对上一阶段改变了什么 | 结果与状态 |', '|---|---|---|---|---|']
    for s in data['stages']:
        lines.append('| '+' | '.join([s['name'],s.get('data_split_display','待核对'),s['goal'],s['delta'],s['result']+' **'+s['status']+'**'])+' |')
    p=ROOT/'docs/MODEL_EVOLUTION.md';body=p.read_text(encoding='utf-8')
    start='<!-- STAGE_TABLE_START -->';end='<!-- STAGE_TABLE_END -->'
    assert body.count(start)==body.count(end)==1
    before,rest=body.split(start);_,after=rest.split(end)
    p.write_text(before+start+'\n\n'+'\n'.join(lines)+'\n\n'+end+after,encoding='utf-8')
    print(f"Refreshed {len(data['stages'])} stages in MODEL_EVOLUTION.md")
    refresh_evidence(data)
    refresh_current_models(data)


def refresh_evidence(data):
    """Generate the whole-history reader view from the same stage ledger."""
    lines=['| 数据划分及当时评价集合 | 历史模型：与谁相比、改了什么 | 当时实际结果 | 证据与可能含义 |',
           '|---|---|---|---|']
    count=0
    for stage in data['stages']:
        if stage['id'] in ('terrain-prior','spectral-controlled'):
            continue
        for comparison in stage.get('comparisons', []):
            evidence=comparison.get('evidence',stage.get('evidence',[]))
            links=[]
            for i,source in enumerate(evidence):
                target=ROOT/source
                if not target.exists():raise ValueError(f'Missing experiment evidence: {source}')
                rel=os.path.relpath(target,ROOT/'docs').replace('\\','/')
                links.append(f'[证据{i+1}]({rel})')
            left=f"**{comparison['name']}**：{comparison['comparison']}"
            context=comparison.get('data_split_display',stage['data_split_display'])+'<br>'+comparison['evaluation_context']
            cells=[context,left,comparison['observation'],comparison['possible_answer']+' '+''.join(links)]
            lines.append('| '+' | '.join(s.replace('|','／').replace('\n',' ') for s in cells)+' |')
            count+=1
    p=ROOT/'docs/RESEARCH_INDEX.md';body=p.read_text(encoding='utf-8')
    start='<!-- EXPERIMENT_EVIDENCE_START -->';end='<!-- EXPERIMENT_EVIDENCE_END -->'
    assert body.count(start)==body.count(end)==1
    before,rest=body.split(start);_,after=rest.split(end)
    p.write_text(before+start+'\n\n'+'\n'.join(lines)+'\n\n'+end+after,encoding='utf-8')
    print(f'Refreshed {count} comparison/diagnostic entries in RESEARCH_INDEX.md')


def refresh_current_models(data):
    stage=next(s for s in data['stages'] if s['id']=='terrain-prior')
    results={r['exp']:r for r in read(ROOT/stage['current_results_source'])}
    base=results[stage['reference_experiment']]['best_val']
    lines=['| 研究身份 | 数据划分 | 模型结构与输入变化 | 地形分支在分割训练之前做了什么 | 验证事件平均 IoU |',
           '|---|---|---|---|---:|']
    for variant in stage['current_variants']:
        row=results[variant['experiment']]
        delta=row['best_val']-base
        value=f"{row['best_val']:.4f}"+('（参照）' if variant['experiment']==stage['reference_experiment'] else f'（比当前参照{delta:+.4f}）')
        cells=[variant['status'],stage['data_split_display'],variant['model_change'],variant['training_method'],value]
        lines.append('| '+' | '.join(c.replace('|','／').replace('\n',' ') for c in cells)+' |')
    p=ROOT/'docs/RESEARCH_INDEX.md';body=p.read_text(encoding='utf-8')
    start='<!-- CURRENT_MODELS_START -->';end='<!-- CURRENT_MODELS_END -->'
    assert body.count(start)==body.count(end)==1
    before,rest=body.split(start);_,after=rest.split(end)
    p.write_text(before+start+'\n\n'+'\n'.join(lines)+'\n\n'+end+after,encoding='utf-8')
    print(f'Refreshed {len(stage["current_variants"])} current-reference/terrain-control rows')


def archive_catalog():
    return read(ROOT/'archive/experiment_history/catalog.json')


def resolve_history_links(record, body):
    mapping={r['original_path']:r['archived_path'] for r in archive_catalog()['files']}
    results=[]
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',body):
        if target.startswith(('http:', 'https:', '#')):continue
        local=target.split('#')[0].strip('<>')
        original=(ROOT/record['original_path']).parent/local
        original=original.resolve()
        try: rel=original.relative_to(ROOT).as_posix()
        except ValueError: continue
        results.append({'original_link':target,'resolved_path':mapping.get(rel,rel)})
    return results[:20]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--policy',action='store_true',help='只读当前跨Agent共享实验协议')
    group.add_argument('--stage',help='阶段id、中文关键词或all；仅读阶段台账')
    group.add_argument('--runs',help='按名称、旧路径或协议查run摘要，不输出完整配置')
    group.add_argument('--history',help='按文件名查归档元数据，不读正文')
    group.add_argument('--history-file',help='显式读取指定原始路径或归档路径的一小段')
    group.add_argument('--refresh-stages',action='store_true')
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--start-line',type=int,default=1)
    p.add_argument('--lines',type=int,default=60)
    a=p.parse_args()
    if a.policy:
        print(json.dumps(read(ASSETS/'research_policy.json'),ensure_ascii=False,indent=2));return
    if a.refresh_stages:refresh();return
    if not any((a.stage,a.runs,a.history,a.history_file)):
        d=stages();s=next(s for s in d['stages'] if s['id']==d['current_stage'])
        print(json.dumps({'policy':d['policy'],'protocol_contract':'docs/assets/research/research_policy.json；scripts/query_research.py --policy','agent_rules':'AGENTS.md','current_stage':s,'read_next':'docs/RESEARCH_INDEX.md；按问题选一份固定文档，不默认读取历史'},ensure_ascii=False,indent=2));return
    if a.stage:
        all_stages=stages()['stages']
        exact=[s for s in all_stages if a.stage==s['id']]
        rows=exact or [s for s in all_stages if a.stage=='all' or a.stage.lower() in json.dumps(s,ensure_ascii=False).lower()]
        if not rows:raise SystemExit('No matching stage')
        print(json.dumps(rows,ensure_ascii=False,indent=2));return
    if a.runs:
        rows=read(ASSETS/'experiment_registry.json')['runs']
        rows=[r for r in rows if a.runs.lower() in json.dumps({k:r.get(k) for k in ['display_name','run_dir','protocol','treatment','data_context']},ensure_ascii=False).lower()]
        keys=['run_id','display_name','protocol','data_context','treatment','status','global_step','run_dir']
        print(json.dumps({'matches':len(rows),'showing':min(len(rows),max(1,a.limit)), 'runs':[{k:r.get(k) for k in keys} for r in rows[:max(1,a.limit)]]},ensure_ascii=False,indent=2));return
    rows=archive_catalog()['files']
    if a.history:
        rows=[r for r in rows if a.history.lower() in r['original_path'].lower()]
        print(json.dumps({'matches':len(rows),'files':rows[:max(1,a.limit)]},ensure_ascii=False,indent=2));return
    exact=[r for r in rows if a.history_file==r['archived_path']]
    rows=exact or [r for r in rows if a.history_file==r['original_path'] and r.get('is_latest',True)]
    if len(rows)!=1:raise SystemExit('Specify exactly one catalogued original_path or archived_path')
    r=rows[0];path=ROOT/r['archived_path']
    if path.suffix.lower() not in ('.md','.txt','.log','.csv','.json','.jsonl'):raise SystemExit('Binary artifact: open directly only when needed')
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'],'Archive changed; inspect before citing'
    body=raw.decode('utf-8-sig');lines=body.splitlines();lo=max(0,a.start_line-1);hi=min(len(lines),lo+max(1,min(a.lines,120)))
    print(json.dumps({'file':r['archived_path'],'sha256_verified':True,'total_lines':len(lines),'range':[lo+1,hi], 'resolved_links':resolve_history_links(r,'\n'.join(lines[lo:hi]))},ensure_ascii=False,indent=2))
    print('\n'.join(f'{i+1}: {lines[i]}' for i in range(lo,hi)))


if __name__=='__main__':main()

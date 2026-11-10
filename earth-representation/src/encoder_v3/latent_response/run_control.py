"""Durable optimizer-attempt accounting and checkpoint/log reconciliation."""
import json
import math
import os
from pathlib import Path
from .checkpoint import digest,atomic_replace


class BudgetExhausted(Exception):
    pass


def lr_multiplier(step,total,warmup):
    if total<1 or warmup<0 or warmup>=total:raise ValueError('Require 0 <= warmup < total')
    if warmup and step<warmup:return (step+1)/warmup
    progress=min(1.,max(0.,(step-warmup)/(total-warmup)))
    return .5*(1+math.cos(math.pi*progress))


def require_execution_identity(approval,config,identity,rows):
    if approval.get('model_identity')!=identity:raise PermissionError('Authorization must bind the exact code/data/model identity')
    if approval.get('suspended',False):raise PermissionError('Training authorization suspended')
    if config.get('enmap_required') and not any(r['role']=='TRAIN' and r['sensor']=='EnMAP' for r in rows):
        raise PermissionError('Required EnMAP TRAIN observations are absent')


def append_event(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())


def _records(path):
    if not path.exists():return [],''
    lines=path.read_text(encoding='utf-8').splitlines(keepends=True);records=[]
    for index,line in enumerate(lines):
        try:records.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise ValueError('Corrupt interior run log; refuse ambiguous recovery')
            return records,line
    return records,''


def reconcile_logs(run,state):
    run=Path(run);events,event_tail=_records(run/'optimizer_attempts.jsonl')
    intents=[e['sequence'] for e in events if e['event']=='intent']
    if intents!=list(range(1,len(intents)+1)):raise ValueError('Noncontiguous optimizer attempt journal')
    if state['cursor'] and not intents:raise ValueError('Checkpoint has no durable optimizer attempt journal')
    if state.get('spent_updates',state['cursor'])>len(intents):raise ValueError('Checkpoint exceeds durable budget ledger')
    rows,log_tail=_records(run/'updates.jsonl')
    committed=[r for r in rows if r['cursor']<=state['cursor']]
    if [r['cursor'] for r in committed]!=list(range(1,state['cursor']+1)):
        raise ValueError('Committed log does not match checkpoint cursor')
    uncommitted=[r for r in rows if r['cursor']>state['cursor']]
    if uncommitted or log_tail or event_tail:
        append_event(run/'recovery_attempts.jsonl',dict(checkpoint_cursor=state['cursor'],uncommitted_records=uncommitted,
            partial_update_tail=log_tail,partial_event_tail=event_tail,budget_rule='all durable intents count, including uncertain or replayed updates'))
    for name,records in [('updates.jsonl',committed),('optimizer_attempts.jsonl',events)]:
        path=run/name;temporary=path.with_suffix('.recovering')
        with temporary.open('w',encoding='utf-8') as stream:
            for row in records:stream.write(json.dumps(row)+'\n')
            stream.flush();os.fsync(stream.fileno())
        atomic_replace(temporary,path)
    # Even an intent without an applied record is conservatively charged to the real budget.
    state['spent_updates']=len(intents)
    return dict(committed=state['cursor'],spent_updates=len(intents),uncommitted=len(uncommitted))

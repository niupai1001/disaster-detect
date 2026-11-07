"""Independently recompute TRAIN-only native reflectance moments; never replace scaling."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from .data import checked_file,sha256


def merge_moments(state,values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x):return state
    if not np.isfinite(x).all():raise ValueError('Nonfinite valid TRAIN reflectance')
    n=len(x);mean=float(x.mean());m2=float(((x-mean)**2).sum());old=state['count'];total=old+n;delta=mean-state['mean']
    state['m2']+=m2+delta*delta*old*n/total;state['mean']+=delta*n/total;state['count']=total
    state['min']=min(state['min'],float(x.min()));state['max']=max(state['max'],float(x.max()))
    return state


def main(args):
    study=json.loads(args.study.read_text(encoding='utf-8'));release=Path(study['release']);members=release/'members.jsonl';stats=release/'statistics.json'
    if sha256(members)!=study['source_identities']['members'] or sha256(stats)!=study['source_identities']['statistics']:raise ValueError('Frozen source inventory/statistics changed')
    legacy=json.loads(stats.read_text(encoding='utf-8-sig'));rows=[row for row in map(json.loads,members.open(encoding='utf-8-sig')) if row['role']=='TRAIN'];states={};start=time.perf_counter();sources=[]
    report=dict(status='RUNNING_TRAIN_STATISTICS_AUDIT',optimizer_updates=0,validation_or_protected_pixel_reads=0,source_identities=study['source_identities'],members_total=len(rows),members_checked=0,normalization_changed=False)
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(args.output)
    save()
    for row in rows:
        if row['units']!='surface_reflectance':raise ValueError('Only original optical surface reflectance allowed')
        path=Path(row['cache']);before=path.stat();checked_file(path,row['cache_sha256'])
        with np.load(path,allow_pickle=False) as arrays:
            for band in row['bands']:
                name=band['band_key'];x=arrays[name];mask=arrays[name+'__valid']
                if mask.dtype!=np.bool_ or x.shape!=tuple(band['shape']) or mask.shape!=x.shape:raise ValueError('Native cache grid/mask mismatch')
                key=row['instrument']+':'+name;state=states.setdefault(key,dict(count=0,mean=0.,m2=0.,min=float('inf'),max=float('-inf')));merge_moments(state,x[mask])
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('TRAIN cache changed during read')
        sources.append(dict(id=row['id'],cache_sha256=row['cache_sha256']));report['members_checked']+=1
        if report['members_checked']%500==0:report['seconds']=time.perf_counter()-start;save();print('TRAIN_STATS',report['members_checked'],len(rows),flush=True)
    bands={}
    for key,state in states.items():
        count=state['count'];std=math.sqrt(max(0.,state['m2']/count)) if count else None;reference=legacy['bands'].get(key)
        bands[key]=dict(valid_pixels=count,mean=state['mean'] if count else None,std_population=std,min=state['min'] if count else None,max=state['max'] if count else None,legacy=reference,count_matches=bool(reference and reference['valid_pixels']==count),mean_difference=None if not reference or not count else state['mean']-reference['mean'],std_difference=None if not reference or std is None else std-reference['std'])
    if sha256(members)!=study['source_identities']['members'] or sha256(stats)!=study['source_identities']['statistics']:raise ValueError('Frozen inventory/statistics changed')
    report.update(status='TRAIN_STATISTICS_RECOMPUTED_NOT_V3_FREEZE',bands=bands,member_identities=sources,seconds=time.perf_counter()-start,missing_legacy_keys=sorted(set(states)-set(legacy['bands'])),legacy_keys_without_train=sorted(set(legacy['bands'])-set(states)),all_counts_match=all(b['count_matches'] for b in bands.values()),max_absolute_mean_difference=max(abs(b['mean_difference']) for b in bands.values() if b['mean_difference'] is not None),max_absolute_std_difference=max(abs(b['std_difference']) for b in bands.values() if b['std_difference'] is not None),limits=['Full existing TRAIN release only, not full-machine or approved V3 membership','Population standard deviations; no new loss scaling selected','Original statistics and every source cache unchanged','No validation, GUARD or TEST pixel opened'])
    save();print(json.dumps({k:report[k] for k in ('status','members_checked','all_counts_match','max_absolute_mean_difference','max_absolute_std_difference','seconds')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

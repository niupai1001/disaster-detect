"""Durable pre-step reservations: crashes never silently return update budget."""
from contextlib import contextmanager
import json
import os
from pathlib import Path


class PilotBudget:
    def __init__(self,path,*,study_sha256,total_limit=400,per_candidate_limit=200):
        if len(study_sha256)!=64 or any(c not in '0123456789abcdef' for c in study_sha256):raise ValueError('Explicit study SHA256 required')
        if not 1<=total_limit<=400 or not 1<=per_candidate_limit<=200:raise ValueError('Pilot limits cannot exceed approved 400/200 caps')
        self.path=Path(path);self.study=study_sha256;self.total_limit=total_limit;self.per_candidate_limit=per_candidate_limit

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);lock=self.path.with_suffix(self.path.suffix+'.lock')
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        try:
            os.write(descriptor,str(os.getpid()).encode());os.fsync(descriptor);os.close(descriptor);descriptor=None
            yield
        finally:
            if descriptor is not None:os.close(descriptor)
            lock.unlink()

    def read(self):
        record=json.loads(self.path.read_text()) if self.path.exists() else dict(study_sha256=self.study,total_limit=self.total_limit,per_candidate_limit=self.per_candidate_limit,events=[])
        if (record['study_sha256'],record['total_limit'],record['per_candidate_limit'])!=(self.study,self.total_limit,self.per_candidate_limit):raise ValueError('Existing pilot belongs to a different study/budget')
        reservations={};completed=set()
        for sequence,event in enumerate(record['events'],1):
            if event['sequence']!=sequence:raise ValueError('Malformed pilot event sequence')
            token=event['reservation']
            if event['kind']=='reserved':
                if token in reservations or event['candidate'] not in ('AdamW','CAdamW'):raise ValueError('Malformed pilot reservation')
                reservations[token]=event
            elif event['kind']=='completed':
                if token not in reservations or token in completed:raise ValueError('Malformed completion record')
                completed.add(token)
            else:raise ValueError('Unknown pilot event')
        if len(reservations)>self.total_limit or any(sum(x['candidate']==name for x in reservations.values())>self.per_candidate_limit for name in ('AdamW','CAdamW')):raise ValueError('Pilot ledger exceeds authorized limits')
        return record,reservations,completed

    def write(self,record):
        temp=self.path.with_suffix(self.path.suffix+'.tmp')
        with temp.open('w',encoding='utf-8') as stream:
            json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,self.path)

    def reserve(self,candidate,logical_update):
        if candidate not in ('AdamW','CAdamW') or type(logical_update) is not int or not 0<=logical_update<200:raise ValueError('Invalid candidate/update')
        with self.locked():
            record,reservations,_=self.read()
            if len(reservations)>=self.total_limit or sum(x['candidate']==candidate for x in reservations.values())>=self.per_candidate_limit:raise RuntimeError('Pilot budget exhausted; no automatic extension')
            token=len(record['events'])+1
            record['events'].append(dict(sequence=token,kind='reserved',reservation=token,candidate=candidate,logical_update=logical_update))
            self.write(record)
        return token

    def complete(self,token):
        with self.locked():
            record,reservations,completed=self.read()
            if token not in reservations or token in completed:raise ValueError('Unknown/already completed pilot reservation')
            record['events'].append(dict(sequence=len(record['events'])+1,kind='completed',reservation=token));self.write(record)

    def summary(self):
        record,reservations,completed=self.read()
        return dict(study_sha256=self.study,reserved=len(reservations),completed=len(completed),incomplete_reservations=sorted(set(reservations)-completed),remaining_total=self.total_limit-len(reservations),remaining_by_candidate={name:self.per_candidate_limit-sum(x['candidate']==name for x in reservations.values()) for name in ('AdamW','CAdamW')})

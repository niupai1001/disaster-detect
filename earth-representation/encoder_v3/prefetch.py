"""Manifest-bound CPU prefetch; worker consumption never commits logical progress."""
from dataclasses import dataclass
import hashlib
import json
import torch
from torch.utils.data import DataLoader


def observations_only(values):
    return values


@dataclass
class UpdateBatch:
    logical_update: int
    observations: list
    plan: dict


class ReplayLoader:
    def __init__(self,dataset,entries,*,source_identities,effective_batch,workers,pin_memory):
        if source_identities!=dataset.identities:raise ValueError('Schedule belongs to another source release')
        if workers<0 or effective_batch<1 or not entries:raise ValueError('Explicit nonempty schedule/batch and worker count required')
        self.dataset=dataset;self.entries=json.loads(json.dumps(entries));self.identities=dict(source_identities)
        self.schedule_hash=hashlib.sha256(json.dumps(self.entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        lookup={row['id']:i for i,row in enumerate(dataset.rows)}
        if len(lookup)!=len(dataset.rows):raise ValueError('Dataset has duplicate identities')
        updates=[entry['logical_update'] for entry in self.entries]
        if not all(type(step) is int and step>=0 for step in updates) or updates!=list(range(updates[0],updates[0]+len(updates))):
            raise ValueError('Schedule updates must be contiguous nonnegative integers')
        self.indices=[]
        for entry in self.entries:
            members=entry['members']
            if len(members)!=effective_batch or len(set(members))!=len(members):raise ValueError('Schedule must explicitly contain the effective batch of distinct members')
            self.indices.append([lookup[member] for member in members])
        self.next_update=updates[0];self.workers=workers;self.pin_memory=pin_memory;self._running=False;self._delivered=None

    def state_dict(self):
        return dict(schedule_sha256=self.schedule_hash,source_identities=self.identities,next_update=self.next_update)

    def load_state_dict(self,state):
        if self._running:raise RuntimeError('Close the old iterator before restoring progress')
        if state['schedule_sha256']!=self.schedule_hash or state['source_identities']!=self.identities:
            raise ValueError('Replay identity changed')
        step=state['next_update']
        if type(step) is not int or not self.entries[0]['logical_update']<=step<=self.entries[-1]['logical_update']+1:raise ValueError('Invalid replay cursor')
        self.next_update=step

    def commit(self,logical_update):
        if not self._running or logical_update!=self.next_update or self._delivered!=logical_update:raise ValueError('Only the current consumed update can be committed')
        self.next_update+=1
        self._delivered=None

    def __iter__(self):
        if self._running:raise RuntimeError('Concurrent replay iterators are not allowed')
        if self.dataset.identities!=self.identities:raise ValueError('Source release changed after loader construction')
        self._running=True
        offset=self.next_update-self.entries[0]['logical_update']
        kwargs=dict(batch_sampler=self.indices[offset:],num_workers=self.workers,pin_memory=self.pin_memory,collate_fn=observations_only)
        if self.workers:kwargs.update(multiprocessing_context='spawn',prefetch_factor=2)
        loader=DataLoader(self.dataset,**kwargs);iterator=None
        try:
            iterator=iter(loader)
            for entry,observations in zip(self.entries[offset:],iterator):
                if entry['logical_update']!=self.next_update:raise RuntimeError('Previous logical update was not committed')
                if [o.identifier for o in observations]!=entry['members']:raise RuntimeError('Prefetch returned different observations')
                if any(o.metadata.get('frozen_source_ids')!=self.identities for o in observations):raise ValueError('Worker returned a different source release')
                self._delivered=entry['logical_update']
                yield UpdateBatch(entry['logical_update'],observations,entry)
                if self.next_update!=entry['logical_update']+1:raise RuntimeError('Consumer must commit before requesting another update')
        finally:
            # PyTorch 2.11 iterator cleanup; prevents leaked Windows spawn workers
            # when an update is interrupted while future batches are prefetched.
            shutdown=getattr(iterator,'_shutdown_workers',None)
            if shutdown is not None:shutdown()
            self._running=False
            self._delivered=None

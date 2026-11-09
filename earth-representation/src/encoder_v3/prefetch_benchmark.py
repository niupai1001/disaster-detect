"""Matched real manifest I/O; startup separated from subsequent batch waits."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import time
import torch
from .data import NativeOpticalDataset
from .prefetch import ReplayLoader


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True);args=p.parse_args()
    torch.cuda.set_device(torch.device(args.device));rows=[];reference=None
    for workers,pinned in ((0,False),(2,False),(2,True)):
        dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN')
        entries=[dict(logical_update=i,members=[dataset.rows[8*i+j]['id'] for j in range(8)]) for i in range(4)]
        loader=ReplayLoader(dataset,entries,source_identities=dataset.identities,effective_batch=8,workers=workers,pin_memory=pinned)
        waits=[];identities=[];pin_checks=[]
        with closing(iter(loader)) as iterator:
            for entry in entries:
                start=time.perf_counter();batch=next(iterator);waits.append(time.perf_counter()-start)
                identities.append([o.identifier for o in batch.observations])
                pin_checks.extend(b.reflectance.is_pinned() for o in batch.observations for b in o.bands)
                loader.commit(batch.logical_update)
        if reference is None:reference=identities
        if identities!=reference:raise RuntimeError('Different source schedule after prefetch')
        if pinned and not all(pin_checks):raise RuntimeError('Custom observation tensors were not pinned')
        rows.append(dict(workers=workers,pin_memory=pinned,batch_wait_seconds=waits,startup_first_batch_seconds=waits[0],subsequent_mean_wait_seconds=sum(waits[1:])/3,completed_updates=0,consumed_diagnostic_batches=4,source_identities=dataset.identities))
        print('PREFETCH_CASE_COMPLETE',workers,pinned,flush=True)
    report=dict(optimizer_updates=0,global_effective_batch=8,rows=rows,limitations=['CPU read/prefetch only, no GPU model workload to hide I/O','Four batches cannot establish long-run optimum','No frozen training membership or worker-count acceptance'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

"""Bounded two-rank communication capability probe; no model or optimizer."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import os
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,init_method,output):
    record=dict(rank=rank,pid=os.getpid(),phase='initializing',cpu_allreduce=False,cuda_allreduce=False,error=None)
    path=Path(output,f'rank_{rank}.json')
    path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    try:
        dist.init_process_group('gloo',init_method=init_method,rank=rank,world_size=2,timeout=timedelta(seconds=15))
        record['phase']='cpu_allreduce';path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        cpu=torch.tensor([float(rank+1)]);dist.all_reduce(cpu)
        record['cpu_allreduce']=float(cpu)==3.
        record['phase']='cuda_allreduce';path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        torch.cuda.set_device(rank)
        gpu=torch.tensor([float(rank+1)],device=f'cuda:{rank}')
        dist.all_reduce(gpu);torch.cuda.synchronize(rank)
        record['cuda_allreduce']=float(gpu)==3.
        record['phase']='completed'
    except Exception as exc:
        record['error']=type(exc).__name__+': '+str(exc)[:2000]
    finally:
        path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        if dist.is_initialized():dist.destroy_process_group()


def join_bounded(context,seconds):
    deadline=time.monotonic()+seconds
    try:
        while time.monotonic()<deadline:
            if context.join(timeout=min(1.,max(0.,deadline-time.monotonic()))):return True
        return False
    finally:
        for process in context.processes:
            if process.is_alive():process.terminate()
        for process in context.processes:
            process.join(timeout=3)
            if process.is_alive():process.kill();process.join(timeout=3)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--transport',choices=['file','tcp'],required=True);args=parser.parse_args()
    report=dict(optimizer_updates=0,world_size=2,torch=torch.__version__,nccl=dist.is_nccl_available(),gloo=dist.is_gloo_available(),ranks=[],status='NOT_RUN')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if torch.cuda.device_count()<2 or not dist.is_gloo_available():report['status']='UNAVAILABLE'
    else:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root=Path(directory).resolve()
            if not root.is_relative_to(Path(__file__).resolve().parent):raise ValueError('Unexpected temporary scope')
            if args.transport=='tcp':
                with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                init_method=f'tcp://127.0.0.1:{port}'
            else:init_method=(root/'rendezvous').as_uri()
            context=mp.spawn(worker,args=(init_method,str(root)),nprocs=2,join=False)
            report['controller_pid']=os.getpid();report['child_pids']=[p.pid for p in context.processes]
            try:report['finished_within_45_seconds']=join_bounded(context,45)
            except Exception as exc:report['controller_error']=type(exc).__name__+': '+str(exc)[:2000]
            report['ranks']=[json.loads((root/f'rank_{rank}.json').read_text()) if (root/f'rank_{rank}.json').exists() else dict(rank=rank,phase='no_worker_record',cuda_allreduce=False) for rank in range(2)]
        if not report.get('finished_within_45_seconds'):report['status']='PROBE_TIMEOUT_OR_WORKER_ERROR_NOT_DDP_ACCEPTANCE'
        else:report['status']='CUDA_COMMUNICATION_ONLY_NOT_DDP_ACCEPTANCE' if all(row.get('cuda_allreduce') for row in report['ranks']) else 'CUDA_COLLECTIVE_FAILED_NO_DDP_ACCEPTANCE'
    report['transport']=args.transport
    report['limitations']=['Scalar collective only; no gradient/normalization/recovery/throughput acceptance','No environment migration, installs, training or model parameter updates']
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

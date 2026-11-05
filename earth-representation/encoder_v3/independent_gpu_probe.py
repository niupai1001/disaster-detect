"""Compare serial versus simultaneous independent GPU workloads, no collectives."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import os
import time
import torch
import torch.multiprocessing as mp
from .data import NativeOpticalDataset
from .diagnostic_batch import prepare_batch,device_views
from .model import Encoder,ReflectanceDecoder
from .joint import accumulate_joint
from .communication_probe import join_bounded


def weight_digest(encoder,decoder):
    h=hashlib.sha256()
    for name,module in [('encoder',encoder),('decoder',decoder)]:
        for key,value in module.state_dict().items():h.update((name+key).encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def worker(rank,args,commands,results):
    status=Path(args.output).parent/f'independent_gpu_worker_{rank}.json'
    def stage(name):status.write_text(json.dumps(dict(rank=rank,pid=os.getpid(),stage=name)),encoding='utf-8')
    try:
        stage('initializing_cuda')
        torch.set_num_threads(2);torch.manual_seed(0);device=f'cuda:{rank}';torch.cuda.set_device(rank)
        stage('reading_source_metadata')
        dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN');observations,rectangles,plans,stds=prepare_batch(dataset,patch_m=args.patch_m)
        stage('transferring_observations')
        views=device_views(observations,rectangles,plans,stds,device)
        stage('constructing_model')
        encoder=Encoder(hyper_hidden=args.hyper_hidden,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(device).eval();decoder=ReflectanceDecoder(indexed=True).to(device).eval()
        initial=weight_digest(encoder,decoder)
        stage('ready')
        results.put(dict(kind='ready',rank=rank,members=[o.identifier for o in observations],weights=initial))
        while True:
            try:mode=commands[rank].get(timeout=1)
            except queue.Empty:continue
            if mode=='stop':return
            stage(mode)
            torch.cuda.synchronize(rank);torch.cuda.reset_peak_memory_stats(rank);start=time.perf_counter()
            report=accumulate_joint(encoder,decoder,views,logical_update=0,chunk_pixels=512,precision='fp32',clip_norm=3.)
            torch.cuda.synchronize(rank);elapsed=time.perf_counter()-start
            if weight_digest(encoder,decoder)!=initial:raise RuntimeError('Zero-update diagnostic changed model state')
            results.put(dict(kind='done',rank=rank,mode=mode,seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(rank),weighted_loss=report['weighted_loss'],effective_batch=report['effective_batch'],gradient_norm=report['gradient_norm_before_clip'],B2=report['losses_by_kind']['B2']))
            stage(mode+'_complete')
    except Exception as exc:
        stage('error_'+type(exc).__name__)
        results.put(dict(kind='error',rank=rank,error=type(exc).__name__+': '+str(exc)))
        raise


def receive(results,expected_kind,expected_ranks,seconds):
    received=[];deadline=time.monotonic()+seconds
    while len(received)<len(expected_ranks):
        if time.monotonic()>deadline:raise TimeoutError('Independent GPU diagnostic exceeded its phase deadline')
        try:message=results.get(timeout=1)
        except queue.Empty:continue
        if message['kind']=='error':raise RuntimeError(str(message))
        if message['kind']!=expected_kind or message['rank'] not in expected_ranks or message['rank'] in [m['rank'] for m in received]:raise ValueError('Unexpected worker message')
        received.append(message)
    return received


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--patch-m',type=float,required=True);p.add_argument('--hyper-hidden',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if torch.cuda.device_count()<2:raise ValueError('Two CUDA devices required')
    context=mp.get_context('spawn');commands=[context.Queue(),context.Queue()];results=context.Queue()
    processes=mp.spawn(worker,args=(args,commands,results),nprocs=2,join=False)
    report=dict(status='RUNNING',phase='initializing',controller_pid=os.getpid(),child_pids=[process.pid for process in processes.processes],optimizer_updates=0,collectives=False,phases=[],limitations=['Identical A/B1 workloads on each GPU, not the final A+B versus A+B+JEPA comparison','Engineering cohort and mask recipe only','Includes no optimizer or checkpoint saving; no frozen execution acceptance'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    try:
        ready=receive(results,'ready',[0,1],120);report['ready']=ready
        if ready[0]['members']!=ready[1]['members'] or ready[0]['weights']!=ready[1]['weights']:raise ValueError('Independent workloads have different initial states/data')
        for rank in (0,1):
            report['phase']=f'warmup_rank_{rank}';save()
            commands[rank].put('warmup');receive(results,'done',[rank],180)
        serial=[];start=time.perf_counter()
        for rank in (0,1):
            report['phase']=f'serial_rank_{rank}';save()
            commands[rank].put('serial');serial.extend(receive(results,'done',[rank],180))
        report['phases'].append(dict(mode='serial',wall_seconds=time.perf_counter()-start,workers=serial));save();print('SERIAL_COMPLETE',flush=True)
        start=time.perf_counter()
        report['phase']='parallel';save()
        for command in commands:command.put('parallel')
        parallel=receive(results,'done',[0,1],180)
        report['phases'].append(dict(mode='parallel',wall_seconds=time.perf_counter()-start,workers=parallel))
        for row in parallel:
            baseline=next(v for v in serial if v['rank']==row['rank'])
            if abs(row['weighted_loss']-baseline['weighted_loss'])>1e-5:raise ValueError('Parallel execution changed diagnostic loss')
        report['aggregate_speedup']=report['phases'][0]['wall_seconds']/report['phases'][1]['wall_seconds'];report['status']='INDEPENDENT_GPU_DIAGNOSTIC_COMPLETE_NOT_FORMAL_ACCEPTANCE'
        for command in commands:command.put('stop')
        if not join_bounded(processes,15):raise TimeoutError('GPU diagnostic workers did not exit')
        report['phase']='completed'
    except Exception as exc:
        report['status']='FAILED';report['error']=type(exc).__name__+': '+str(exc)
        for process in processes.processes:
            if process.is_alive():process.terminate()
        for process in processes.processes:
            process.join(timeout=3)
            if process.is_alive():process.kill();process.join(timeout=3)
    finally:save()
    print(json.dumps({key:report[key] for key in ('status','optimizer_updates','phases')},indent=2))
    raise SystemExit(1 if report['status']=='FAILED' else 0)

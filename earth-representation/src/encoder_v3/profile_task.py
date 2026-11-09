"""One complete real task, named operator profiling, no optimizer updates."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import sys
import cProfile
import pstats
import torch
from . import spatial_index
from .data import NativeOpticalDataset
from .diagnostic_batch import prepare_batch,device_views
from .model import Encoder,ReflectanceDecoder
from .streaming import stream_task


def scoped(function,name,timings=None):
    def run(*args,**kwargs):
        start=time.perf_counter()
        try:
            with torch.profiler.record_function(name):return function(*args,**kwargs)
        finally:
            if timings is not None:
                row=timings.setdefault(name,dict(calls=0,wall_seconds=0.));row['calls']+=1;row['wall_seconds']+=time.perf_counter()-start
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('release','contracts','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--view-index',type=int,required=True)
    parser.add_argument('--patch-m',type=float,required=True)
    parser.add_argument('--activity',choices=['python','cpu','cuda'],required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    if not args.device.startswith('cuda:') or not 0<=args.view_index<16:raise ValueError('Explicit CUDA device and one of 16 diagnostic views required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.worker:
        child=subprocess.Popen([sys.executable,'-m','encoder_v3.profile_task',*sys.argv[1:],'--worker'],creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        timed_out=False
        try:code=child.wait(timeout=90)
        except subprocess.TimeoutExpired:
            timed_out=True;child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            code=1
        if code:
            record=json.loads(args.output.read_text(encoding='utf-8')) if args.output.exists() else {}
            record.update(status='PROFILER_TIMEOUT' if timed_out else 'PROFILER_WORKER_FAILED',owned_worker_pid=child.pid,worker_exit_code=child.returncode,optimizer_updates=0)
            args.output.write_text(json.dumps(record,indent=2),encoding='utf-8')
        raise SystemExit(code)
    record=dict(status='RUNNING',optimizer_updates=0,activity=args.activity)
    def phase(name):
        record['phase']=name;args.output.write_text(json.dumps(record,indent=2),encoding='utf-8');print(name,flush=True)
    phase('initializing')
    torch.manual_seed(0);torch.set_num_threads(2)
    dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN')
    observations,rectangles,plans,stds=prepare_batch(dataset,patch_m=args.patch_m)
    view=device_views(observations,rectangles,plans,stds,args.device)[args.view_index]
    encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(args.device).eval();decoder=ReflectanceDecoder(indexed=True).to(args.device).eval()
    kwargs=dict(train_stds=view.train_stds,chunk_pixels=512,loss_weight=1.,backward=True,precision='fp32')
    phase('warmup');stream_task(encoder,decoder,view.task,view.plan,**kwargs);torch.cuda.synchronize(args.device)
    encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
    timings={}
    for owner,attribute,name in [(encoder.srf,'forward','v3.srf'),(encoder.hyper,'forward','v3.hypernetwork'),(encoder,'native_grouped','v3.native_grouped'),(encoder,'_prepare','v3.layout_and_projection'),(encoder,'_local_tokens','v3.local_transformer'),(encoder.global_net,'forward','v3.global_transformer'),(spatial_index,'aggregate','v3.decoder_geometry_aggregate'),(decoder.mlp,'forward','v3.decoder_mlp')]:
        setattr(owner,attribute,scoped(getattr(owner,attribute),name,timings))
    start=time.perf_counter()
    if args.activity=='python':
        phase('executing_python_profile');profile=cProfile.Profile();profile.enable()
        result=stream_task(encoder,decoder,view.task,view.plan,**kwargs)
        torch.cuda.synchronize(args.device);profile.disable();elapsed=time.perf_counter()-start
        functions=[dict(file=key[0],line=key[1],function=key[2],primitive_calls=value[0],calls=value[1],self_wall_seconds=value[2],cumulative_wall_seconds=value[3]) for key,value in pstats.Stats(profile).stats.items()]
        report=dict(status='PYTHON_WALL_PROFILE_NOT_GPU_KERNEL_TIMING',optimizer_updates=0,device=args.device,activity=args.activity,observation=view.task.metadata['id'],kind=view.task.kind,patch_m=args.patch_m,source_identities=dataset.identities,profiled_wall_seconds=elapsed,supported_supervision_pixels=sum(b['count'] for b in result['bands']),tokens=result['tokens'],named_ranges=timings,top_self_wall=sorted(functions,key=lambda r:r['self_wall_seconds'],reverse=True)[:40],top_cumulative_wall=sorted(functions,key=lambda r:r['cumulative_wall_seconds'],reverse=True)[:40],limitations=['Python wall times include host overhead and possible GPU waits; they are not GPU kernel durations or utilization','Single complete engineering task with profiler overhead; not whole-experiment throughput','Named ranges overlap hierarchically and must not be added','No optimizer, communication, validation or saving in this task'])
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(status=report['status'],profiled_wall_seconds=elapsed,named_ranges=timings),indent=2));raise SystemExit(0)
    phase('starting_profiler')
    activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if args.activity=='cuda' else [])
    with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as profile:
        phase('executing_profiled_task')
        with torch.profiler.record_function('v3.complete_task'):
            result=stream_task(encoder,decoder,view.task,view.plan,**kwargs)
        torch.cuda.synchronize(args.device)
    phase('collecting_events')
    def row(event):return dict(name=event.key,count=event.count,self_cpu_us=event.self_cpu_time_total,total_cpu_us=event.cpu_time_total,self_device_us=event.self_device_time_total,total_device_us=event.device_time_total)
    events=profile.key_averages();rows=[row(event) for event in events]
    report=dict(status='OPERATOR_PROFILE_NOT_THROUGHPUT_ACCEPTANCE',optimizer_updates=0,device=args.device,activity=args.activity,observation=view.task.metadata['id'],kind=view.task.kind,patch_m=args.patch_m,source_identities=dataset.identities,profiled_wall_seconds=time.perf_counter()-start,supported_supervision_pixels=sum(b['count'] for b in result['bands']),tokens=result['tokens'],cuda_device_events_observed=any(r['self_device_us']>0 for r in rows),named_ranges=[r for r in rows if r['name'].startswith('v3.')],top_cpu=sorted(rows,key=lambda r:r['self_cpu_us'],reverse=True)[:30],top_device=sorted(rows,key=lambda r:r['self_device_us'],reverse=True)[:30],limitations=['Single complete real B1/A engineering task, not an end-to-end experiment','Profiler instrumentation changes timings; use matched unprofiled microbatch report for throughput','CPU activity alone cannot measure GPU kernel durations or utilization','Named ranges overlap hierarchically; times must not be added as independent stage totals','I/O, prefetch, optimizer, validation, communication and saving are outside this profile'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','kind','tokens','supported_supervision_pixels','profiled_wall_seconds','cuda_device_events_observed','named_ranges')},indent=2))

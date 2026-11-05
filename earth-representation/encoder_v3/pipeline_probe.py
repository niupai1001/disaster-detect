"""Matched synchronous versus one-lookahead complete input/backward pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import torch
from .data import NativeOpticalDataset
from .joint import accumulate_joint
from .model import Encoder,ReflectanceDecoder
from .pilot_protocol import load_study,prepare_update
from .plan_prefetch import PlanPrefetch,move_views


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cell-geometry',action='store_true');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.worker:
        child=subprocess.Popen([sys.executable,'-m','encoder_v3.pipeline_probe',*sys.argv[1:],'--worker'],creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        try:code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            code=1
        if code:
            report=json.loads(args.output.read_text(encoding='utf-8')) if args.output.exists() else {}
            report.update(status='FAILED_OR_DEADLINE',owned_worker_pid=child.pid,worker_exit_code=child.returncode);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise SystemExit(code)
    torch.manual_seed(0);torch.set_num_threads(2);study=load_study(args.study);dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN');device='cuda:0'
    report=dict(status='RUNNING',phase='warming_up',optimizer_updates=0,cell_indexed_geometry=args.cell_geometry,rows=[],device_samples=[],study_sha256=study['candidate_sha256'],limitations=['Four fixed updates, no optimizer or checkpoint I/O; not full-experiment performance','Device utilization includes desktop/other contexts and is not process-specific','Single timing per pipeline; measurements do not establish broad optimal worker settings'])
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(device).eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True).to(device).eval()
    kwargs=dict(chunk_pixels=512,precision='fp32',clip_norm=None)
    cpu,_=prepare_update(study,dataset,0,'cpu',cell_indexed_geometry=args.cell_geometry);accumulate_joint(encoder,decoder,move_views(cpu,device),logical_update=0,**kwargs);torch.cuda.synchronize(device);del cpu
    baseline=None;baseline_gradient=None;stop=threading.Event()
    def sample():
        while not stop.is_set():
            try:
                result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                if result.returncode==0:report['device_samples'].append(dict(time=time.time(),phase=report['phase'],raw=result.stdout.strip()))
            except (OSError,subprocess.TimeoutExpired):pass
            stop.wait(1.)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        for prefetch in (False,True):
            report['phase']='prefetch' if prefetch else 'synchronous';save();rows=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
            stream=PlanPrefetch(lambda index:prepare_update(study,dataset,index,'cpu',cell_indexed_geometry=args.cell_geometry),start=0,stop=4,identity=study['candidate_sha256'],prefetch=prefetch);iterator=iter(stream)
            try:
                for index in range(4):
                    before=time.perf_counter();step,(cpu_views,signature)=next(iterator);wait=time.perf_counter()-before
                    result=accumulate_joint(encoder,decoder,move_views(cpu_views,device),logical_update=index,**kwargs);torch.cuda.synchronize(device)
                    rows.append(dict(update=step,signature=signature,data_wait_seconds=wait,weighted_loss=result['weighted_loss'],supported_pixels=sum(b['count'] for row in result['rows'] for b in row['result']['bands'])))
                    stream.commit(index)
            finally:iterator.close()
            seconds=time.perf_counter()-start;gradient=torch.cat([p.grad.detach().cpu().double().flatten() for module in (encoder,decoder) for p in module.parameters()])
            if baseline is None:baseline=rows;baseline_gradient=gradient
            if [r['signature'] for r in rows]!=[r['signature'] for r in baseline] or [r['supported_pixels'] for r in rows]!=[r['supported_pixels'] for r in baseline]:raise ValueError('Lookahead changed exposure')
            report['rows'].append(dict(prefetch=prefetch,seconds=seconds,updates=rows,observations_per_second=32/seconds,supervision_pixels_per_second=sum(r['supported_pixels'] for r in rows)/seconds,data_wait_seconds=sum(r['data_wait_seconds'] for r in rows),peak_allocated_bytes=torch.cuda.max_memory_allocated(device),max_weighted_loss_difference=max(abs(a['weighted_loss']-b['weighted_loss']) for a,b in zip(rows,baseline)),gradient_relative_l2=float((gradient-baseline_gradient).norm()/baseline_gradient.norm().clamp_min(1e-24))));save()
        report['speedup']=report['rows'][0]['seconds']/report['rows'][1]['seconds'];report['status']='MATCHED_PIPELINE_MEASUREMENT_NOT_FULL_ACCEPTANCE';report['phase']='completed'
    except Exception as exc:report.update(status='FAILED',error=type(exc).__name__+': '+str(exc));raise
    finally:stop.set();sampler.join(timeout=7);save()

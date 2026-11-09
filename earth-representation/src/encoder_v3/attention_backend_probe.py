"""Bounded tiny FP32 attention backend measurements; no optimizer updates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings
from .processes import launch_python


def measure(output):
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention
    from .gpu_update_check import difference
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(0)
    rows=[];references={}
    for length in (128,512):
        inputs=[torch.randn(1,4,length,32,device='cuda',dtype=torch.float32) for _ in range(3)]
        for backend in ('MATH','FLASH_ATTENTION','EFFICIENT_ATTENTION','CUDNN_ATTENTION','DEFAULT'):
            row=dict(length=length,backend=backend,precision='FP32')
            q,k,v=[x.detach().clone().requires_grad_(True) for x in inputs]
            try:
                from contextlib import nullcontext
                context=nullcontext() if backend=='DEFAULT' else sdpa_kernel(getattr(SDPBackend,backend))
                with warnings.catch_warnings(record=True) as caught,context:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                        result=scaled_dot_product_attention(q,k,v,dropout_p=0.);result.square().mean().backward();torch.cuda.synchronize()
                row.update(status='FORWARD_BACKWARD_PASS',warnings=[str(w.message) for w in caught],dispatch_operators=[event.key for event in profiler.key_averages() if 'attention' in event.key])
                state=dict(prediction=result.detach().cpu(),gradients=[x.grad.detach().cpu() for x in (q,k,v)])
                if backend=='MATH':references[length]=state
                else:row['vs_math']={key:difference(references[length][key],state[key]) for key in state}
            except Exception as exc:row.update(status='UNAVAILABLE_OR_FAILED',error=type(exc).__name__+': '+str(exc))
            rows.append(row)
    report=dict(status='TINY_FP32_ATTENTION_BACKENDS_MEASURED_NOT_MODEL_ACCEPTANCE',torch_version=torch.__version__,device=torch.cuda.get_device_name(0),cuda_logical_index=0,rows=rows,optimizer_updates=0,limitations=['Small synthetic lengths only; no full-window feasibility, speedup or peak-memory acceptance','CPU dispatch names identify the selected path, not CUDA kernel timings','No precision, architecture, window or execution default changed'])
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:measure(args.output)
    else:
        record=args.output.with_suffix('.controller.json');log=args.output.with_suffix('.log')
        if record.exists() or args.output.exists():raise ValueError('Probe artifacts already exist')
        with log.open('x',encoding='utf-8') as stream:
            start=time.perf_counter();child=launch_python(['-m','encoder_v3.attention_backend_probe',*sys.argv[1:],'--worker'],stdout=stream)
            controller=dict(status='RUNNING',owned_worker_pid=child.pid,maximum_seconds=90)
            record.write_text(json.dumps(controller,indent=2),encoding='utf-8')
            try:code=child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                code=1
            controller.update(status='COMPLETED' if code==0 else 'FAILED_NO_AUTOMATIC_RETRY',worker_exit_code=child.returncode,seconds=time.perf_counter()-start)
            record.write_text(json.dumps(controller,indent=2),encoding='utf-8');raise SystemExit(code)

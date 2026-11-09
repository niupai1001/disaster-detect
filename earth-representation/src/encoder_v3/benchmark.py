"""Matched synthetic performance probe. Zero optimizer updates, no admission claim."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from .model import Encoder, BandObservation


def run(output, device):
    torch.manual_seed(0)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; do not report CPU timing as GPU acceptance')
    source=BandObservation(torch.rand(24,24,device=device),torch.ones(24,24,dtype=torch.bool,device=device),torch.tensor([450.,500.,550.],device=device),torch.tensor([0.,1.,0.],device=device),(10.,0.,0.,0.,-10.,240.),'EPSG:32631')
    model=Encoder(hyper_hidden=64).to(device).eval()
    initial=copy.deepcopy(model.state_dict()); rows=[]; reference=None; reference_grad=None
    sync=lambda: torch.cuda.synchronize(device) if device.startswith('cuda') else None
    for execution_batch in (1,2,4,8):
        model.load_state_dict(initial); model.local_batch_size=execution_batch
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            model([source],patch_m=60,origin_xy=(0,0)).tokens.square().mean().backward()
        durations=[]
        if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
        for _ in range(3):
            model.zero_grad(set_to_none=True); sync(); start=time.perf_counter()
            z=model([source],patch_m=60,origin_xy=(0,0)).tokens
            z.square().mean().backward(); sync(); durations.append(time.perf_counter()-start)
        grad=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        if reference is None: reference=z.detach().clone(); reference_grad=grad.detach().clone()
        rows.append(dict(local_execution_batch=execution_batch,seconds=durations,mean_seconds=sum(durations)/len(durations),max_output_difference=float((z-reference).detach().abs().max()),relative_gradient_difference=float((grad-reference_grad).detach().norm()/reference_grad.norm().clamp_min(1e-12)),peak_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,tokens=len(z)))
    for key,value in model.state_dict().items():
        if not torch.equal(value,initial[key]): raise RuntimeError('Performance probe changed weights')
    report=dict(optimizer_updates=0,data='synthetic reflectance-like values; NOT real-data acceptance',hyper_hidden=64,hyper_hidden_status='candidate_only',device=device,torch=torch.__version__,rows=rows,scope='local patch batching only; not global effective-batch or dual-GPU acceptance',nccl=torch.distributed.is_nccl_available(),cuda_devices=torch.cuda.device_count(),peer_access=torch.cuda.can_device_access_peer(0,1) if torch.cuda.device_count()>1 else None)
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); run(args.output,args.device)

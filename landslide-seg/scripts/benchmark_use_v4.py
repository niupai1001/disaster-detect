"""Explicit bounded encoder/consumer measurements. B owns benchmark outputs."""
import argparse
import json
import time
import numpy as np
import torch
import pretrain_use_v4 as pt
import run_use_v2 as rt


def measure(model,observation,device,warmup=30,repeats=100,rounds=3):
    model.eval(); rows=[]
    with torch.no_grad():
        for repeat in range(rounds):
            for _ in range(warmup): model(observation)
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            times=[]
            for _ in range(repeats):
                start=time.perf_counter(); model(observation); torch.cuda.synchronize(device)
                times.append((time.perf_counter()-start)*1000)
            rows.append(dict(round=repeat,ms=times,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),
                images_per_second=observation['x'].shape[0]*1000/float(np.median(times)),
                peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device)))
    return rows


def main():
    from run_use_v4 import load_encoder
    from landslide_seg.models.use_v4_reconstruction import Segmenter
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:1'); p.add_argument('--channels',type=int,choices=[3,6,10,16],default=10)
    p.add_argument('--size',type=int,choices=[128,256,512],default=128); p.add_argument('--kind',choices=['encoder','consumer'],default='encoder')
    p.add_argument('--source'); p.add_argument('--out',required=True); a=p.parse_args(); torch.set_num_threads(4); pt.deterministic()
    if a.kind=='consumer' and a.channels!=10: raise ValueError('Real consumer only supports C10')
    with rt.gpu_guard(a.device) as gate:
        model=load_encoder(a.source)
        if a.kind=='consumer': model=Segmenter(model)
        model=model.to(a.device).eval()
        o=dict(x=torch.randn(1,a.channels,a.size,a.size,device=a.device),pixel_valid=torch.ones(1,a.channels,a.size,a.size,dtype=torch.bool,device=a.device))
        rows=measure(model,o,a.device)
        rt.write_json(a.out,dict(device_gate=gate,kind=a.kind,channels=a.channels,size=a.size,rows=rows,
            parameters=sum(p.numel() for p in model.parameters()),precision='FP32',tf32=False,
            limitation='forward-only; excludes IO and state hashes; B must add v3/reference and training/profile costs'))


if __name__=='__main__': main()

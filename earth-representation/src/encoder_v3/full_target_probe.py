"""All native B1 targets, chunked exact gradients, no optimizer updates."""
import argparse
import json
from pathlib import Path
import time
import torch
from .data import NativeOpticalDataset,to_device
from .model import Encoder,ReflectanceDecoder
from .tasks import spectral_task
from .streaming import prepare_task,stream_task


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--patch-m',type=float,required=True);p.add_argument('--hyper-hidden',type=int,required=True);p.add_argument('--chunk-pixels',type=int,required=True);p.add_argument('--precision',choices=['fp32','bf16'],required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.manual_seed(0);dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN');cpu_observation=dataset[args.index]
    cpu_task=spectral_task(cpu_observation,seed=0,update=0);first=cpu_observation.bands[0]
    origin=(first.affine[2],first.affine[5]+first.affine[4]*first.reflectance.shape[0]);start=time.perf_counter()
    plan=prepare_task(cpu_task,patch_m=args.patch_m,origin_xy=origin,chunk_pixels=args.chunk_pixels);planning_seconds=time.perf_counter()-start
    observation=to_device(cpu_observation,args.device);task=spectral_task(observation,seed=0,update=0)
    encoder=Encoder(hyper_hidden=args.hyper_hidden,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(args.device).eval();decoder=ReflectanceDecoder(indexed=True).to(args.device).eval()
    if args.device.startswith('cuda'):torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
    start=time.perf_counter()
    result=stream_task(encoder,decoder,task,plan,train_stds=dataset.engineering_train_stds(cpu_observation),chunk_pixels=args.chunk_pixels,loss_weight=1.,backward=True,precision=args.precision)
    if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
    elapsed=time.perf_counter()-start
    finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in list(encoder.parameters())+list(decoder.parameters()))
    if not finite:raise FloatingPointError('Nonfinite full-target gradients')
    report=dict(status='ENGINEERING_FULL_TARGET_BACKWARD_ONLY',optimizer_updates=0,observation_id=observation.identifier,source_identities=dataset.identities,precision=args.precision,planning_seconds=planning_seconds,forward_backward_seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,finite_gradients=finite,result=result,normalization='Historical hash-bound TRAIN-only source statistics; not V3 frozen statistics',limitations=['One TRAIN observation, no optimizer step or training claim','No target sampling: every valid native hidden-band pixel is included or counted unsupported','Hypernetwork width and execution/precision remain candidates'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

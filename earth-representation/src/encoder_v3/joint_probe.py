"""Real effective batch eight A/B1 backward. Not an optimizer pilot."""
import argparse
import json
from pathlib import Path
import time
import torch
from .data import NativeOpticalDataset
from .diagnostic_batch import prepare_batch,device_views
from .model import Encoder,ReflectanceDecoder
from .joint import accumulate_joint


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--patch-m',type=float,required=True);p.add_argument('--hyper-hidden',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.manual_seed(0);torch.set_num_threads(2);dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN')
    start=time.perf_counter();observations,rectangles,plans,stds=prepare_batch(dataset,patch_m=args.patch_m);planning=time.perf_counter()-start
    views=device_views(observations,rectangles,plans,stds,args.device)
    encoder=Encoder(hyper_hidden=args.hyper_hidden,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(args.device).eval();decoder=ReflectanceDecoder(indexed=True).to(args.device).eval()
    if args.device.startswith('cuda'):torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
    start=time.perf_counter();result=accumulate_joint(encoder,decoder,views,logical_update=0,chunk_pixels=512,precision='fp32',clip_norm=3.)
    if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
    report=dict(status='ENGINEERING_JOINT_BACKWARD_ONLY',optimizer_updates=0,device=args.device,patch_m=args.patch_m,planning_seconds=planning,forward_backward_seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,source_identities=dataset.identities,members=[o.identifier for o in observations],diagnostic_rectangles=rectangles,result=result,limitations=['First four intact four-band S2 and first four L8 records, not a frozen training cohort','Fixed patch-aligned half-width diagnostic rectangles; training mask distribution not approved','A/B1 only; B2 is absent because V3 pair review remains incomplete','Full valid native targets; no optimizer step or convergence claim'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ('status','optimizer_updates','device','patch_m','planning_seconds','forward_backward_seconds','peak_bytes')},indent=2));print('losses',result['losses_by_kind'])

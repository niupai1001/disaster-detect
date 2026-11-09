"""Explicit real-cache forward/backward probe, with zero optimizer updates."""
import argparse
import json
import time
from pathlib import Path
import torch
from .data import NativeOpticalDataset,to_device,sha256
from .model import Encoder,ReflectanceDecoder
from .tasks import spectral_task,spatial_task


def run(args):
    torch.manual_seed(0)
    data=NativeOpticalDataset(args.release,args.contracts,role='TRAIN')
    observation=to_device(data[args.index],args.device)
    model=Encoder(hyper_hidden=args.hyper_hidden,local_batch_size=args.local_batch_size,native_mode=args.native_mode,layout_mode=args.layout_mode).to(args.device).eval()
    decoder=ReflectanceDecoder(indexed=args.indexed).to(args.device)
    first=observation.bands[0];a,_,x,_,e,y=first.affine
    origin=(x,y+e*first.reflectance.shape[0])
    tasks=[spectral_task(observation,seed=0,update=0)]
    if args.rectangle is not None: tasks.append(spatial_task(observation,args.rectangle))
    rows=[]
    for task in tasks:
        model.zero_grad(set_to_none=True); decoder.zero_grad(set_to_none=True)
        if args.device.startswith('cuda'): torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
        start=time.perf_counter(); z=model(task.visible,patch_m=args.patch_m,origin_xy=origin)
        target=task.targets[0]; ids=torch.where(target.mask.flatten())[0][:args.target_pixels]
        if len(ids)==0: raise ValueError('No valid diagnostic target pixels')
        bounds=target.band.cells()[ids]
        condition=model.srf(target.band.wavelengths_nm,target.band.response)
        output,supported=decoder(z,bounds,condition,kind=task.kind,mask_bounds=None if task.rectangle is None else bounds.new_tensor(task.rectangle))
        if not bool(supported.any()): raise ValueError('No supported diagnostic target')
        truth=target.band.reflectance.flatten()[ids]; mse=(output[supported]-truth[supported]).square().mean();mse.backward()
        if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
        gradients=[p.grad for p in model.hyper.parameters()]
        rows.append(dict(kind=task.kind,visible_bands=len(task.visible),tokens=len(z.tokens),target_band=target.band.record_id,requested_pixels=len(ids),supported_pixels=int(supported.sum()),mse=float(mse.detach()),hyper_gradient_finite=all(g is not None and bool(torch.isfinite(g).all()) for g in gradients),seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None))
    report=dict(status='ENGINEERING_REAL_CACHE_PROBE_ONLY',optimizer_updates=0,observation_id=observation.identifier,role=observation.metadata['role'],source_cache_sha256=observation.metadata['cache_sha256'],source_identities=data.identities,band_shapes=[list(b.reflectance.shape) for b in observation.bands],patch_m=args.patch_m,hyper_hidden_candidate=args.hyper_hidden,local_execution_batch=args.local_batch_size,native_mode=args.native_mode,indexed_decoder=args.indexed,device=args.device,rows=rows,limitations=['One TRAIN observation, fixed diagnostic target prefix; not representative evaluation','Hypernetwork width is explicit candidate, not frozen','No V3 full-data admission or pair qualification','No optimizer updates; no claim of training or convergence'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--hyper-hidden',type=int,required=True);p.add_argument('--patch-m',type=float,required=True);p.add_argument('--local-batch-size',type=int,required=True);p.add_argument('--target-pixels',type=int,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--rectangle',type=float,nargs=4);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-mode',choices=['serial','grouped'],default='serial');p.add_argument('--indexed',action='store_true');p.add_argument('--layout-mode',choices=['reference','indexed'],default='reference')
    run(p.parse_args())

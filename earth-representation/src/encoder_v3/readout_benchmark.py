"""Matched real-data Decoder benchmark; frozen Encoder, zero training updates."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from .data import NativeOpticalDataset,to_device
from .model import Encoder,ReflectanceDecoder
from .tasks import spectral_task


def run(args):
    torch.manual_seed(0); dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN'); observation=to_device(dataset[args.index],args.device)
    task=spectral_task(observation,seed=0,update=0);first=observation.bands[0]
    origin=(first.affine[2],first.affine[5]+first.affine[4]*first.reflectance.shape[0])
    encoder=Encoder(hyper_hidden=args.hyper_hidden,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(args.device).eval()
    with torch.no_grad():
        representation=encoder(task.visible,patch_m=args.patch_m,origin_xy=origin)
        target=task.targets[0];ids=torch.where(target.mask.flatten())[0][:args.target_pixels];targets=target.band.cells()[ids]
        condition=encoder.srf(target.band.wavelengths_nm,target.band.response)
    dense=ReflectanceDecoder().to(args.device).eval(); sparse=ReflectanceDecoder(indexed=True).to(args.device).eval();sparse.load_state_dict(dense.state_dict())
    rows=[];reference=None;reference_grad=None
    for indexed,decoder in [(False,dense),(True,sparse)]:
        durations=[]
        for repeat in range(4):
            decoder.zero_grad(set_to_none=True)
            z=replace(representation,tokens=representation.tokens.detach().clone().requires_grad_())
            if args.device.startswith('cuda'):torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
            start=time.perf_counter();prediction,supported=decoder(z,targets,condition,kind='B1')
            prediction[supported].square().mean().backward()
            if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
            elapsed=time.perf_counter()-start
            if repeat:durations.append(elapsed)
        if reference is None:reference=prediction.detach();reference_grad=z.tokens.grad.detach()
        rows.append(dict(indexed=indexed,seconds=durations,mean_seconds=sum(durations)/len(durations),supported_pixels=int(supported.sum()),max_output_difference=float((prediction[supported]-reference[supported]).detach().abs().max()),relative_z_gradient_difference=float((z.tokens.grad-reference_grad).norm()/reference_grad.norm().clamp_min(1e-12)),peak_bytes=torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None))
    report=dict(optimizer_updates=0,data='one real TRAIN observation; B1 diagnostic prefix, not full evaluation',observation_id=observation.identifier,source_identities=dataset.identities,patch_m=args.patch_m,tokens=len(representation.tokens),target_pixels=len(ids),device=args.device,rows=rows,speedup=rows[0]['mean_seconds']/rows[1]['mean_seconds'],limitations=['No Encoder timing, optimizer, data loading or saving in readout timings','Tiny-sample performance evidence only','No formal precision/execution acceptance'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--hyper-hidden',type=int,required=True);p.add_argument('--patch-m',type=float,required=True);p.add_argument('--target-pixels',type=int,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

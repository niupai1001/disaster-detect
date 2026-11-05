"""Matched FP32/BF16 diagnostics, plus an explicit large BF16 case. No updates."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import torch
from .data import NativeOpticalDataset,to_device
from .model import Encoder,ReflectanceDecoder
from .tasks import spectral_task


def execute(model,decoder,observation,*,patch_m,device,bf16):
    task=spectral_task(observation,seed=0,update=0);first=observation.bands[0]
    origin=(first.affine[2],first.affine[5]+first.affine[4]*first.reflectance.shape[0])
    target=task.targets[0];ids=torch.where(target.mask.flatten())[0][:64];bounds=target.band.cells()[ids]
    model.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else nullcontext():
        z=model(task.visible,patch_m=patch_m,origin_xy=origin)
        condition=model.srf(target.band.wavelengths_nm,target.band.response)
        prediction,supported=decoder(z,bounds,condition,kind='B1')
        if not bool(supported.all()):raise ValueError('Diagnostic target has unsupported pixels')
        loss=(prediction.float()-target.band.reflectance.flatten()[ids].float()).square().mean()
    loss.backward();torch.cuda.synchronize(device)
    gradients=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return dict(precision='BF16 autocast, FP32 parameters/loss' if bf16 else 'FP32',tokens=len(z.tokens),loss=float(loss.detach()),seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(device),finite_gradients=bool(torch.isfinite(gradients).all())),prediction.detach().float().cpu(),gradients.detach().float().cpu()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--hyper-hidden',type=int,required=True);p.add_argument('--reference-index',type=int,required=True);p.add_argument('--large-index',type=int,required=True);p.add_argument('--patch-m',type=float,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.manual_seed(0);dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN')
    model=Encoder(hyper_hidden=args.hyper_hidden,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(args.device).eval();decoder=ReflectanceDecoder(indexed=True).to(args.device).eval()
    source=to_device(dataset[args.reference_index],args.device)
    fp32,p0,g0=execute(model,decoder,source,patch_m=args.patch_m,device=args.device,bf16=False)
    bf16,p1,g1=execute(model,decoder,source,patch_m=args.patch_m,device=args.device,bf16=True)
    p0,p1,g0,g1=(value.double() for value in (p0,p1,g0,g1))
    cosine=torch.dot(g0,g1)/(g0.norm()*g1.norm()).clamp_min(1e-24)
    report=dict(optimizer_updates=0,small_observation_id=source.identifier,matched=[fp32,bf16],relative_prediction_difference=float((p1-p0).norm()/p0.norm().clamp_min(1e-12)),relative_gradient_difference=float((g1-g0).norm()/g0.norm().clamp_min(1e-12)),gradient_cosine=float(cosine),comparison_accumulation='float64 on CPU',large_case=None,limitations=['Precision tolerances not yet confirmed; this is measurement, not acceptance','No optimizer-step or BF16 checkpoint recovery comparison yet','Real TRAIN diagnostic target prefix only'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print('MATCHED_PRECISION_COMPLETE',flush=True)
    large_source=to_device(dataset[args.large_index],args.device)
    large,_,_=execute(model,decoder,large_source,patch_m=args.patch_m,device=args.device,bf16=True)
    report['large_case']=dict(large,observation_id=large_source.identifier,patch_m=args.patch_m)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

"""Matched real full-target microbatch diagnostics; never steps an optimizer."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from .data import NativeOpticalDataset,sha256
from .diagnostic_batch import prepare_batch,device_views
from .model import Encoder,ReflectanceDecoder
from .joint import accumulate_joint


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('release','contracts','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--patch-m',type=float,required=True)
    parser.add_argument('--repeats',type=int,required=True)
    parser.add_argument('--compare-cell-index',action='store_true');args=parser.parse_args()
    if not args.device.startswith('cuda:') or not 1<=args.repeats<=3:raise ValueError('Explicit CUDA device and 1..3 bounded repeats required')
    report=dict(status='RUNNING',optimizer_updates=0,device=args.device,patch_m=args.patch_m,rows=[],limitations=['Intact eight-observation engineering cohort; not the complete V3 inventory or a pilot authorization','A/B1 only; no B2 physical pairing has passed review','Historical TRAIN-only statistics, not newly frozen V3 statistics','No optimizer-step, recovery, BF16 or formal numerical-tolerance acceptance','Timing excludes loading/planning/transfer; full experiment time remains unmeasured'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    def phase(name):report['phase']=name;save();print(name,flush=True)
    phase('initializing')
    torch.set_num_threads(2);torch.manual_seed(0)
    start=time.perf_counter();dataset=NativeOpticalDataset(args.release,args.contracts,role='TRAIN')
    observations,rectangles,plans,stds=prepare_batch(dataset,patch_m=args.patch_m)
    report['read_and_plan_seconds']=time.perf_counter()-start
    phase('transferring')
    start=time.perf_counter();views=device_views(observations,rectangles,plans,stds,args.device);torch.cuda.synchronize(args.device)
    report['transfer_seconds']=time.perf_counter()-start
    encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(args.device).eval()
    decoder=ReflectanceDecoder(indexed=True).to(args.device).eval()
    modules=[encoder,decoder];initial=[{k:v.detach().cpu().clone() for k,v in module.state_dict().items()} for module in modules]
    report.update(source_identities=dataset.identities,members=[o.identifier for o in observations],diagnostic_rectangles=rectangles,torch=torch.__version__,source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    baseline=None;baseline_grad=None
    try:
        configurations=[(1,False),(1,True)] if args.compare_cell_index else [(s,False) for s in (1,2,4,8)]
        for size,cell_indexed in configurations:
            decoder.cell_indexed=cell_indexed
            phase(f'microbatch_{size}_cell_index_{cell_indexed}_warmup')
            kwargs=dict(logical_update=0,chunk_pixels=512,precision='fp32',clip_norm=None,microbatch_observations=size)
            accumulate_joint(encoder,decoder,views,**kwargs);torch.cuda.synchronize(args.device)
            durations=[];peaks=[]
            for repeat in range(args.repeats):
                phase(f'microbatch_{size}_cell_index_{cell_indexed}_repeat_{repeat}')
                torch.cuda.reset_peak_memory_stats(args.device);start=time.perf_counter()
                result=accumulate_joint(encoder,decoder,views,**kwargs)
                torch.cuda.synchronize(args.device);durations.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated(args.device))
            gradient=torch.cat([p.grad.detach().flatten().cpu().double() for module in modules for p in module.parameters()])
            if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('Nonfinite diagnostic gradient')
            if baseline is None:baseline=result;baseline_grad=gradient
            counts={(row['kind'],row['observation'],band['band']):band['count'] for row in result['rows'] for band in row['result']['bands']}
            expected={(row['kind'],row['observation'],band['band']):band['count'] for row in baseline['rows'] for band in row['result']['bands']}
            if counts!=expected or result['global_tokens']!=baseline['global_tokens']:raise RuntimeError('Microbatch execution changed exposure/support/tokens')
            seconds=statistics.median(durations);pixels=sum(counts.values())
            report['rows'].append(dict(microbatch_observations=size,cell_indexed=cell_indexed,effective_batch=8,seconds=durations,median_seconds=seconds,valid_observations_per_second=8/seconds,supported_supervision_pixels_per_second=pixels/seconds,supported_supervision_pixels=pixels,peak_bytes=max(peaks),global_tokens=result['global_tokens'],global_padded_tokens=result['global_padded_tokens'],global_padding_fraction=result['global_padding_fraction'],losses_by_kind=result['losses_by_kind'],weighted_loss=result['weighted_loss'],absolute_weighted_loss_difference=abs(result['weighted_loss']-baseline['weighted_loss']),relative_gradient_difference=float((gradient-baseline_grad).norm()/baseline_grad.norm().clamp_min(1e-24)),max_gradient_difference=float((gradient-baseline_grad).abs().max()),gradient_cosine=float(torch.dot(gradient,baseline_grad)/(gradient.norm()*baseline_grad.norm()).clamp_min(1e-24)),speedup_vs_microbatch_1=report['rows'][0]['median_seconds']/seconds if report['rows'] else 1.))
            save();del gradient
        for module,original in zip(modules,initial):
            if any(not torch.equal(value.detach().cpu(),original[key]) for key,value in module.state_dict().items()):raise RuntimeError('Diagnostic changed parameters')
        report['status']='REAL_MICROBATCH_MEASUREMENTS_NOT_ACCEPTANCE';phase('completed')
    except Exception as exc:
        report['status']='FAILED';report['error']=type(exc).__name__+': '+str(exc);save();raise
    print(json.dumps(dict(status=report['status'],rows=report['rows']),indent=2))

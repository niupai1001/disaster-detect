"""Bounded synthetic high-resolution CUDA engineering probe; never steps parameters."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import torch
from .adaptive_window import AdaptiveWindowEncoder,adaptive_layout
from .model import BandObservation


def run(args):
    torch.manual_seed(123);torch.set_num_threads(4)
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.75,device)
    model=AdaptiveWindowEncoder(exponent=args.exponent,hyper_hidden=64,local_batch_size=64).to(device).train()
    report=dict(status='RUNNING',optimizer_updates=0,input_role='synthetic_engineering_only_not_data_admission',exponent=args.exponent,exponent_status='explicit_test_fixture_not_frozen_configuration',device=torch.cuda.get_device_name(device),device_uuid=str(torch.cuda.get_device_properties(device).uuid),torch_version=torch.__version__,dtype='FP32',allocated_memory_cap_fraction=.75,warmup_passes_per_case=1,timed_passes_per_case=2,cases=[],limitations=['Single visible band; not a multispectral observation or effective batch 8','Backward objective is latent squared mean; no A/B fit or optimizer step','60/120m are diagnostic field-of-view sizes, not Merge window sizes','CUDA event stage spans include launch gaps; not pure kernel throughput','No throughput acceptance or approved exponent inferred from this probe'])
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    args.output.parent.mkdir(parents=True,exist_ok=True);save()
    for fov in (60.,120.):
        for gsd in (.5,1.21):
            side=math.ceil(fov/gsd);start=time.perf_counter()
            b=BandObservation(torch.rand(side,side),torch.ones(side,side,dtype=torch.bool),torch.tensor([450.,500.,550.]),torch.tensor([0.,1.,0.]),(gsd,0.,100.,0.,-gsd,200.),'EPSG:32650','synthetic')
            origin=(100.,200.-side*gsd);layout=adaptive_layout(b,exponent=args.exponent,origin_xy=origin)
            layout_seconds=time.perf_counter()-start
            torch.cuda.synchronize();transfer_start=time.perf_counter()
            b=BandObservation(b.reflectance.pin_memory().to(device,non_blocking=True),b.valid_mask.pin_memory().to(device,non_blocking=True),b.wavelengths_nm.to(device),b.response.to(device),b.affine,b.crs,b.record_id)
            torch.cuda.synchronize();transfer_seconds=time.perf_counter()-transfer_start
            timed=[]
            for repetition in range(3):
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                stages=defaultdict(list);stacks=defaultdict(list);handles=[]
                def before(name):
                    def hook(module,inputs):
                        event=torch.cuda.Event(enable_timing=True);event.record();stacks[name].append(event)
                    return hook
                def after(name):
                    def hook(module,inputs,output):
                        event=torch.cuda.Event(enable_timing=True);event.record();stages[name].append((stacks[name].pop(),event))
                    return hook
                for name,module in [('srf',model.srf),('hypernetwork',model.hyper),('local_transformer',model.local),('global_transformer',model.global_net)]:
                    handles.extend((module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name))))
                start=time.perf_counter();r=model([b],origin_xy=origin);torch.cuda.synchronize();forward_seconds=time.perf_counter()-start
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();back_start=time.perf_counter()
                loss=r.tokens.square().mean();loss.backward();end.record();torch.cuda.synchronize()
                back_seconds=time.perf_counter()-back_start
                finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                for handle in handles:handle.remove()
                measurement=dict(forward_seconds=forward_seconds,backward_seconds=back_seconds,forward_backward_seconds=forward_seconds+back_seconds,backward_cuda_span_ms=begin.elapsed_time(end),forward_module_cuda_spans_ms={k:sum(a.elapsed_time(z) for a,z in v) for k,v in stages.items()},peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),finite_gradients=finite,tokens=len(r.tokens))
                if repetition:timed.append(measurement)
                del r,loss
            lengths=layout['lengths']
            case=dict(gsd_m=gsd,requested_fov_m=fov,actual_fov_m=side*gsd,native_shape=[side,side],native_pixels=side*side,window_pixels=layout['window_pixels'],window_m=layout['window_m'],local_sequence_without_cls=dict(min=min(lengths),max=max(lengths),mean=statistics.mean(lengths)),tokens=len(lengths),padding_ratio=0.,end_to_end_scope='synthetic creation + standalone geometry + transfer + encoder forward/backward; no real IO/Decoder/optimizer/checkpoint',setup_and_geometry_seconds=layout_seconds,transfer_seconds=transfer_seconds,timed=timed,median_forward_backward_seconds=statistics.median(x['forward_backward_seconds'] for x in timed))
            case['diagnostic_end_to_end_seconds']=layout_seconds+transfer_seconds+case['median_forward_backward_seconds']
            case['visible_pixels_per_forward_backward_second']=side*side/case['median_forward_backward_seconds']
            report['cases'].append(case);save();print(json.dumps(case),flush=True)
            del b,layout;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('adaptive_window.py'),Path(__file__).with_name('model.py'),Path(__file__).with_name('layout.py')]}
    report['status']='ENGINEERING_PROBE_COMPLETE';save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exponent',type=float,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())

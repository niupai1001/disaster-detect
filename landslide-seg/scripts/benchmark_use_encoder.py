"""Encoder-only same-device benchmark; reports scope explicitly."""
import argparse
import json
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from landslide_seg.models.use_v2 import USEv2
from landslide_seg.models.use_v3 import USEv3


def output_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(output_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)): return sum(output_bytes(v) for v in value)
    return 0


def measure(model,observation,warmup=30,repeats=100):
    device=observation['x'].device
    sync=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    with torch.inference_mode():
        for _ in range(warmup): model(observation)
        sync()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        timings=[]
        for _ in range(repeats):
            start=time.perf_counter(); result=model(observation); sync()
            timings.append((time.perf_counter()-start)*1000)
    median=float(np.median(timings))
    return dict(median_ms=median,p90_ms=float(np.percentile(timings,90)),throughput=1000*len(observation['x'])/median,
                peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                output_bytes=output_bytes(result),parameters=sum(p.numel() for p in model.parameters()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:1'); p.add_argument('--out',required=True)
    p.add_argument('--component',choices=['encoder','frozen','unet','segformer','training','direct10'],default='encoder')
    p.add_argument('--encoder',choices=['v2','v3'],default='v3')
    args=p.parse_args(); torch.set_num_threads(4)
    if args.component=='training': return real_update_cost(args.encoder,args.device,args.out)
    records=[]
    for c in ((10,) if args.component=='direct10' else (3,6,10,16)):
        for size in (128,256):
            for batch in (1,8):
                for dtype in (torch.float32,torch.bfloat16):
                    if dtype==torch.bfloat16 and (not args.device.startswith('cuda') or not torch.cuda.is_bf16_supported()): continue
                    for repeat in range(3):
                        torch.manual_seed(1234)
                        observation=dict(x=torch.randn(batch,c,size,size,device=args.device,dtype=torch.float32),
                                         pixel_valid=torch.ones(batch,c,size,size,device=args.device,dtype=torch.bool))
                        for name,cls in ((('direct10',USEv2),) if args.component=='direct10' else (('v2',USEv2),('v3',USEv3))):
                            model=cls().eval().to(device=args.device,dtype=torch.float32)
                            if args.component=='direct10':
                                from landslide_seg.models.use_downstream import build_visual
                                visual,_=build_visual('bridge10',str(Path(__file__).resolve().parents[1]/'weights/segformer/segformer-b0-finetuned-ade-512-512'))
                                class Direct(torch.nn.Module):
                                    def __init__(self,net): super().__init__(); self.net=net
                                    def forward(self,o): return self.net(o['x'])
                                model=Direct(visual)
                            if args.component=='frozen':
                                from landslide_seg.models.use_v3_reconstruction import FrozenUSE
                                model=FrozenUSE(model)
                            elif args.component in ('unet','segformer'):
                                from landslide_seg.models.use_v3_reconstruction import FullDecoder
                                model=FullDecoder(model,args.component,Path(__file__).resolve().parents[1]/'weights/segformer/segformer-b0-finetuned-ade-512-512')
                            model=model.eval().to(device=args.device,dtype=torch.float32)
                            if dtype==torch.bfloat16:
                                class Autocast(torch.nn.Module):
                                    def __init__(self,net): super().__init__(); self.net=net
                                    def forward(self,o):
                                        with torch.autocast(device_type='cuda',dtype=torch.bfloat16): return self.net(o)
                                model=Autocast(model)
                            try:
                                result=measure(model,observation)
                                result['theory']=theoretical_macs(model,observation)
                                result['profile']=profile_top10(model,observation)
                            except torch.cuda.OutOfMemoryError: result=dict(error='OOM')
                            records.append(dict(model=name,c=c,size=size,batch=batch,dtype=str(dtype),round=repeat,**result))
                            del model
                            if args.device.startswith('cuda'): torch.cuda.empty_cache()
                        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
                        Path(args.out).write_text(json.dumps(dict(scope=args.component+' forward; full update measured separately',records=records),indent=2))




def training_cost(model,observation,calibration,iterations=10):
    """Disposable FP32 three-view copy; 2 samples, four accumulated microbatches."""
    import copy
    import pretrain_use_v3 as pt
    trial=copy.deepcopy(model).train(); optimizer=torch.optim.AdamW(trial.parameters(),lr=.0006,weight_decay=.01)
    device=observation['x'].device
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    timings=[]
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True); forward=backward=0.
        for micro in range(4):
            sync(); t=time.perf_counter()
            raw,_=pt.views(trial,observation,1234+step*100+micro*3,(16,32,64)[step%3])
            loss=sum(v/calibration[k]*(.1 if k in ('gradient','consistency') else 1)/3.2 for k,v in raw.items())/4
            sync(); forward+=time.perf_counter()-t; t=time.perf_counter()
            loss.backward(); sync(); backward+=time.perf_counter()-t
        t=time.perf_counter(); torch.nn.utils.clip_grad_norm_(trial.parameters(),1.); optimizer.step(); sync()
        timings.append(dict(forward_seconds=forward,backward_seconds=backward,optimizer_seconds=time.perf_counter()-t))
    return dict(scope='three views; microbatch2 x accumulation4; disposable copy',records=timings)


def profile_top10(model,observation):
    activities=[torch.profiler.ProfilerActivity.CPU]
    if observation['x'].is_cuda: activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,with_flops=True) as prof:
        with torch.inference_mode(): model(observation)
    events=sorted(prof.key_averages(),key=lambda e:getattr(e,'device_time_total',0) or e.cpu_time_total,reverse=True)[:10]
    return dict(operators=[dict(name=e.key,cpu_us=e.cpu_time_total,device_us=getattr(e,'device_time_total',None),calls=e.count) for e in events],
                partial_flops=sum(e.flops for e in prof.key_averages()),
                coverage='PyTorch operator estimates only; attention/softmax/norm coverage incomplete; not total MAC/FLOPs')




def real_update_cost(kind,device,out,warmup=30,repeats=100,rounds=3):
    """Disposable original v2 or v3 update, including 6 S2 + 2 Landsat IO."""
    import pretrain_use_v3 as new
    import pretrain_use_v2 as old
    from run_use_v2 import read,write_json,gpu_guard
    splits,stats,_,_=new.inputs('multisensor')
    calibration=read(new.ROOT/('configs/use_v2/calibration.json' if kind=='v2' else 'outputs/use_v3/engineering/calibration.json'))
    all_records=[]
    with gpu_guard(device) as gate:
        for round_index in range(rounds):
            model=(old.initialize('attention',device) if kind=='v2' else new.initialize(device))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
            def sync():
                if device.startswith('cuda'): torch.cuda.synchronize(device)
            if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
            records=[]
            for step in range(1,warmup+repeats+1):
                sync(); begin=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                rows=new.sample_rows(splits['TRAIN'],step); io=forward=backward=0.
                for micro in range(4):
                    sync(); t=time.perf_counter()
                    current=rows[micro*2:micro*2+2]
                    original=new.load_rows(current,stats,device,1234+step*8+micro*2)
                    sync(); io+=time.perf_counter()-t; t=time.perf_counter()
                    if kind=='v3':
                        raw,_=new.views(model,original,1234+step*100+micro*3,(16,32,64)[(step-1)%3])
                        denom=calibration['coefficients'][current[0]['sensor']]
                        loss=sum(v/denom[k]*(.1 if k in ('gradient','consistency') else 1)/3.2 for k,v in raw.items())/4
                    else:
                        losses=[old.objective(model,original,task,1234+step*100+micro*3+i,
                            [calibration['coefficients'][r['sensor']][task] for r in current]) for i,task in enumerate(('spectral','fine','context'))]
                        loss=sum(r[0] for r in losses if r is not None)/12
                    sync(); forward+=time.perf_counter()-t; t=time.perf_counter()
                    loss.backward();sync(); backward+=time.perf_counter()-t
                t=time.perf_counter(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();sync()
                opt=time.perf_counter()-t; total=time.perf_counter()-begin
                if step>warmup: records.append(dict(io_ms=io*1000,forward_ms=forward*1000,backward_ms=backward*1000,optimizer_ms=opt*1000,total_ms=total*1000))
            metrics={key:dict(median=float(np.median([r[key] for r in records])),p90=float(np.percentile([r[key] for r in records],90))) for key in records[0]}
            all_records.append(dict(round=round_index,metrics=metrics,records=records,throughput=8000/metrics['total_ms']['median'],
                peak_bytes=torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None))
            write_json(out,dict(kind=kind,gate=gate,successful_cost_updates=(round_index+1)*(warmup+repeats),formal_updates=0,rounds=all_records))
            del model,optimizer
            if device.startswith('cuda'): torch.cuda.empty_cache()



def theoretical_macs(model,observation):
    """Count conv/linear and attention matmuls; exclude normalization/activation/IO."""
    from landslide_seg.models.use_v3 import SpatialBlock
    total=0; handles=[]
    def hook(module,args,output):
        nonlocal total
        if isinstance(module,torch.nn.MultiheadAttention):
            q,k,v=args[:3]; b,lq,d=q.shape; lk=k.shape[1]
            total+=b*(2*lq+2*lk)*d*d+2*b*lq*lk*d
        elif isinstance(module,torch.nn.Linear): total+=output.numel()*module.in_features
        elif isinstance(module,torch.nn.Conv2d): total+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,SpatialBlock):
            b,d,h,w=args[0].shape
            if module.global_attention: total+=2*b*(h*w)**2*d
            else:
                offset=2 if module.shifted else 0
                windows=b*((h+offset+3)//4)*((w+offset+3)//4)
                total+=2*windows*16*16*d
    for module in model.modules():
        if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.MultiheadAttention,SpatialBlock)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode(): model(observation)
    finally:
        for handle in handles: handle.remove()
    return dict(macs=total,flops_matmul_conv=2*total,
                included='convolution, linear projections, spectral/window/global attention QK and AV matmuls',
                excluded='normalization, softmax, nonlinearities, bias/addition, interpolation, masks, hashing, IO')

if __name__=='__main__':
    if '--component' in sys.argv and sys.argv[sys.argv.index('--component')+1]=='training': main()
    else:
        from run_use_v2 import gpu_guard
        device=sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'cuda:1'
        with gpu_guard(device): main()

"""Compare real-data FP32/BF16 loss, gradients and actual attention kernels."""
import argparse
import json
from pathlib import Path
import time
import torch
from .data import load_data, sample_from_dataset, build_model, ROOT
from .train import write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(17)
    ds,resolver=load_data();config=json.loads((ROOT/'configs/config_shared_mae_v1.json').read_text());config['patch_metres']=128.
    rows=sorted([(i,r) for i,r in enumerate(ds['TRAIN'].rows) if r['sensor']=='S2'],key=lambda pair:max(b['shape'][0]*b['shape'][1] for b in pair[1]['bands']))
    index,row=rows[len(rows)//2];sample=sample_from_dataset(ds['TRAIN'],index,resolver,'cuda:0')
    model=build_model(config).cuda();report=dict(id=row['id'],L=128.,scope='Numerical engineering test only; no chosen production L',cases=[])
    for kind in ('A','B1'):
        reference=None
        for precision in ('float32','bfloat16'):
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bfloat16'):
                    losses,_=model([sample],kind=kind,seed=17)
                losses.mean().backward()
            torch.cuda.synchronize();grads={n:p.grad.detach().float().clone() for n,p in model.named_parameters() if p.grad is not None}
            loss=float(losses.detach());kernels=[e.key for e in profile.key_averages() if 'attention' in e.key or 'flash' in e.key]
            case=dict(task=kind,precision=precision,loss=loss,seconds=time.perf_counter()-start,kernels=kernels)
            if reference is None:reference=(loss,grads)
            else:
                case['loss_relative_error']=abs(loss-reference[0])/max(abs(reference[0]),1e-8)
                denominator=sum(float(g.square().sum()) for g in reference[1].values())
                numerator=sum(float((grads[n]-g).square().sum()) for n,g in reference[1].items())
                case['gradient_relative_l2']=(numerator/max(denominator,1e-20))**.5
                case['numerical_pass']=case['loss_relative_error']<=.01 and case['gradient_relative_l2']<=.05
            report['cases'].append(case);write(args.output/'report.json',report)
    report['status']='PASS' if all(c.get('numerical_pass',True) for c in report['cases']) else 'FAIL'
    write(args.output/'report.json',report)


if __name__=='__main__':main()

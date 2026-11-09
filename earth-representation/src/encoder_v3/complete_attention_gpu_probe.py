"""Matched CUDA complete-attention component diagnostic; no parameter updates."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from torch import nn
from .complete_attention import query_chunked_transformer


def main():
    torch.set_num_threads(2);torch.manual_seed(41);torch.cuda.set_per_process_memory_fraction(.75)
    model=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',batch_first=True,norm_first=True),4,enable_nested_tensor=False).cuda().train()
    report=dict(status='RUNNING',scope='Synthetic complete-global-attention component, not full Encoder/Decoder or real supervision',optimizer_updates=0,device_uuid=str(torch.cuda.get_device_properties(0).uuid),dtype='FP32',matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,query_chunk_size=256,all_queries_and_keys_preserved=True,cases=[],limits=['No optimizer update or recovery comparison','Numerical deltas measured, no new acceptance tolerance frozen','Actual SDPA kernel backend not instrumented','Component timing excludes IO, geometry, Merge, Decoder, validation and checkpoint costs'])
    root=Path(__file__).resolve().parent;out=root/'reports/complete_attention_gpu_comparison.json'
    def save():out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    save()
    for n in (512,4096):
        data=torch.randn(n,128,device='cuda');target=torch.randn_like(data);measurements={};outputs={};gradients={}
        for name in ('reference','query_chunked'):
            times=[];peaks=[]
            for repetition in range(3):
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();x=data.detach().clone().requires_grad_();start=time.perf_counter()
                z=model(x[None])[0] if name=='reference' else query_chunked_transformer(model,x,query_chunk_size=256)
                loss=(z-target).square().mean();loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if repetition:times.append(elapsed);peaks.append(torch.cuda.max_memory_allocated())
                outputs[name]=z.detach().cpu();gradients[name]=torch.cat([x.grad.flatten()]+[p.grad.flatten() for p in model.parameters()]).detach().cpu()
                del x,z,loss
            measurements[name]=dict(seconds=times,median_seconds=statistics.median(times),peak_allocated_bytes=max(peaks),tokens_per_second=n/statistics.median(times))
        a,b=gradients['reference'].double(),gradients['query_chunked'].double()
        row=dict(tokens=n,measurements=measurements,prediction_max_abs=float((outputs['reference']-outputs['query_chunked']).abs().max()),gradient_relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),gradient_max_abs=float((a-b).abs().max()),speed_ratio=measurements['reference']['median_seconds']/measurements['query_chunked']['median_seconds'],default_adopted=False)
        report['cases'].append(row);save();print(json.dumps(row),flush=True)
        del data,target,outputs,gradients,a,b;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    report['status']='MATCHED_CUDA_MEASUREMENT_COMPLETE_NOT_FORMAL_ACCEPTANCE';report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'complete_attention.py')};save()


if __name__=='__main__':main()

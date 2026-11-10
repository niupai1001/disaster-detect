"""Paired complete-native batch forward/backward benchmark; no optimizer updates."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from .data import ROOT,load_data,build_model,EpochPlan,identity
from .train import batch_loss,write
from encoder_v3.latent_response.preflight import build_inputs
from encoder_v3.latent_response.data import make_view
from encoder_v3.latent_response.execution import view_loss
from encoder_v3.latent_response.checkpoint import parameter_sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh evidence directory required')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads((ROOT/'configs/config_shared_mae_v1.json').read_text());datasets,resolver=load_data()
    rows=EpochPlan(datasets['TRAIN'].rows,17,32).batch(0)
    if not any(r['sensor']=='EnMAP' for r in rows):rows[-1]=next(r for r in datasets['TRAIN'].rows if r['sensor']=='EnMAP')
    report=dict(status='RUNNING',identity=identity(config,datasets,resolver),members=[r['id'] for r in rows],
                batch_size=32,optimizer_updates=0,cases=[],scope='Paired native observations and effective Batch32; architecture/masks differ; no-update forward/backward throughput, not full Epoch timing')
    for kind in ('legacy','float32','bfloat16'):
        if kind=='legacy':_,old_ds,_,model,_=build_inputs()
        else:model=build_model(config)
        model.to(config['device']);before=parameter_sha(model)
        for repeat in range(2):
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            if kind=='legacy':
                ds=old_ds['TRAIN'];lookup={r['id']:i for i,r in enumerate(ds.rows)}
                for row in rows:
                    obs=ds[lookup[row['id']]];channels=ds.channels(obs,config['device'])
                    for task in ('A','B1'):
                        view=make_view(obs,channels,task,seed=17,update=0)
                        view_loss(model,view,backward=True,loss_weight=.5/len(rows),chunk_pixels=16384)
            else:
                batch_loss(model,datasets['TRAIN'],rows,resolver,dict(config,precision=kind),backward=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            report['cases'].append(dict(implementation=kind,repeat=repeat,seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated()))
            write(args.output/'report.json',report);print(kind,repeat,elapsed,flush=True)
        assert parameter_sha(model)==before
        del model;gc.collect();torch.cuda.empty_cache()
    warm={c['implementation']:c['seconds'] for c in report['cases'] if c['repeat']==1}
    report.update(status='PASS',fp32_speedup=warm['legacy']/warm['float32'],bf16_speedup=warm['legacy']/warm['bfloat16'],
                  bf16_vs_fp32=warm['float32']/warm['bfloat16'])
    write(args.output/'report.json',report)


if __name__=='__main__':main()

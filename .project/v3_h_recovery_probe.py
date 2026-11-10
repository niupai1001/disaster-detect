"""Synchronous, no-update replay of the failed native batch."""
import ssl
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','1')
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
sys.path.insert(0,str(ROOT/'src'))
import torch
from encoder_v3.hierarchical.train import load_data,build_model,sample_from_dataset,EpochPlan,write
from encoder_v3.latent_response.checkpoint import parameter_sha

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New evidence path required')
    torch.cuda.set_device(args.device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    release=ROOT/'outputs/encoder_v3/hierarchical/release_mae_accelerated_v1/execution_gpu1'
    config=json.loads((release/'config.json').read_text());config['device']=args.device
    model=build_model(config).to(args.device)
    payload=torch.load(release/'formal_run/last.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model']);before=parameter_sha(model)
    datasets,resolver=load_data();dataset=datasets['TRAIN'];plan=EpochPlan(dataset.rows,config['seed'],32)
    rows=plan.batch(208);lookup={row['id']:i for i,row in enumerate(dataset.rows)}
    report={'status':'RUNNING','cursor_replayed':209,'weights_cursor':payload['runner']['cursor'],'updates':0,'cases':[],'synchronous_cuda':True}
    try:
        for row in rows:
            sample=sample_from_dataset(dataset,lookup[row['id']],resolver,args.device)
            for kind in model.available_tasks(sample):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize()
                print('START',sample.identifier,kind,flush=True);start=time.perf_counter()
                loss,_=model([sample],kind=kind,seed=config['seed'],epoch=0)
                torch.cuda.synchronize();loss.mean().backward();torch.cuda.synchronize()
                assert bool(torch.isfinite(loss).all())
                report['cases'].append({'id':sample.identifier,'task':kind,'loss':float(loss.detach().mean()),'seconds':time.perf_counter()-start})
                write(args.output,report);del loss,sample
                sample=sample_from_dataset(dataset,lookup[row['id']],resolver,args.device)
        assert parameter_sha(model)==before
        report.update(status='PASS',parameters_unchanged=True)
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:write(args.output,report)

if __name__=='__main__':main()

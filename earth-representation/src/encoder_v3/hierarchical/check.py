"""Real native-data engineering evidence; never starts formal training."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from encoder_v3.latent_response.checkpoint import parameter_sha
from .train import load_data,sample_from_dataset,identity,build_model,write,batch_loss,save,resume,update


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['samples','capacity','batch','stability'],required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New evidence directory required')
    args.output.mkdir(parents=True)
    config=json.loads(args.config.read_text());torch.set_num_threads(2);torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datasets,resolver=load_data();dataset=datasets['TRAIN'];binding=identity(config,datasets,resolver)
    model=build_model(config).to(config['device'])
    report=dict(mode=args.mode,identity=binding,status='RUNNING',updates=0,cases=[],torch=torch.__version__,device=torch.cuda.get_device_name(config['device']))
    write(args.output/'report.json',report)
    def pixels(row):return sum(b['shape'][0]*b['shape'][1] for b in row['bands'])
    try:
        if args.mode in ('samples','capacity'):
            if args.mode=='capacity':rows=[max(dataset.rows,key=pixels)]
            else:
                rows=[]
                for sensor in sorted({r['sensor'] for r in dataset.rows}):
                    candidates=[r for r in dataset.rows if r['sensor']==sensor]
                    rows.append(sorted(candidates,key=pixels)[len(candidates)//2])
            lookup={r['id']:i for i,r in enumerate(dataset.rows)}
            before=parameter_sha(model)
            for row in rows:
                sample=sample_from_dataset(dataset,lookup[row['id']],resolver,config['device'])
                for kind in model.available_tasks(sample):
                    print('START',row['sensor'],row['id'],kind,pixels(row),flush=True)
                    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    model.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bfloat16'):
                        losses,details=model([sample],kind=kind,seed=config['seed'])
                    losses.mean().backward();torch.cuda.synchronize()
                    if not bool(torch.isfinite(losses).all()):raise ValueError('Nonfinite loss')
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1e9,error_if_nonfinite=True)
                    info=dict(id=row['id'],sensor=row['sensor'],task=kind,pixels=pixels(row),seconds=time.perf_counter()-start,
                        allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),gradient_norm=float(norm),
                        details={k:v for k,v in details[0].items() if k not in ('predictions','target_masks','online_features')})
                    report['cases'].append(info);write(args.output/'report.json',report);print('PASS',info,flush=True)
            assert parameter_sha(model)==before
            report['parameter_unchanged']=True
        else:
            from encoder_v3.shared_mae.data import EpochPlan
            plan=EpochPlan(dataset.rows,config['seed'],config['batch_size'])
            rows=plan.batch(0)
            if args.mode=='batch':
                start=time.perf_counter();before=parameter_sha(model)
                value,records=batch_loss(model,dataset,rows,resolver,config,backward=True)
                torch.cuda.synchronize()
                assert parameter_sha(model)==before
                report['cases'].append(dict(loss=value,seconds=time.perf_counter()-start,members=records,observations=len(rows),
                    allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
            else:
                # Fixed native cohort includes each admitted source; an engineering run, not an Epoch.
                cohort=[]
                for sensor in sorted({r['sensor'] for r in dataset.rows}):
                    cohort += sorted([r for r in dataset.rows if r['sensor']==sensor],key=pixels)[:2]
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
                scheduler=torch.optim.lr_scheduler.StepLR(optimizer,100,.5)
                state=dict(cursor=0);runtime=dict(config,batches_per_epoch=100)
                for step in range(6):
                    print('UPDATE',step+1,flush=True)
                    record=update(model,dataset,cohort,resolver,optimizer,scheduler,state,runtime)
                    report['updates']+=1;report['cases'].append(record);write(args.output/'report.json',report)
                save(args.output/'resume.pt',model,optimizer,scheduler,state,binding)
                update(model,dataset,cohort,resolver,optimizer,scheduler,state,runtime);report['updates']+=1
                expected=parameter_sha(model)
                resume(args.output/'resume.pt',model,optimizer,scheduler,state,binding)
                update(model,dataset,cohort,resolver,optimizer,scheduler,state,runtime);report['updates']+=1
                assert expected==parameter_sha(model)
                report['exact_next_update_restored']=True
        report.update(status='PASS',parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:write(args.output/'report.json',report)


if __name__=='__main__':main()

"""Short two-GPU numerical, throughput, capacity and recovery checks; isolated updates."""
import ssl
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','1')
from pathlib import Path
import sys,json,time,argparse,statistics
ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
sys.path.insert(0,str(ROOT/'src'))
import torch
from encoder_v3.hierarchical.train import load_data,build_model,identity,batch_loss,EpochPlan,update,save,resume,write
from encoder_v3.hierarchical.parallel import TwoDeviceBatch,partition_rows
from encoder_v3.hierarchical.verification import compare_snapshots,fingerprint
from encoder_v3.latent_response.checkpoint import snapshot

def gradients(model):
    return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New evidence directory required')
    args.output.mkdir(parents=True)
    torch.cuda.set_device(1);torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'outputs/encoder_v3/hierarchical/release_mae_accelerated_v1/execution_gpu1'
    config=json.loads((source/'config.json').read_text());config.update(devices=['cuda:1','cuda:0'],checkpoint_every_batches=1)
    write(args.output/'config.json',config)
    datasets,resolver=load_data();dataset=datasets['TRAIN'];binding=identity(config,datasets,resolver)
    payload=torch.load(source/'formal_run/last.pt',map_location='cpu',weights_only=False)
    serial=build_model(config).to(config['device']);model=build_model(config).to(config['device'])
    serial.load_state_dict(payload['model']);model.load_state_dict(payload['model'])
    worker=TwoDeviceBatch(model,dataset,resolver,config)
    report={'status':'RUNNING','identity':binding,'devices':[torch.cuda.get_device_name(i) for i in (1,0)],'config':config,
        'source_cursor':payload['runner']['cursor'],'synchronous_cuda':True,'formal_updates':0,'engineering_updates':0,'paired':[]}
    plan=EpochPlan(dataset.rows,config['seed'],32)
    try:
        for cursor in (200,208):
            cohort=plan.batch(cursor);cases=[]
            for repeat in range(2):
                serial.zero_grad(set_to_none=True);torch.cuda.synchronize(1);start=time.perf_counter()
                a,ar=batch_loss(serial,dataset,cohort,resolver,config,epoch=0,backward=True);torch.cuda.synchronize(1)
                serial_seconds=time.perf_counter()-start;expected=gradients(serial)
                start=time.perf_counter();b,br=worker.backward(cohort,0);seconds=time.perf_counter()-start
                received=gradients(model);assert expected.keys()==received.keys()
                assert abs(a-b)<2e-6
                error=sum(float((expected[n]-received[n]).square().sum()) for n in expected)
                scale=sum(float(v.square().sum()) for v in expected.values());relative=(error/max(scale,1e-30))**.5
                assert relative<3e-4,relative
                assert [r['id'] for r in ar]==[r['id'] for r in br]
                for x,y in zip(ar,br):
                    assert x['tasks'].keys()==y['tasks'].keys()
                    for kind in x['tasks']:
                        for key in ('tokens','targets','attention_edges','jepa_counts'):assert x['tasks'][kind][key]==y['tasks'][kind][key]
                cases.append({'repeat':repeat,'single_seconds':serial_seconds,'dual_seconds':seconds,'speedup':serial_seconds/seconds,
                    'loss_difference':abs(a-b),'gradient_relative_l2':relative,'observations':len(cohort)})
                print('PAIR',cursor+1,repeat,cases[-1],flush=True)
            report['paired'].append({'cursor':cursor+1,'cases':cases});write(args.output/'report.json',report)
        # Worst native members run simultaneously, retaining all tokens and targets.
        pixels=lambda r:sum(b['shape'][0]*b['shape'][1] for b in r['bands'])
        largest=sorted(dataset.rows,key=pixels,reverse=True)[:2]
        for device in worker.devices:torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();value,records=worker.backward(largest,0)
        report['capacity']={'status':'PASS','seconds':time.perf_counter()-start,'members':[r['id'] for r in largest],
            'pixels':[pixels(r) for r in largest],'assignment':[[r['id'] for r in part] for part in partition_rows(largest)],
            'peak_allocated':{d:torch.cuda.max_memory_allocated(d) for d in worker.devices},'loss':value,'records':records}
        write(args.output/'report.json',report);print('CAPACITY_PASS',flush=True)
        # Use a small four-source native panel for complete CUDA next-update recovery.
        cohort=[min([r for r in dataset.rows if r['sensor']==sensor],key=pixels) for sensor in sorted({r['sensor'] for r in dataset.rows})]
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
        optimizer.load_state_dict(payload['optimizer'])
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,100,.5);state={'cursor':200,'spent_updates':200}
        runtime=dict(config,batches_per_epoch=len(plan))
        update(model,dataset,cohort,resolver,optimizer,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
        save(args.output/'recovery.pt',model,optimizer,scheduler,state,binding)
        update(model,dataset,cohort,resolver,optimizer,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
        expected=snapshot(model,optimizer,state,scheduler)
        resume(args.output/'recovery.pt',model,optimizer,scheduler,state,binding)
        update(model,dataset,cohort,resolver,optimizer,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
        report['complete_next_update_restored']=compare_snapshots(expected,snapshot(model,optimizer,state,scheduler))
        report['replica_weights_exact']=fingerprint(model.state_dict())==fingerprint(worker.replica.state_dict())
        assert report['replica_weights_exact']
        timings=[case for pair in report['paired'] for case in pair['cases']]
        report['median_speedup']=statistics.median(c['speedup'] for c in timings)
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        worker.close();write(args.output/'report.json',report)
    print('DUAL_GATE_PASS',report['median_speedup'],flush=True)

if __name__=='__main__':main()

"""Isolated real-data throughput, capacity, recovery and learning gates."""
import ssl,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','0')
from pathlib import Path
import sys,json,time,argparse,statistics
ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
sys.path.insert(0,str(ROOT/'src'))
import torch
from encoder_v3.hierarchical.train import load_data,build_model,identity,sample_from_dataset,batch_loss,EpochPlan,update,validate,save,resume,write
from encoder_v3.hierarchical.parallel import TwoDeviceBatch
from encoder_v3.hierarchical.verification import compare_snapshots
from encoder_v3.latent_response.checkpoint import snapshot,parameter_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['pipeline','learning'],required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/config_v3_h_descriptor_free_v1.json').read_text())
    torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.set_device(config['device'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sets,resolver=load_data();ds=sets['TRAIN'];binding=identity(config,sets,resolver)
    model=build_model(config).to(config['device']);initial=parameter_sha(model)
    pixels=lambda r:sum(b['shape'][0]*b['shape'][1] for b in r['bands'])
    selected=[sorted([r for r in ds.rows if r['sensor']==s],key=pixels)[len([r for r in ds.rows if r['sensor']==s])//2] for s in sorted({r['sensor'] for r in ds.rows})]
    lookup={r['id']:i for i,r in enumerate(ds.rows)};plan=EpochPlan(ds.rows,17,32)
    report={'status':'RUNNING','identity':binding,'initial_parameter_sha256':initial,'formal_updates':0,'engineering_updates':0,
        'config':config,'mode':args.mode,'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'CUDA_LAUNCH_BLOCKING':os.environ['CUDA_LAUNCH_BLOCKING'],'cases':[]}
    worker=None
    def flush():write(args.output/'report.json',report)
    def sync():
        for d in config['devices']:torch.cuda.synchronize(d)
    def grads():return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    flush()
    try:
        if args.mode=='learning':
            samples=[sample_from_dataset(ds,lookup[r['id']],resolver,config['device']) for r in selected]
            def measure():
                result=[]
                with torch.no_grad():
                    for row,s in zip(selected,samples):
                        losses,details=model([s],kind='A',seed=17,return_predictions=True);detail=details[0]
                        groups={}
                        for i,c in enumerate(s.channels):
                            if bool(detail['target_masks'][i].any()):groups.setdefault(c.grid,[]).append(i)
                        indices=max(groups.values(),key=len);masks=torch.stack([detail['target_masks'][i] for i in indices]);common=masks.all(0)
                        if not bool(common.any()):
                            counts=masks.sum(0);point=int(counts.flatten().argmax())
                            indices=[i for i,mask in zip(indices,masks) if bool(mask.flatten()[point])]
                            common=torch.stack([detail['target_masks'][i] for i in indices]).all(0)
                        assert len(indices)>1 and bool(common.any()),'At least two QA-valid target bands on a common native pixel required'
                        pred=torch.stack([detail['predictions'][i][common] for i in indices]);truth=torch.stack([s.channels[i].values[common] for i in indices])
                        baselines=[float((c.values[mask]-c.values[keep].mean()).square().mean()) for c,mask,keep in zip(s.channels,detail['target_masks'],[c.valid&~m for c,m in zip(s.channels,detail['target_masks'])]) if bool(mask.any()) and bool(keep.any())]
                        result.append({'source':row['sensor'],'id':row['id'],'A_mse':float(losses[0]),'visible_mean_mse':statistics.mean(baselines),
                            'spectral_band_count':len(indices),'common_hidden_pixels':int(common.sum()),
                            'prediction_band_rms_z':float((pred-pred.mean(0)).square().mean().sqrt()),'truth_band_rms_z':float((truth-truth.mean(0)).square().mean().sqrt())})
                return result
            report['before']=measure();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
            scheduler=torch.optim.lr_scheduler.StepLR(opt,1000,.5);state={'cursor':0};runtime=dict(config,batches_per_epoch=1000);history=[]
            for step in range(64):
                start=time.perf_counter();record=update(model,ds,selected,resolver,opt,scheduler,state,runtime)
                history.append({'step':step+1,'loss':record['loss'],'seconds':time.perf_counter()-start});report['engineering_updates']+=1
                if step==0:
                    report['gradient_norms']={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None and (n.startswith('online.memory') or n.startswith('online.convolution'))}
                    assert all(v>0 for v in report['gradient_norms'].values())
                if (step+1)%16==0:report['history']=history;flush();print('FIXED_TRAIN',step+1,record['loss'],flush=True)
            report['after']=measure();a=statistics.mean(r['A_mse'] for r in report['before']);b=statistics.mean(r['A_mse'] for r in report['after'])
            assert b<a,(a,b)
            assert all(r['prediction_band_rms_z']>1e-4 for r in report['after'])
            report.update(initial_mean_A=a,final_mean_A=b,limits='Fixed four-source TRAIN cohort, unchanged epoch0 mask. Engineering weights are discarded; not formal VAL or convergence evidence.')
        else:
            # Two largest intact observations run concurrently, including all native targets.
            worker=TwoDeviceBatch(model,ds,resolver,config,validation_dataset=sets['VAL'])
            largest=sorted(ds.rows,key=pixels,reverse=True)[:2]
            sync();start=time.perf_counter();value,records=worker.backward(largest,0);sync()
            report['capacity']={'seconds':time.perf_counter()-start,'loss':value,'pixels':[pixels(r) for r in largest],
                'peak_allocated':{d:torch.cuda.max_memory_allocated(d) for d in config['devices']},'members':records}
            assert all(set(r['tasks'])=={'A'} for r in records);flush();print('DUAL_CAPACITY_PASS',report['capacity']['seconds'],flush=True)
            for network in (model,worker.replica):network._layouts.clear();network.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            cohort=plan.batch(0);times=[]
            for repeat in range(3):
                if repeat==0:
                    model.zero_grad(set_to_none=True);a,ar=batch_loss(model,ds,cohort,resolver,config,backward=True);expected=grads()
                sync();start=time.perf_counter();b,br=worker.backward(cohort,0);sync();times.append(time.perf_counter()-start)
                if repeat==0:
                    actual=grads();assert expected.keys()==actual.keys() and abs(a-b)<2e-6
                    error=sum(float((expected[n]-v).square().sum()) for n,v in actual.items());scale=sum(float(v.square().sum()) for v in expected.values())
                    relative=(error/max(scale,1e-30))**.5;assert relative<3e-4,relative
                    assert [r['id'] for r in ar]==[r['id'] for r in br]
                    report['single_dual']={'loss_difference':abs(a-b),'gradient_relative_l2':relative}
            report['dual_batch_seconds']=times;flush();print('NEW_DUAL_BATCH',times,flush=True)
            worker.close();worker=None
            old_config=dict(config);old_config.pop('frontend');old_config.pop('channel_keys');old_config.pop('max_updates')
            reference=build_model(old_config).to(config['device']);old_worker=TwoDeviceBatch(reference,ds,resolver,old_config)
            old_times=[]
            try:
                for _ in range(3):
                    sync();start=time.perf_counter();old_worker.backward(cohort,0);sync();old_times.append(time.perf_counter()-start)
            finally:old_worker.close()
            report['descriptor_dual_batch_seconds']=old_times;report['warm_batch_speedup']=statistics.median(old_times[1:])/statistics.median(times[1:])
            del reference,old_worker;torch.cuda.empty_cache();flush();print('PAIRED_SPEED',old_times,times,report['warm_batch_speedup'],flush=True)
            worker=TwoDeviceBatch(model,ds,resolver,config,validation_dataset=sets['VAL']) if not model._layouts else None
            # Replicas must be constructed before geometry caches exist.
            if worker is None:
                model._layouts.clear();worker=TwoDeviceBatch(model,ds,resolver,config,validation_dataset=sets['VAL'])
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
            scheduler=torch.optim.lr_scheduler.StepLR(opt,100,.5);state={'cursor':0};runtime=dict(config,batches_per_epoch=len(plan))
            update(model,ds,cohort,resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
            save(args.output/'recovery.pt',model,opt,scheduler,state,binding)
            update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1;expected=snapshot(model,opt,state,scheduler)
            resume(args.output/'recovery.pt',model,opt,scheduler,state,binding)
            update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
            report['complete_next_update_restored']=compare_snapshots(expected,snapshot(model,opt,state,scheduler))
            before=snapshot(model,opt,state,scheduler);sync();start=time.perf_counter();v=validate(model,sets['VAL'],resolver,opt,scheduler,state,runtime,parallel=worker);sync()
            report['dual_validation']={'seconds':time.perf_counter()-start,'metric':v['metric'],'observations':v['observations']}
            assert v['observations']==383 and all(set(r['tasks'])=={'A'} for r in v['members'])
            report['validation_state_restored']=compare_snapshots(before,snapshot(model,opt,state,scheduler));flush();print('RESTORE_VAL_PASS',report['dual_validation'],flush=True)
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    worker.backward(plan.batch(2),0);sync()
                (args.output/'operator_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=25),encoding='utf-8')
                report['operator_profile']='operator_profile.txt'
            except RuntimeError as exc:report['operator_profile_unavailable']=repr(exc)
            assert parameter_sha(model)!=initial  # Only isolated recovery updates above.
        report['status']='PASS'
    except BaseException as exc:report.update(status='FAILED',error=repr(exc));raise
    finally:
        if worker is not None:worker.close()
        flush()
    print('DESCRIPTOR_FREE_GATE_PASS',args.mode,flush=True)


if __name__=='__main__':main()

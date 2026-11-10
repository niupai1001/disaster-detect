"""Bounded A-only equivalence, speed, dual-validation and asynchronous recovery gate."""
import ssl
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING','0')
import sys,json,time,hashlib,argparse,importlib.util,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
sys.path.insert(0,str(ROOT/'src'))
import torch
from encoder_v3.hierarchical.train import load_data,build_model,identity,EpochPlan,batch_loss,sample_from_dataset,update,save,resume,validate,write
from encoder_v3.hierarchical.parallel import TwoDeviceBatch
from encoder_v3.hierarchical.verification import compare_snapshots,fingerprint
from encoder_v3.latent_response.checkpoint import snapshot,parameter_sha


def reference_class():
    folder=ROOT/'outputs/encoder_v3/hierarchical/release_mae_dual_v1/runtime/src/encoder_v3/hierarchical'
    name='encoder_v3.frozen_ab_reference'
    spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return importlib.import_module(name+'.model').HierarchicalMAE


def gradients(model):return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}


def edge_sha(layout):
    h=hashlib.sha256()
    for edges in layout[3]:
        for begin,end,index,*_ in edges:h.update(index.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['paired','pipeline'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New evidence directory required')
    args.output.mkdir(parents=True)
    config=json.loads((ROOT/'configs/config_v3_h_a_only_v1.json').read_text());torch.cuda.set_device(config['device']);torch.set_num_threads(2)
    torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datasets,resolver=load_data();ds=datasets['TRAIN'];binding=identity(config,datasets,resolver)
    model=build_model(config).to(config['device']);initial=parameter_sha(model)
    old=reference_class()(**{k:config[k] for k in ('patch_metres','mask_ratio','encoder_depth','lambda_jepa','ema_momentum')}).to(config['device']);old.load_state_dict(model.state_dict())
    result={'status':'RUNNING','mode':args.mode,'identity':binding,'config':config,'initialization':'Fresh seed17; reference copies candidate initialization; no historical checkpoint loaded',
        'initial_parameter_sha256':initial,'formal_updates':0,'engineering_updates':0,'CUDA_LAUNCH_BLOCKING':os.environ['CUDA_LAUNCH_BLOCKING'],'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'torch':torch.__version__,'cases':[]}
    write(args.output/'report.json',result);worker=None
    try:
        if args.mode=='paired':
            pixels=lambda r:sum(b['shape'][0]*b['shape'][1] for b in r['bands'])
            selected=[sorted([r for r in ds.rows if r['sensor']==s],key=pixels)[len([r for r in ds.rows if r['sensor']==s])//2] for s in sorted({r['sensor'] for r in ds.rows})]
            selected.append(max(ds.rows,key=pixels));lookup={r['id']:i for i,r in enumerate(ds.rows)}
            for row in selected:
                sample=sample_from_dataset(ds,lookup[row['id']],resolver,config['device']);case={'id':row['id'],'source':row['sensor'],'pixels':pixels(row),'backends':{}};expected=None
                for label,network in [('reference_A_only',old),('candidate_A_only',model)]:
                    network._layouts.clear();network.zero_grad(set_to_none=True);torch.cuda.empty_cache();timings=[]
                    for repeat in range(2):
                        network.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        losses,details=network([sample],kind='A',seed=config['seed']);losses.mean().backward();torch.cuda.synchronize()
                        timings.append({'repeat':repeat,'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated()})
                        if repeat==0:
                            layout=network.layout(sample)
                            with torch.no_grad():_,debug=network([sample],kind='A',seed=config['seed'],return_predictions=True)
                            measured={'loss':float(losses[0]),'gradients':gradients(network),'edges':edge_sha(layout),'tokens':details[0]['tokens'],'targets':details[0]['targets'],
                                'predictions':[p.cpu() for p in debug[0]['predictions']],'masks':[p.cpu() for p in debug[0]['target_masks']]}
                            if expected is None:expected=measured
                            else:
                                assert abs(expected['loss']-measured['loss'])<2e-6
                                assert expected['edges']==measured['edges'] and expected['tokens']==measured['tokens'] and expected['targets']==measured['targets']
                                for x,y in zip(expected['masks'],measured['masks']):assert torch.equal(x,y)
                                for x,y in zip(expected['predictions'],measured['predictions']):torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-4)
                                assert expected['gradients'].keys()==measured['gradients'].keys()
                                error=sum(float((expected['gradients'][n]-v).square().sum()) for n,v in measured['gradients'].items());norm=sum(float(v.square().sum()) for v in expected['gradients'].values())
                                relative=(error/max(norm,1e-30))**.5;assert relative<3e-4,relative
                                case.update(loss_difference=abs(expected['loss']-measured['loss']),gradient_relative_l2=relative,tokens=measured['tokens'],targets=measured['targets'],edges_exact=True)
                            del layout,debug,measured
                        del losses,details
                    case['backends'][label]=timings
                    network.zero_grad(set_to_none=True);network._layouts.clear();torch.cuda.empty_cache()
                case['cold_speedup']=case['backends']['reference_A_only'][0]['seconds']/case['backends']['candidate_A_only'][0]['seconds']
                case['warm_speedup']=case['backends']['reference_A_only'][1]['seconds']/case['backends']['candidate_A_only'][1]['seconds']
                result['cases'].append(case);write(args.output/'report.json',result);print('PAIR_A',case['source'],case['pixels'],case['cold_speedup'],case['warm_speedup'],flush=True)
            assert parameter_sha(model)==initial and parameter_sha(old)==initial
        else:
            del old;torch.cuda.empty_cache();worker=TwoDeviceBatch(model,ds,resolver,config,validation_dataset=datasets['VAL']);plan=EpochPlan(ds.rows,config['seed'],config['batch_size'])
            for cursor in (0,208):
                rows=plan.batch(cursor);timings={}
                for label,weights in [('AB_same_candidate',{'A':.7,'B1':.3}),('A_only_candidate',{'A':1.,'B1':0.})]:
                    worker.config=dict(config,task_weights=weights);torch.cuda.synchronize(1);torch.cuda.synchronize(0);start=time.perf_counter();value,records=worker.backward(rows,0)
                    timings[label]={'seconds':time.perf_counter()-start,'loss':value,'tasks':sorted({kind for m in records for kind in m['tasks']})}
                assert timings['A_only_candidate']['tasks']==['A']
                worker.config=config;result['cases'].append({'cursor':cursor+1,'timing':timings,'removing_B1_speedup':timings['AB_same_candidate']['seconds']/timings['A_only_candidate']['seconds']});write(args.output/'report.json',result)
                print('PIPELINE',result['cases'][-1],flush=True)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay']);scheduler=torch.optim.lr_scheduler.StepLR(opt,100,.5);state={'cursor':0};runtime=dict(config,batches_per_epoch=len(plan))
            # One full native update then exact next-update restoration, no formal state is reused.
            rows=plan.batch(0);update(model,ds,rows,resolver,opt,scheduler,state,runtime,parallel=worker);result['engineering_updates']+=1
            save(args.output/'recovery.pt',model,opt,scheduler,state,binding)
            update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);result['engineering_updates']+=1;expected=snapshot(model,opt,state,scheduler)
            resume(args.output/'recovery.pt',model,opt,scheduler,state,binding)
            update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);result['engineering_updates']+=1
            result['complete_next_update_restored']=compare_snapshots(expected,snapshot(model,opt,state,scheduler))
            validation_times={};before=snapshot(model,opt,state,scheduler)
            for label,parallel in [('single',None),('dual',worker)]:
                start=time.perf_counter();v=validate(model,datasets['VAL'],resolver,opt,scheduler,state,runtime,parallel=parallel)
                validation_times[label]={'seconds':time.perf_counter()-start,'metric':v['metric'],'observations':v['observations']}
                assert all(set(m['tasks'])=={'A'} for m in v['members'])
                assert all(compare_snapshots(before,snapshot(model,opt,state,scheduler)).values())
            assert abs(validation_times['single']['metric']-validation_times['dual']['metric'])<1e-6
            result['validation']=validation_times;write(args.output/'report.json',result);print('VALIDATION',validation_times,flush=True)
            # Short replay includes the old failure cohort and both extreme native scenes plus successors.
            replay=[0,1,2,3,4,*range(195,211),337,338,424,425]
            times=[]
            for cursor in replay:
                start=time.perf_counter();value,records=worker.backward(plan.batch(cursor),cursor//len(plan));elapsed=time.perf_counter()-start
                assert all(set(m['tasks'])=={'A'} for m in records)
                times.append({'cursor':cursor+1,'seconds':elapsed,'loss':value});result['replay']=times;write(args.output/'report.json',result);print('REPLAY_A',cursor+1,elapsed,flush=True)
            result['replay_median_batch_seconds']=statistics.median(t['seconds'] for t in times)
        result.update(status='PASS',parameter_state_after=fingerprint(model.state_dict()))
    except BaseException as exc:
        result.update(status='FAILED',error=repr(exc));raise
    finally:
        if worker is not None:worker.close()
        write(args.output/'report.json',result)
    print('A_ONLY_GATE_PASS',args.mode,result['CUDA_LAUNCH_BLOCKING'],flush=True)


if __name__=='__main__':main()

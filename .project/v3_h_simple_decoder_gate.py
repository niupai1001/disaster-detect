"""Isolated real-data gates for the native residual Decoder; never resumes a formal run."""
import ssl,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('CUDA_LAUNCH_BLOCKING','0')
from pathlib import Path
import sys,json,time,argparse,statistics,hashlib,importlib.util
import torch
ROOT=Path(__file__).resolve().parents[1]/'earth-representation';sys.path.insert(0,str(ROOT/'src'))
from encoder_v3.hierarchical.train import load_data,build_model,identity,sample_from_dataset,batch_loss,EpochPlan,update,validate,save,resume,write,warmup_updates,lr_multiplier
from encoder_v3.hierarchical.parallel import TwoDeviceBatch
from encoder_v3.hierarchical.verification import compare_snapshots,fingerprint
from encoder_v3.latent_response.checkpoint import snapshot,parameter_sha


def original_model(config):
    name='encoder_v3.hierarchical_simple_reference';folder=ROOT/'outputs/encoder_v3/hierarchical/release_no_enmap_v1/runtime/src/encoder_v3/hierarchical'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    model_class=__import__(name+'.model',fromlist=['HierarchicalMAE']).HierarchicalMAE
    return model_class(**{k:config[k] for k in ('patch_metres','mask_ratio','encoder_depth','lambda_jepa','ema_momentum')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['samples','precision','learning','pipeline'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=128)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/config_v3_h_simple_decoder_v1.json').read_text())
    torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.set_device(config['device']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=config.get('tf32_matmul',False);torch.backends.cudnn.allow_tf32=False
    sets,resolver=load_data(config);ds=sets['TRAIN'];plan=EpochPlan(ds.rows,17,32);binding=identity(config,sets,resolver)
    assert len(ds.rows)==7442 and len(plan)==233 and len(plan.batch(232))==18
    model=build_model(config).to(config['device']);initial=parameter_sha(model)
    pixels=lambda r:sum(b['shape'][0]*b['shape'][1] for b in r['bands'])
    cohort=[]
    for source in ('S2','L8','L9','EnMAP'):
        rows=sorted([r for r in ds.rows if r['sensor']==source],key=pixels);cohort.append(rows[len(rows)//2])
    report={'status':'RUNNING','mode':args.mode,'identity':binding,'initial_parameter_sha256':initial,'formal_updates':0,'engineering_updates':0,'cases':[],
        'architecture':'Original physical Encoder; fixed IDW8 visible raw residual baseline; 16-wide projected context; physical metadata+Memory band condition; four bilinear coefficients; no Decoder attention.',
        'limits':'New Decoder architecture, so total speed comparison is not mathematical model equivalence; Encoder compaction separately checked for equivalence. Training fit controls are not independent validation.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    worker=None
    def flush():write(args.output/'report.json',report)
    def sync():
        for d in config['devices']:torch.cuda.synchronize(d)
    flush()
    try:
        if args.mode=='precision':
            lookup={r['id']:i for i,r in enumerate(ds.rows)}
            with torch.no_grad():model.decoder.output.weight.normal_(std=.01)
            report['head_perturbation_std']=.01;report['gates']={'loss_abs':.0005,'loss_relative':.01,'gradient_relative_l2':.05}
            for row in cohort+[max(ds.rows,key=pixels)]:
                sample=sample_from_dataset(ds,lookup[row['id']],resolver,config['device']);case={'source':row['sensor'],'id':row['id'],'pixels':pixels(row),'backends':{}}
                first=None
                for label in ('float32','tf32'):
                    torch.backends.cuda.matmul.allow_tf32=label=='tf32'
                    model.zero_grad(set_to_none=True);sync();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=label=='bfloat16'):loss,_=model([sample],kind='A',seed=17)
                    loss.mean().backward();sync();elapsed=time.perf_counter()-start
                    grads={n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
                    assert all(torch.isfinite(v).all() for v in grads.values())
                    case['backends'][label]={'seconds':elapsed,'loss':float(loss.detach()),'peak_allocated':torch.cuda.max_memory_allocated()}
                    if first is None:first=(float(loss.detach()),grads)
                    else:
                        assert first[1].keys()==grads.keys()
                        relative=(sum(float((first[1][n]-v).square().sum()) for n,v in grads.items())/sum(float(v.square().sum()) for v in first[1].values()))**.5
                        case['gradient_relative_l2']=relative;case['loss_difference']=abs(float(loss.detach())-first[0])
                        assert relative<.05 and case['loss_difference']<max(.0005,.01*abs(first[0])),case
                    del loss
                report['cases'].append(case);model.zero_grad(set_to_none=True);model._layouts.clear();torch.cuda.empty_cache();flush();print('SIMPLE_PRECISION',case,flush=True)
        elif args.mode=='samples':
            lookup={r['id']:i for i,r in enumerate(ds.rows)}
            reference=original_model(config).to(config['device'])
            reference.online.load_state_dict(model.online.state_dict())
            for row in cohort+[max(ds.rows,key=pixels)]:
                sample=sample_from_dataset(ds,lookup[row['id']],resolver,config['device']);case={'source':row['sensor'],'id':row['id'],'pixels':pixels(row),'backends':{}}
                for label,network in [('original',reference),('native',model)]:
                    timings=[]
                    for repeat in range(2):
                        network.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
                        loss,detail=network([sample],kind='A',seed=17);loss.mean().backward();sync()
                        norm=torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],1e9,error_if_nonfinite=True)
                        timings.append({'seconds':time.perf_counter()-start,'loss':float(loss.detach()),'baseline':detail[0].get('baseline_mae'),'gradient_norm':float(norm),'peak_allocated':torch.cuda.max_memory_allocated()})
                        if label=='native':assert abs(float(loss.detach())-detail[0]['baseline_mae'])<1e-6
                        del loss,detail
                    case['backends'][label]=timings;network.zero_grad(set_to_none=True);network._layouts.clear();torch.cuda.empty_cache()
                case['warm_speedup']=case['backends']['original'][1]['seconds']/case['backends']['native'][1]['seconds']
                report['cases'].append(case);flush();print('SIMPLE_SAMPLE',row['sensor'],case['warm_speedup'],case['backends']['native'][1],flush=True)
            assert parameter_sha(model)==initial
        elif args.mode=='learning':
            # Four-source fixed TRAIN engineering fit. Keep true native targets and QA unchanged.
            lookup={r['id']:i for i,r in enumerate(ds.rows)};samples=[sample_from_dataset(ds,lookup[r['id']],resolver,config['device']) for r in cohort]
            # Include the previously displayed L9 case rather than selecting by a favorable error.
            for identifier in ('obs_8d69627e3f5637470adc7041','obs_9f912ff58940f9eb94093fd1'):
                if not any(r['id']==identifier for r in cohort):
                    selected=next(r for r in ds.rows if r['id']==identifier)
                    cohort.append(selected);samples.append(sample_from_dataset(ds,lookup[selected['id']],resolver,config['device']))
            initial_encoder={n:p.detach().cpu().clone() for n,p in model.online.named_parameters()};report['initial_encoder_fingerprint']=fingerprint(model.online.state_dict())
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0.)
            model.online.requires_grad_(False)
            report['cohort']=[{'id':r['id'],'source':r['sensor']} for r in cohort];report['learning_lr']=1e-3;report['initial_losses']=[];report['baseline_losses']=[]
            with torch.no_grad():
                for sample in samples:
                    loss,d=model([sample],kind='A',seed=17);report['initial_losses'].append(float(loss));report['baseline_losses'].append(d[0]['baseline_mae'])
            history=[]
            for step in range(args.steps):
                if step==args.steps//2:
                    model.online.requires_grad_(True)
                if step>=args.steps//2:
                    for group in opt.param_groups:group['lr']=config['lr']*min(1.,(step-args.steps//2+1)/50)
                opt.zero_grad(set_to_none=True);value=0.;start=time.perf_counter()
                for sample in samples:
                    loss,_=model([sample],kind='A',seed=17);(loss.mean()/len(samples)).backward();value+=float(loss.detach())/len(samples)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],3.,error_if_nonfinite=True);opt.step();report['engineering_updates']+=1
                history.append({'step':step+1,'loss':value,'seconds':time.perf_counter()-start})
                if step+1==args.steps//2:
                    assert fingerprint(model.online.state_dict())==report['initial_encoder_fingerprint']
                    assert all(p.grad is None for p in model.online.parameters())
                    with torch.no_grad():report['decoder_only_losses']=[float(model([s],kind='A',seed=17)[0]) for s in samples]
                    report['decoder_only_encoder_unchanged']=True
                if (step+1)%16==0:report['history']=history;flush();print('SIMPLE_FIT',step+1,value,flush=True)
            report['final_losses']=[]
            with torch.no_grad():
                for sample in samples:report['final_losses'].append(float(model([sample],kind='A',seed=17)[0]))
                index=next(i for i,r in enumerate(cohort) if r['sensor']=='L9' and r['id']=='obs_8d69627e3f5637470adc7041')
                sample=samples[index];layout=model.layout(sample);keeps,targets,_=model.masks(sample,layout,'A',17,0)
                encoded,valids,conditions=model.online(sample,layout,keeps);context=model.decoder.context(encoded,valids,layout)
                numeric=context.new_tensor([c.physical.encoded()[0] for c in sample.channels])
                from encoder_v3.hierarchical.simple_decoder import visible_condition
                bands=[]
                for i,c in enumerate(sample.channels):
                    bands.append(visible_condition(conditions['output'][i:i+1],numeric[i:i+1],c.values[None],keeps[i])[0][0])
                coefficients=model.decoder.coefficients(context,torch.stack(bands),torch.arange(len(context),device=context.device))
                report['L9_coefficient_case']={'id':cohort[index]['id'],'source':cohort[index]['sensor']}
                report['L9_learned_coefficient_band_span']=float((coefficients.max(0).values-coefficients.min(0).values).abs().max())
                assert report['L9_learned_coefficient_band_span']>1e-5
            report['learned_gain_over_baseline']=[b-v for b,v in zip(report['baseline_losses'],report['final_losses'])]
            report['quality_gates']={'all_cases_max_relative_regression':.01,'absolute_roundoff_allowance':1e-7,'nontrivial_cases_must_improve':['L8','EnMAP','textured S2'],'mean_must_improve':True}
            assert all(v<=b*1.01+1e-7 for v,b in zip(report['final_losses'],report['baseline_losses'])),report['learned_gain_over_baseline']
            assert sum(report['final_losses'])<sum(report['baseline_losses'])
            for i,r in enumerate(cohort):
                if r['sensor'] in ('L8','EnMAP') or r['id']=='obs_9f912ff58940f9eb94093fd1':assert report['final_losses'][i]<report['baseline_losses'][i]
            report['final_encoder_fingerprint']=fingerprint(model.online.state_dict())
            report['encoder_parameter_delta_l2']=sum(float((p.detach().cpu()-initial_encoder[n]).square().sum()) for n,p in model.online.named_parameters())**.5
            assert report['final_encoder_fingerprint']!=report['initial_encoder_fingerprint'] and report['encoder_parameter_delta_l2']>0
            torch.save({'model':model.state_dict(),'identity':binding,'engineering_only':True},args.output/'engineering.pt')
        else:
            worker=TwoDeviceBatch(model,ds,resolver,config,validation_dataset=sets['VAL'])
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:lr_multiplier(n,len(plan)*30,50));state={'cursor':0};runtime=dict(config,batches_per_epoch=len(plan))
            update(model,ds,plan.batch(0),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
            for rows in (plan.batch(1),plan.batch(232)):
                model.zero_grad(set_to_none=True);a,ar=batch_loss(model,ds,rows,resolver,config,backward=True)
                first={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
                b,br=worker.backward(rows,0);second={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
                assert first.keys()==second.keys() and abs(a-b)<2e-6
                relative=(sum(float((first[n]-v).square().sum()) for n,v in second.items())/max(sum(float(v.square().sum()) for v in first.values()),1e-30))**.5
                assert relative<3e-4
                report['cases'].append({'observations':len(rows),'loss_difference':abs(a-b),'gradient_relative_l2':relative});flush()
            largest=sorted(ds.rows,key=pixels,reverse=True)[:2];sync();start=time.perf_counter();loss,records=worker.backward(largest,0);sync()
            report['maximum_dual']={'seconds':time.perf_counter()-start,'pixels':[pixels(r) for r in largest],'loss':loss,'peak':{d:torch.cuda.max_memory_allocated(d) for d in config['devices']}};flush()
            save(args.output/'recovery.pt',model,opt,scheduler,state,binding)
            update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1;expected=snapshot(model,opt,state,scheduler)
            resume(args.output/'recovery.pt',model,opt,scheduler,state,binding)
            update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
            report['complete_recovery']=compare_snapshots(expected,snapshot(model,opt,state,scheduler));flush()
            before=snapshot(model,opt,state,scheduler);sync();start=time.perf_counter();v=validate(model,sets['VAL'],resolver,opt,scheduler,state,runtime,parallel=worker);sync()
            report['validation']={'metric':v['metric'],'baseline':v['baseline_metric'],'observations':v['observations'],'seconds':time.perf_counter()-start}
            report['validation_restores_state']=compare_snapshots(before,snapshot(model,opt,state,scheduler));write(args.output/'validation.json',v);flush()
            assert abs(v['baseline_metric']-.14418921053145567)<1e-6
            # Compare real Batch32 with the immutable former production model, warmed equally.
            speed_rows=plan.batch(2);reference=original_model(config).to(config['device'])
            reference.online.load_state_dict(model.online.state_dict())
            old_config=dict(config,residual_readout=False,decoder_type='hierarchical',tf32_matmul=False)
            old_worker=TwoDeviceBatch(reference,ds,resolver,old_config)
            speeds={}
            try:
                for label,runner,tf32 in (('original_fp32',old_worker,False),('native_tf32',worker,True)):
                    torch.backends.cuda.matmul.allow_tf32=tf32;times=[]
                    for _ in range(3):
                        sync();start=time.perf_counter();runner.backward(speed_rows,0);sync();times.append(time.perf_counter()-start)
                    speeds[label]=times
            finally:
                old_worker.close();reference._layouts.clear();old_worker.replica._layouts.clear();del reference,old_worker
                torch.backends.cuda.matmul.allow_tf32=config.get('tf32_matmul',False);torch.cuda.empty_cache()
            report['matched_batch32_seconds']=speeds;report['matched_warm_speedup']=statistics.median(speeds['original_fp32'][1:])/statistics.median(speeds['native_tf32'][1:]);flush()
            print('SIMPLE_MATCHED_BATCH32',speeds,report['matched_warm_speedup'],flush=True)
            replay=[]
            for cursor in (0,1,2,3,4,5,208,337,338,424,425):
                sync();start=time.perf_counter();value,records=worker.backward(plan.batch(cursor),cursor//len(plan));sync()
                replay.append({'cursor':cursor+1,'seconds':time.perf_counter()-start,'loss':value});report['replay']=replay;flush();print('SIMPLE_REPLAY',replay[-1],flush=True)
            report['replay_median_seconds']=statistics.median(r['seconds'] for r in replay)
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    batch_loss(model,ds,cohort,resolver,config,backward=True);sync()
                (args.output/'operator_profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30),encoding='utf-8')
                report['operator_profile']='operator_profile.txt'
            except RuntimeError as exc:report['operator_profile_unavailable']=repr(exc)
        report['status']='PASS'
    except BaseException as exc:report.update(status='FAILED',error=repr(exc));raise
    finally:
        if worker is not None:worker.close()
        flush()
    print('SIMPLE_DECODER_GATE_PASS',args.mode,flush=True)


if __name__=='__main__':main()

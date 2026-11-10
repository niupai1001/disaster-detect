"""Paired immutable-reference speed/correctness and full native no-update Epoch replay."""
import ssl
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import importlib.util
import sys
import time
import json
from pathlib import Path
import statistics
import hashlib
import torch

ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
sys.path.insert(0,str(ROOT/'src'))
from encoder_v3.hierarchical.train import load_data,sample_from_dataset,build_model,write,identity,batch_loss,EpochPlan
from encoder_v3.hierarchical.model import canonical_weights
from encoder_v3.hierarchical.verification import fingerprint,migrate_single_parent_checkpoint,gradient_contributions
from encoder_v3.latent_response.checkpoint import validation_transaction
from encoder_v3.hierarchical.train import save


def reference_class():
    path=ROOT/'outputs/encoder_v3/hierarchical/release_mae_v1/runtime/src/encoder_v3/hierarchical'
    name='encoder_v3.hierarchical_reference'
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return importlib.import_module(name+'.model').HierarchicalMAE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['paired','epoch'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--maximum',action='store_true')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--backend',choices=['candidate','baseline'],default='candidate')
    parser.add_argument('--device')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/config_v3_h_mae_v1.json')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New evidence file required')
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads(args.config.read_text())
    if args.device:config['device']=args.device
    torch.cuda.set_device(config['device'])
    datasets,resolver=load_data();dataset=datasets['TRAIN']
    baseline=reference_class()(**{k:config[k] for k in ('patch_metres','mask_ratio','encoder_depth','lambda_jepa','ema_momentum')}).to(config['device'])
    candidate=build_model(config).to(config['device'])
    checkpoint=ROOT/'outputs/encoder_v3/hierarchical/release_mae_v1/formal_run/last.pt'
    source_payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    weights=source_payload['model']
    baseline.load_state_dict(weights);candidate.load_state_dict(canonical_weights(weights))
    result=dict(mode=args.mode,config=config,identity=identity(config,datasets,resolver),updates=0,cases=[],status='RUNNING',checkpoint=str(checkpoint),torch=torch.__version__)
    result.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv)
    write(args.output,result)
    try:
        if args.mode=='paired':
            if args.maximum:
                chosen=[max(range(len(dataset.rows)),key=lambda i:sum(b['shape'][0]*b['shape'][1] for b in dataset.rows[i]['bands']))]
            elif args.count==4:
                chosen=[]
                for sensor in sorted({r['sensor'] for r in dataset.rows}):
                    indices=[i for i,r in enumerate(dataset.rows) if r['sensor']==sensor]
                    chosen.append(sorted(indices,key=lambda i:sum(b['shape'][0]*b['shape'][1] for b in dataset.rows[i]['bands']))[len(indices)//2])
            else:
                rows=EpochPlan(dataset.rows,config['seed'],config['batch_size']).batch(0)[:args.count]
                lookup={r['id']:i for i,r in enumerate(dataset.rows)};chosen=[lookup[r['id']] for r in rows]
            for i in chosen:
                sample=sample_from_dataset(dataset,i,resolver,config['device'])
                for kind in baseline.available_tasks(sample):
                    baseline._layouts.clear();candidate._layouts.clear();torch.cuda.empty_cache()
                    record=dict(id=sample.identifier,sensor=dataset.rows[i]['sensor'],task=kind,backends={})
                    comparisons={}
                    for label,model in [('baseline',baseline),('candidate',candidate)]:
                        timing=[]
                        for repeat in range(args.repeats):
                            print('START',label,record['sensor'],kind,repeat,flush=True)
                            model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                            start=time.perf_counter()
                            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bfloat16'):
                                loss,details=model([sample],kind=kind,seed=config['seed'])
                            loss.mean().backward();torch.cuda.synchronize()
                            timing.append(dict(seconds=time.perf_counter()-start,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
                            if repeat==0:
                                with torch.no_grad():
                                    _,debug=model([sample],kind=kind,seed=config['seed'],return_predictions=True)
                                comparisons[label]=dict(loss=loss.detach().cpu(),grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None},
                                    prediction=[p.detach().cpu() if p is not None else None for p in debug[0]['predictions']],
                                    encoded=[p.detach().cpu() for p in debug[0]['online_features']],
                                    masks=[m.cpu() for m in debug[0]['target_masks']],tokens=details[0]['tokens'],targets=details[0]['targets'],edges=details[0]['attention_edges'])
                                del debug
                            del loss,details
                        record['backends'][label]=timing
                        model.zero_grad(set_to_none=True);model._layouts.clear();torch.cuda.empty_cache()
                    a,b=comparisons['baseline'],comparisons['candidate']
                    a['grad']=canonical_weights(a['grad'])
                    torch.testing.assert_close(a['loss'],b['loss'],atol=2e-6,rtol=2e-5)
                    assert a['tokens']==b['tokens'] and a['targets']==b['targets'] and a['edges']==b['edges']
                    for x,y in zip(a['masks'],b['masks']):assert torch.equal(x,y)
                    prediction_error=max(float((x-y).abs().max()) for x,y in zip(a['prediction'],b['prediction']) if x is not None)
                    for x,y in zip(a['prediction'],b['prediction']):
                        if x is not None:torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-4)
                    for x,y in zip(a['encoded'],b['encoded']):torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-4)
                    diff=sum(float((a['grad'][n]-b['grad'].get(n,torch.zeros_like(v))).square().sum()) for n,v in a['grad'].items())
                    norm=sum(float(v.square().sum()) for v in a['grad'].values())
                    relative=(diff/max(norm,1e-30))**.5
                    assert relative<3e-4,relative
                    record.update(loss=float(b['loss'][0]),prediction_max_abs_error=prediction_error,gradient_relative_l2=relative,
                        tokens=b['tokens'],targets=b['targets'],attention_edges=b['edges'],
                        speedup=statistics.median(t['seconds'] for t in record['backends']['baseline'])/statistics.median(t['seconds'] for t in record['backends']['candidate']))
                    result['cases'].append(record);write(args.output,result);print('PASS',record['sensor'],kind,'speedup',record['speedup'],'grad',relative,flush=True)
        else:
            model=candidate if args.backend=='candidate' else baseline
            other=baseline if args.backend=='candidate' else candidate
            del other
            plan=EpochPlan(dataset.rows,config['seed'],config['batch_size'])
            replay_loss=batch_loss
            if args.backend=='baseline':
                replay_loss=importlib.import_module('encoder_v3.hierarchical_reference.train').batch_loss
                result['identity']=json.loads((ROOT/'outputs/encoder_v3/hierarchical/release_mae_v1/identity.json').read_text())
                result['execution_override']={'device':config['device'],'note':'Independent baseline replay on matching RTX5090; no DDP or formal updates'}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
            if args.backend=='baseline':optimizer.load_state_dict(source_payload['optimizer'])
            else:
                migrated=migrate_single_parent_checkpoint(source_payload,baseline,model,optimizer,result['identity'])
                optimizer.load_state_dict(migrated['optimizer'])
            original_model=fingerprint(model.state_dict());original_optimizer=fingerprint(optimizer.state_dict())
            replay_state={'cursor':0,'spent_updates':0,'status':'NO_UPDATE_REPLAY','performance_replay_only':True}
            replay_identity=dict(result['identity'],purpose='performance_no_update_replay')
            replay_checkpoint=args.output.with_suffix('.replay.pt')
            start=time.perf_counter();seen=[];validations=[]
            result.update(backend=args.backend,epoch_train_observations=len(dataset.rows),batches=len(plan))
            if args.backend=='candidate':
                first=plan.batch(0)[0];index=next(i for i,r in enumerate(dataset.rows) if r['id']==first['id'])
                sample=sample_from_dataset(dataset,index,resolver,config['device']);begin=time.perf_counter()
                result['gradient_diagnostics']=gradient_contributions(model,sample,config)
                result['gradient_diagnostics_seconds']=time.perf_counter()-begin
            for cursor in range(len(plan)):
                rows=plan.batch(cursor);model.zero_grad(set_to_none=True);begin=time.perf_counter()
                value,records=replay_loss(model,dataset,rows,resolver,config,backward=True,epoch=0)
                torch.cuda.synchronize();seen.extend(r['id'] for r in rows)
                case=dict(cursor=cursor+1,observations=len(rows),seconds=time.perf_counter()-begin,loss=value)
                result['cases'].append(case);replay_state['cursor']=cursor+1
                due=plan.validation_due(cursor+1,config['validations_per_epoch'])
                if due:
                    model.zero_grad(set_to_none=True);begin=time.perf_counter()
                    with validation_transaction(model,optimizer,replay_state),torch.no_grad():
                        model.eval()
                        val,records=replay_loss(model,datasets['VAL'],datasets['VAL'].rows,resolver,dict(config,seed=config['validation_seed']),epoch=0)
                    torch.cuda.synchronize()
                    validations.append(dict(cursor=cursor+1,seconds=time.perf_counter()-begin,loss=val,observations=len(records)))
                    result['validations']=validations
                if due or (cursor+1)%config['checkpoint_every_batches']==0:
                    begin=time.perf_counter();save(replay_checkpoint,model,optimizer,None,replay_state,replay_identity)
                    case['checkpoint_seconds']=time.perf_counter()-begin
                result['elapsed_seconds']=time.perf_counter()-start
                write(args.output,result);print('REPLAY',cursor+1,len(plan),result['elapsed_seconds'],flush=True)
            assert len(seen)==len(set(seen))==len(dataset.rows)
            result['members_once']=True
            assert len(validations)==config['validations_per_epoch']
            result['parameters_unchanged']=fingerprint(model.state_dict())==original_model
            result['optimizer_unchanged']=fingerprint(optimizer.state_dict())==original_optimizer
            assert result['parameters_unchanged'] and result['optimizer_unchanged']
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAILED',error=repr(exc));raise
    finally:write(args.output,result)


if __name__=='__main__':main()

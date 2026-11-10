"""Module attribution for every admitted source, A/B1, MAE/Hybrid, and temporary optimizer/EMA costs."""
import ssl
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import sys
import json
import time
import hashlib
from collections import defaultdict
import torch

ROOT=Path(__file__).resolve().parents[1]/'earth-representation'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT.parent/'.project'))
from encoder_v3.hierarchical.train import load_data,sample_from_dataset,build_model,identity,write
from encoder_v3.hierarchical.model import SparseAttention,ParentAttention,GlobalBlock,canonical_weights
from encoder_v3.hierarchical.verification import fingerprint,migrate_single_parent_checkpoint
from encoder_v3.latent_response.checkpoint import validation_transaction
from v3_h_speed import reference_class


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('New profile artifact required')
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads((ROOT/'configs/config_v3_h_mae_v1.json').read_text());torch.cuda.set_device(config['device'])
    datasets,resolver=load_data();dataset=datasets['TRAIN']
    source=ROOT/'outputs/encoder_v3/hierarchical/release_mae_v1/formal_run/last.pt'
    payload=torch.load(source,map_location='cpu',weights_only=False)
    report={'status':'RUNNING','optimizer_updates_persisted':0,'results':[],'optimizer':[],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes':['Module CUDA events are inclusive and instrumented; overlapping totals must not be summed.',
                 'Clean end-to-end speed comes from no-hook paired and full-cycle replays, not these event timings.',
                 'Same FP32/deterministic/TF32-disabled settings as production. Temporary optimizer/EMA calls are transactionally restored.']}
    try:
        for weight in (0.,.2):
            phase=dict(config,lambda_jepa=weight);model=build_model(phase).to(config['device']);model.load_state_dict(canonical_weights(payload['model']))
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
            migrated=migrate_single_parent_checkpoint(payload,reference_class()(**{k:phase[k] for k in ('patch_metres','mask_ratio','encoder_depth','lambda_jepa','ema_momentum')}),model,optimizer,identity(phase,datasets,resolver))
            optimizer.load_state_dict(migrated['optimizer']);original=fingerprint(model.state_dict())
            for sensor in sorted({r['sensor'] for r in dataset.rows}):
                indices=[i for i,r in enumerate(dataset.rows) if r['sensor']==sensor]
                index=sorted(indices,key=lambda i:sum(b['shape'][0]*b['shape'][1] for b in dataset.rows[i]['bands']))[len(indices)//2]
                begin=time.perf_counter();sample=sample_from_dataset(dataset,index,resolver,config['device']);torch.cuda.synchronize();load_seconds=time.perf_counter()-begin
                begin=time.perf_counter();layout=model.layout(sample);torch.cuda.synchronize();geometry_seconds=time.perf_counter()-begin
                for task in model.available_tasks(sample):
                    for repeat in range(args.repeats):
                        events=defaultdict(list);starts=defaultdict(list);hooks=[];originals=[]
                        def before(name):
                            def callback(*args):
                                event=torch.cuda.Event(enable_timing=True);event.record();starts[name].append(event)
                            return callback
                        def after(name):
                            def callback(*args):
                                end=torch.cuda.Event(enable_timing=True);end.record();events[name].append((starts[name].pop(),end))
                            return callback
                        for name,module in model.named_modules():
                            if isinstance(module,(SparseAttention,ParentAttention,GlobalBlock)) or name in ('online','target','online.memory','online.convolution','target.memory','target.convolution','output','predictor','predictor_context','predictor_position','predictor_output') or name.startswith('predictor_output.'):
                                hooks += [module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name),always_call=True)]
                                if isinstance(module,(SparseAttention,ParentAttention,GlobalBlock)) or name in ('output','predictor','predictor_context','predictor_position') or name.startswith('predictor_output.'):
                                    hooks += [module.register_full_backward_pre_hook(before(name+'.backward')),module.register_full_backward_hook(after(name+'.backward'))]
                        def instrument(obj,attribute,name):
                            original=getattr(obj,attribute)
                            def wrapped(*a,**kw):
                                before(name)()
                                try:return original(*a,**kw)
                                finally:after(name)()
                            originals.append((obj,attribute,original));setattr(obj,attribute,wrapped)
                        for label,encoder in [('online',model.online),('target',model.target)]:
                            instrument(encoder,'native',label+'.native_inclusive');instrument(encoder,'fuse',label+'.channel_fusion')
                        instrument(sys.modules[model.__class__.__module__],'position_features','position_encoding')
                        seen=set()
                        for maps in layout[2]:
                            for level,mapping in enumerate(maps):
                                if id(mapping) not in seen:
                                    instrument(mapping,'aggregate',('fine','middle','coarse')[level]+'.area_aggregation')
                                    instrument(mapping,'support',('fine','middle','coarse')[level]+'.QA_support');seen.add(id(mapping))
                                    instrument(mapping,'native',('fine','middle','coarse')[level]+'.native_readout_mapping')
                        model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                        resident=torch.cuda.memory_allocated();excluded=set();saved={}
                        def storage(tensor):return tensor.untyped_storage().data_ptr()
                        for p in model.parameters():excluded.add(storage(p))
                        def exclude(value):
                            if isinstance(value,torch.Tensor):excluded.add(storage(value))
                            elif isinstance(value,(list,tuple)):
                                for v in value:exclude(v)
                            elif hasattr(value,'source'):
                                for v in vars(value).values():exclude(v)
                        exclude((layout[2],layout[3],layout[4]))
                        for channel in sample.channels:exclude((channel.values,channel.valid))
                        def pack(tensor):
                            pointer=storage(tensor)
                            if tensor.device.type=='cuda' and pointer not in excluded:saved[pointer]=tensor.untyped_storage().nbytes()
                            return tensor
                        first=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
                        begin=time.perf_counter();first.record()
                        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
                            loss,details=model([sample],kind=task,seed=phase['seed'])
                        middle.record();loss.mean().backward();last.record();torch.cuda.synchronize()
                        elapsed=time.perf_counter()-begin
                        for h in hooks:h.remove()
                        for obj,attribute,original_method in originals:setattr(obj,attribute,original_method)
                        report['results'].append({'lambda_jepa':weight,'source':sensor,'task':task,'repeat':repeat,'sample':sample.identifier,
                            'load_seconds':load_seconds,'geometry_seconds':geometry_seconds,'instrumented_wall_seconds':elapsed,
                            'forward_cuda_ms':first.elapsed_time(middle),'backward_cuda_ms':middle.elapsed_time(last),
                            'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved(),
                            'saved_activation_storage_bytes_excluding_parameters_inputs_geometry':sum(saved.values()),
                            'peak_above_resident_bytes_including_temporaries_and_gradients':max(0,torch.cuda.max_memory_allocated()-resident),
                            'inclusive_cuda_ms':{name:sum(a.elapsed_time(b) for a,b in pairs) for name,pairs in events.items()},
                            'call_counts':{name:len(pairs) for name,pairs in events.items()},
                            'details':{k:v for k,v in details[0].items() if k not in ('predictions','target_masks','online_features')}})
                        write(args.output,report);print('PROFILE',weight,sensor,task,repeat,elapsed,flush=True)
            model.zero_grad(set_to_none=True);loss,_=model([sample],kind='A',seed=config['seed']);loss.mean().backward()
            before_model=fingerprint(model.state_dict());before_opt=fingerprint(optimizer.state_dict());runner={}
            with validation_transaction(model,optimizer,runner):
                for repeat in range(3):
                    a,b,c,d=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    a.record();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['clip_norm'],error_if_nonfinite=True);b.record()
                    optimizer.step();c.record()
                    if weight:model.update_ema()
                    d.record();torch.cuda.synchronize()
                    report['optimizer'].append({'lambda_jepa':weight,'repeat':repeat,'clip_cuda_ms':a.elapsed_time(b),'optimizer_cuda_ms':b.elapsed_time(c),'ema_cuda_ms':c.elapsed_time(d)})
            assert fingerprint(model.state_dict())==before_model and fingerprint(optimizer.state_dict())==before_opt
            assert fingerprint(model.state_dict())==original
            report.setdefault('identities',[]).append(identity(phase,datasets,resolver))
            model.zero_grad(set_to_none=True);model._layouts.clear();del model,optimizer,loss;torch.cuda.empty_cache()
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:write(args.output,report)


if __name__=='__main__':main()

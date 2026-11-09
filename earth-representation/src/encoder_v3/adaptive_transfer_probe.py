"""Matched before/after metadata transfer test. No optimizer or new model choices."""
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types
import torch
from .adaptive_window import AdaptiveWindowEncoder
from .model import BandObservation,ReflectanceDecoder


def main():
    root=Path(__file__).resolve().parent
    module=types.ModuleType('encoder_v3._adaptive_reference');module.__package__='encoder_v3';sys.modules[module.__name__]=module
    path=root/'reports/source_snapshots/adaptive_window.py.before_adaptive_window_packed'
    exec(compile(path.read_text(encoding='utf-8-sig'),str(path),'exec'),module.__dict__)
    torch.set_num_threads(4);torch.manual_seed(123);torch.cuda.set_per_process_memory_fraction(.75)
    reference=module.AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64,local_batch_size=64).cuda().train()
    candidate=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64,local_batch_size=64).cuda().train();candidate.load_state_dict(reference.state_dict())
    head=ReflectanceDecoder(indexed=True,cell_indexed=True).cuda().train()
    b=BandObservation(torch.rand(240,240,device='cuda'),torch.ones(240,240,dtype=torch.bool,device='cuda'),torch.tensor([450.,500.,550.],device='cuda'),torch.tensor([0.,1.,0.],device='cuda'),(.5,0.,100.,0.,-.5,200.),'EPSG:32650','synthetic')
    ids=torch.arange(0,240*240,113,device='cuda');queries=b.cells()[ids];truth=b.reflectance.flatten()[ids]
    saved={};history={'reference':[],'packed':[]};times={'reference':[],'packed':[]}
    def execute(name,model,measure):
        model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        r=model([b],origin_xy=(100.,80.));prediction,support=head(r,queries,model.srf(b.wavelengths_nm,b.response),kind='B1')
        assert bool(support.all());loss=(prediction-truth).square().mean();loss.backward();torch.cuda.synchronize();seconds=time.perf_counter()-start
        if measure:times[name].append(seconds)
        grads=torch.cat([p.grad.flatten() for p in list(model.parameters())+list(head.parameters())]).detach().cpu()
        state=dict(prediction=prediction.detach().cpu(),loss=float(loss.detach()),grad=grads,geometry=[torch.cat(getattr(r,key)).detach().cpu() for key in ('cells','coverage_cells','visible_cells','receptive_cells')],peak_bytes=torch.cuda.max_memory_allocated())
        saved[name]=state
        if measure:history[name].append(dict(prediction=state['prediction'],grad=state['grad']))
    execute('reference',reference,False);execute('packed',candidate,False)
    for order in [('reference','packed'),('packed','reference'),('reference','packed')]:
        for name in order:execute(name,reference if name=='reference' else candidate,True)
    a,c=saved['reference'],saved['packed']
    geometry_equal=all(torch.equal(x,y) for x,y in zip(a['geometry'],c['geometry'],strict=True))
    diff=(a['grad'].double()-c['grad'].double()).norm()/a['grad'].double().norm().clamp_min(1e-12)
    med={k:statistics.median(v) for k,v in times.items()}
    report=dict(status='MATCHED_MEASUREMENT_COMPLETE',optimizer_updates=0,input_role='synthetic_engineering_only',exponent_fixture=.5,exponent_frozen=False,field_of_view_m=120.,gsd_m=.5,tokens=2916,decoder_query_count=len(ids),decoder_scope='deterministic stride subset, not full training supervision',times_seconds=times,medians_seconds=med,speed_ratio=med['reference']/med['packed'],prediction_max_abs=float((a['prediction']-c['prediction']).abs().max()),loss_abs=abs(a['loss']-c['loss']),gradient_relative_l2=float(diff),geometry_bitwise_equal=geometry_equal,peak_allocated_bytes={k:saved[k]['peak_bytes'] for k in saved},limitations=['Single synthetic observation; not complete experiment throughput','No optimizer update or recovery validation','Synthetic Decoder target subset is solely a matched diagnostic, not reduced training supervision'])
    report['status']='MEASURED_NOT_FORMAL_ACCEPTANCE'
    report['same_path_repeat_gradient_relative_l2']={name:max(float((x['grad'].double()-values[0]['grad'].double()).norm()/values[0]['grad'].double().norm().clamp_min(1e-12)) for x in values) for name,values in history.items()}
    report['same_path_repeat_prediction_max_abs']={name:max(float((x['prediction']-values[0]['prediction']).abs().max()) for x in values) for name,values in history.items()}
    report['prior_attempt']='Exact-bitwise assertion failed; original failure log retained. This run records all deltas without changing tolerance.'
    (root/'reports/adaptive_transfer_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

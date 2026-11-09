"""Matched native tasks, gradients, batch throughput; zero optimizer updates."""
import copy
import json
import time
import torch

from .encoder_v2 import EncoderConfig,ObservationEncoderV2
from .encoder_v2_fast import FastObservationEncoderV2,fast_run_task
from .encoder_v2_tasks import spectral_task,run_task,balanced_loss,tensor_hash
from .g5_protocol import G5,ObservationStream,statistics
from .g5_train import batch_backward
from .check_encoder_v2 import state_hash
from .expansion_data import read,save


def gradients(model):return torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters() if p.grad is not None])


def main():
    torch.set_num_threads(4);config=read(G5/'config.json');torch.cuda.set_device(config['device'])
    checkpoint=torch.load(G5/'last.pt',map_location='cpu',weights_only=False);stream=ObservationStream(config['device'])
    models={key:cls(EncoderConfig(local_width=16,query_chunk=2048)).to(config['device'])
        for key,cls in [('original',ObservationEncoderV2),('fast',FastObservationEncoderV2)]}
    for model in models.values():model.load_state_dict(checkpoint['model']);model.train()
    initial=state_hash(models['original']);assert initial==state_hash(models['fast'])
    index=next(i for i in stream.indices if stream.train.rows[i]['sensor']=='S2' and stream.train.rows[i]['band_count']==10)
    row,image=stream.image(index);task=spectral_task(image,['B11','B12']);comparisons=[]
    for precision in ['fp32','bf16']:
        values={}
        for name,model in models.items():
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                rows,_,cases=(fast_run_task if name=='fast' else run_task)(model,task,row,statistics(),save_predictions=True,target_sample_pixels=1024,logical_update=1)
                loss=balanced_loss(rows)
            loss.backward();values[name]=dict(loss=float(loss.detach()),gradient=gradients(model),prediction=torch.cat([c['prediction'].float() for c in cases]))
        original,fast=values['original'],values['fast']
        relative_loss=abs(fast['loss']-original['loss'])/max(abs(original['loss']),1e-12)
        relative_grad=float((fast['gradient']-original['gradient']).norm()/original['gradient'].norm().clamp_min(1e-12))
        maximum_delta=float((fast['prediction']-original['prediction']).abs().max())
        assert relative_loss<.001 and relative_grad<.01,(precision,relative_loss,relative_grad)
        comparisons.append(dict(precision=precision,relative_loss=relative_loss,relative_gradient=relative_grad,max_prediction_delta=maximum_delta))
    rng=copy.deepcopy(stream.rng.bit_generator.state);bench={};reference_plans=None
    for name,model in models.items():
        from . import g5_train
        g5_train.run_task=fast_run_task if name=='fast' else run_task
        stream.rng.bit_generator.state=copy.deepcopy(rng)
        rows,exposure,timing,plans=batch_backward(model,stream,1,config,benchmark=True)
        grad=gradients(model)
        assert torch.isfinite(grad).all() and state_hash(model)==initial
        if reference_plans is None:reference_plans=plans;original_gradient=grad
        else:assert plans==reference_plans
        bench[name]=dict(timing=timing,exposure=exposure,model_hash_after=state_hash(model))
    full_relative_gradient=float((grad-original_gradient).norm()/original_gradient.norm().clamp_min(1e-12))
    assert full_relative_gradient<.01,full_relative_gradient
    # Largest admitted source uses the same native extent, masks and target count.
    largest=max(stream.indices,key=lambda i:sum(b['shape'][0]*b['shape'][1] for b in stream.train.rows[i]['bands']))
    meta,image=stream.image(largest);model=models['fast'];model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        rows,_,_=fast_run_task(model,spectral_task(image,['B11']),meta,statistics(),target_sample_pixels=2048,logical_update=1);loss=balanced_loss(rows)
    loss.backward();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
    assert torch.isfinite(gradients(model)).all() and peak<config['peak_memory_cap_bytes'] and state_hash(model)==initial
    save(G5/'acceleration_check.json',dict(status='PASS',optimizer_updates=0,checkpoint_update=checkpoint['state']['update'],
        comparisons=comparisons,batch32=bench,full_batch_relative_gradient=full_relative_gradient,
        measured_speedup=bench['original']['timing']['seconds']/bench['fast']['timing']['seconds'],
        largest_native_window=dict(member=meta['id'],peak_bytes=peak),model_hash=initial,
        limitations='One matched full batch on shared active GPU, after warmup comparisons. Real continuation speed still needs observation; equivalent algebra permits floating-point rounding differences.'))
    print('ACCELERATION_PASS',json.dumps(comparisons),bench['original']['timing']['seconds'],bench['fast']['timing']['seconds'],flush=True)


if __name__=='__main__':main()

"""One matched effective batch and largest native pair, no optimizer updates."""
import copy
import json
import time

import torch

from . import g5_train
from .encoder_v2 import EncoderConfig
from .encoder_v2_fast import fast_run_task
from .encoder_v2_fast_geometry import GeometryFastObservationEncoderV2
from .encoder_v2_microbatch import BatchedObservationEncoderV2,batch_backward,batched_tasks
from .encoder_v2_tasks import training_spatial_task,spectral_task,balanced_loss
from .g5_protocol import G5,ObservationStream,statistics
from .check_g5_acceleration import gradients
from .check_encoder_v2 import state_hash
from .expansion_data import read,save


def main():
    torch.set_num_threads(4);config=read(G5/'config.json');torch.cuda.set_device(config['device'])
    payload=torch.load(G5/'last.pt',map_location='cpu',weights_only=False)
    stream=ObservationStream(config['device']);rng=copy.deepcopy(payload['state']['sampler_rng']);update=payload['state']['update']+1
    model=BatchedObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(config['device']);model.load_state_dict(payload['model']);initial=state_hash(model)
    g5_train.run_task=fast_run_task;results={};reference=None;plans_ref=None;exposure_ref=None
    # The same parameters execute serial and batched paths, no optimizer is constructed.
    for micro in [1,2,4]:
        stream.rng.bit_generator.state=copy.deepcopy(rng)
        rows,exposure,timing,plans=(g5_train.batch_backward(model,stream,update,config,benchmark=True) if micro==1 else batch_backward(model,stream,update,config,benchmark=True,microbatch=micro))
        grad=gradients(model);assert torch.isfinite(grad).all() and state_hash(model)==initial
        loss=sum(float(r['loss']) for entries in rows.values() for r in entries)
        if reference is None:reference=grad;plans_ref=plans;exposure_ref=exposure;reference_loss=loss
        else:
            assert plans==plans_ref and exposure==exposure_ref
            relative=float((grad-reference).norm()/reference.norm().clamp_min(1e-12));relative_loss=abs(loss-reference_loss)/max(reference_loss,1e-12)
            assert relative<.01 and relative_loss<.001,(relative,relative_loss)
        results[str(micro)]=dict(timing=timing,exposure=exposure,relative_gradient=0 if micro==1 else relative,relative_loss=0 if micro==1 else relative_loss)
        model.zero_grad(set_to_none=True)
        print('MICROBATCH_RESULT',micro,json.dumps(results[str(micro)]),flush=True)
    # Four largest real windows, not a convenient crop; both A and B.
    largest=sorted(stream.indices,key=lambda i:sum(b['shape'][0]*b['shape'][1] for b in stream.train.rows[i]['bands']),reverse=True)[:4]
    sources=[stream.image(i) for i in largest];largest_checks=[]
    for kind in ['A','B']:
        jobs=[]
        for row,image in sources:
            hidden=[image.bands[-1].band_key]
            task=training_spatial_task(image,row['id'],update,excluded=hidden) if kind=='A' else spectral_task(image,hidden)
            jobs.append((row,task))
        model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=batched_tasks(model,jobs,statistics(),config['train_target_samples_per_band'],update)
            loss=sum(balanced_loss(rows) for rows,_,_ in out)
        loss.backward();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
        assert torch.isfinite(gradients(model)).all() and peak<config['peak_memory_cap_bytes']
        largest_checks.append(dict(task=kind,peak_bytes=peak,members=[row['id'] for row,_ in sources]))
        del out,loss
        print('LARGEST_PAIR_PASS',kind,peak,flush=True)
    assert state_hash(model)==initial
    report=dict(status='PASS',optimizer_updates=0,checkpoint_update=payload['state']['update'],effective_batch=32,microbatch=4,
        matched_batch=results,largest_native_group=largest_checks,model_hash=initial,speedup=results['1']['timing']['seconds']/results['4']['timing']['seconds'],
        caveat='One matched real batch; rounding changes under BF16, continued live speed must be measured; no member/target/task/budget change')
    save(G5/'acceleration/microbatch_check.json',report);print('MICROBATCH_PASS',report['speedup'],flush=True)


if __name__=='__main__':main()

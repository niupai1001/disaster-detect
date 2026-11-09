"""Resume the same G5 checkpoint with verified single-GPU execution optimizations."""
import json
import time
from pathlib import Path

from . import g5_train
from .encoder_v2_fast import fast_run_task
from .encoder_v2_fast_geometry import GeometryFastObservationEncoderV2
from .g5_protocol import G5
from .encoder_v2_runtime import LOG_ROOT
from .expansion_data import ROOT,read,save,digest


def main():
    manifest=read(G5/'acceleration/manifest.json')
    for relative,sha in manifest['sha256'].items():assert digest(ROOT/relative)==sha,relative
    assert read(G5/'acceleration_check.json')['status']=='PASS'
    assert read(G5/'geometry_acceleration_probe.json')['largest_native_A_B']['status']=='PASS'
    original_save=g5_train.save
    def save_retry(*args,**kwargs):
        # Windows readers can briefly prevent atomic replacement of status.json.
        for attempt in range(100):
            try:return original_save(*args,**kwargs)
            except PermissionError:
                if attempt==99:raise
                time.sleep(.05)
    g5_train.save=save_retry
    restored={}
    original_restore=g5_train.restore
    def restore(*args,**kwargs):
        payload=original_restore(*args,**kwargs)
        # Keep all already-written metric IDs unique if a partial step is replayed.
        path=LOG_ROOT/read(G5/'config.json')['run_id']/'metrics.jsonl'
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-65536));tail=handle.read().decode('utf-8',errors='ignore')
        for line in tail.splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            payload['viewer_sequence']=max(payload['viewer_sequence'],row.get('extensions',{}).get('writer_sequence',0))
        payload['state']['execution_revision']='single_gpu_fast_v2_geometry'
        payload['state']['execution_manifest_sha256']=digest(G5/'acceleration/manifest.json')
        restored.update(payload)
        return payload
    original_batch=g5_train.batch_backward
    def batch_backward(model,stream,update,config,run=None,**kwargs):
        state=restored['state'];step=state['update']
        history=[json.loads(line) for line in (G5/'evaluation_history.jsonl').read_text(encoding='utf-8').splitlines()]
        if step%config['validation_interval']==0 and history[-1]['update']<step:
            # last.pt preceded the interrupted validation; finish selection before the next update.
            score,_=g5_train.evaluation(model,stream,config,run,step,challenge=step%config['challenge_interval']==0)
            if score<state['best_score']:
                state['best_score']=score;state['early_stop']['best']=score
                g5_train.checkpoint(G5/'best.pt',model,optimizer_ref[0],state,run,config['device'])
            if score<state['significant_best']*(1-.001):
                state['significant_best']=score;state['lr_wait']=state['stop_wait']=0
            else:state['lr_wait']+=1;state['stop_wait']+=1
            if step>=2000 and state['lr_wait']>=4:
                state['lr_base']=max(config['minimum_lr'],state['lr_base']*.5);state['lr_wait']=0
            g5_train.checkpoint(G5/'last.pt',model,optimizer_ref[0],state,run,config['device'])
        g5_train.batch_backward=original_batch
        return original_batch(model,stream,update,config,run,**kwargs)
    optimizer_ref=[]
    restore_with_state=restore
    def restore_capture(path,model,optimizer,device):
        optimizer_ref.append(optimizer)
        return restore_with_state(path,model,optimizer,device)
    g5_train.restore=restore_capture;g5_train.batch_backward=batch_backward
    g5_train.ObservationEncoderV2=GeometryFastObservationEncoderV2;g5_train.run_task=fast_run_task
    for run_id in [read(G5/'config.json')['run_id'],'encoder_v2_G5_live_view']:
        path=LOG_ROOT/run_id/'run.json';meta=read(path)
        meta['config']['execution_revision']=dict(name='single_gpu_fast_v2_geometry',pool_rows=256,query_batch=8192,
            per_forward_SRF_weight_reuse=True,per_view_exact_geometry_neighbor_reuse=True,manifest_sha256=digest(G5/'acceleration/manifest.json'),
            model_parameters_and_training_contract_unchanged=True)
        temporary=path.with_suffix('.tmp');save(temporary,meta);temporary.replace(path)
    g5_train.launch(resume=True)


if __name__=='__main__':main()

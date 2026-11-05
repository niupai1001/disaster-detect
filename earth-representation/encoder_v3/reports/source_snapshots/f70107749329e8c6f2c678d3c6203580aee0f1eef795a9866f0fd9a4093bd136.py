"""Matched complete FP32 pipelines with/without geometry cache; no updates."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .data import NativeOpticalDataset,sha256
from .geometry_cache import NativeGeometryCache
from .gpu_update_check import difference
from .joint import accumulate_joint
from .model import Encoder,ReflectanceDecoder
from .pilot_protocol import load_study
from .plan_prefetch import PlanPrefetch,move_views
from .pilot_protocol import prepare_update
from .processes import launch_python


def run(args):
    if args.output.exists():raise ValueError('Probe report already exists')
    study=load_study(args.study);root=args.run_root.resolve();checkpoint=root/'fitting/AdamW/u000100.pt'
    if not checkpoint.exists():raise ValueError('Warmup checkpoint required')
    lock=root/'pilot_run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.fsync(fd)
    report=dict(status='RUNNING',optimizer_updates=0,study_sha256=study['candidate_sha256'],checkpoint_sha256=sha256(checkpoint),rows=[],comparisons=[],source_hashes={name:sha256(Path(__file__).with_name(name)) for name in ('model.py','layout.py','streaming.py','geometry_cache.py','geometry_cache_gpu_probe.py')},limitations=['Four fixed real schedule entries, one order after warmup; no optimizer/checkpoint/validation time','Prediction capture adds identical diagnostic overhead to both paths','Only compared inputs; not long-run cache stability, maximum-window or update/resume acceptance','No execution default changed'])
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(min(1.,12*1024**3/torch.cuda.get_device_properties(0).total_memory),0);torch.manual_seed(0)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['frozen_ids']['study']!=study['candidate_sha256']:raise ValueError('Different checkpoint study')
        encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').cuda().eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True).cuda().eval()
        encoder.load_state_dict(state['modules']['encoder']);decoder.load_state_dict(state['modules']['decoder']);del state
        dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
        if dataset.identities!=study['source_identities']:raise ValueError('Different data snapshot')
        config=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'));limits=json.loads((Path(__file__).parent/config['confirmed_preflight']['numerical_equivalence']['review']).read_text(encoding='utf-8'))
        if limits['review_type']!='explicit_user_design_confirmation':raise ValueError('Unreviewed comparison bounds')
        warm,_=prepare_update(study,dataset,0,'cpu',cell_indexed_geometry=True)
        accumulate_joint(encoder,decoder,move_views(warm,'cuda:0'),logical_update=0,chunk_pixels=512,precision='fp32',clip_norm=3);torch.cuda.synchronize();del warm
        cache=NativeGeometryCache(data_version=study['source_identities']);reference=[];started=time.perf_counter();save()
        for enabled in (False,True):
            rows=[];before=time.perf_counter();torch.cuda.reset_peak_memory_stats(0)
            with cache.installed() if enabled else nullcontext():
                stream=PlanPrefetch(lambda index:prepare_update(study,dataset,index,'cpu',cell_indexed_geometry=True),start=0,stop=4,identity=study['candidate_sha256'],prefetch=True);iterator=iter(stream)
                try:
                    for index in range(4):
                        begin=time.perf_counter();logical,(cpu_views,signature)=next(iterator);wait=time.perf_counter()-begin
                        predictions=[];supports=[]
                        def capture(module,inputs,result):predictions.append(result[0].detach().clone());supports.append(result[1].detach().clone())
                        hook=decoder.register_forward_hook(capture)
                        try:result=accumulate_joint(encoder,decoder,move_views(cpu_views,'cuda:0'),logical_update=index,chunk_pixels=512,precision='fp32',clip_norm=3)
                        finally:hook.remove()
                        torch.cuda.synchronize();step_seconds=time.perf_counter()-begin
                        support=torch.cat(supports).cpu();prediction=torch.cat(predictions).cpu()[support]
                        gradients=torch.cat([p.grad.detach().cpu().double().flatten() for module in (encoder,decoder) for p in module.parameters()])
                        counts=[[(b['band'],b.get('count'),b.get('unsupported')) for b in row['result']['bands']] for row in result['rows']]
                        sample=dict(signature=signature,prediction=prediction,gradients=gradients,support=support,counts=counts,result=result)
                        if not enabled:reference.append(sample)
                        else:
                            old=reference[index]
                            if old['signature']!=signature or old['counts']!=counts or not torch.equal(old['support'],support):raise ValueError('Cache changed task exposure/support')
                            prediction_difference=difference(old['prediction'],prediction);gradient_difference=difference(old['gradients'],gradients)
                            nmse=max(abs(a['nmse']-b['nmse']) for ra,rb in zip(old['result']['rows'],result['rows']) for a,b in zip(ra['result']['bands'],rb['result']['bands']) if a.get('nmse') is not None)
                            task_loss=max(abs(old['result'][key]-result[key]) for key in ('A_loss','B_loss'))
                            passed=prediction_difference['max_absolute']<=limits['prediction_max_absolute'] and gradient_difference['relative_l2']<=limits['gradient_relative_l2'] and nmse<=limits['per_band_nmse_max_absolute'] and task_loss<=limits['per_task_loss_max_absolute']
                            report['comparisons'].append(dict(index=index,predictions=prediction_difference,gradients=gradient_difference,max_band_nmse_difference=nmse,max_task_loss_difference=task_loss,passed=passed))
                            if not passed:raise ValueError('Geometry cache numerical bounds failed')
                        rows.append(dict(index=logical,signature=signature,data_wait_seconds=wait,forward_backward_with_transfer_seconds=step_seconds-wait,end_to_end_seconds=step_seconds,supported_pixels=sum(b['count'] for row in result['rows'] for b in row['result']['bands']),losses={key:result[key] for key in ('A_loss','B_loss','weighted_loss')}));stream.commit(logical)
                        print('cache',enabled,'input',index,'seconds',step_seconds,flush=True)
                        if time.perf_counter()-started>600:raise RuntimeError('Probe time bound reached')
                finally:iterator.close()
            seconds=time.perf_counter()-before
            report['rows'].append(dict(cache=enabled,seconds=seconds,steps=rows,observations_per_second=32/seconds,supervised_pixels_per_second=sum(r['supported_pixels'] for r in rows)/seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(0)));save()
        report.update(status='MATCHED_REAL_GPU_GEOMETRY_CACHE_MEASUREMENTS_PASS_NO_UPDATES',speedup=report['rows'][0]['seconds']/report['rows'][1]['seconds'],pipeline_step_speedup=sum(r['end_to_end_seconds'] for r in report['rows'][0]['steps'])/sum(r['end_to_end_seconds'] for r in report['rows'][1]['steps']),cache=cache.summary());save()
    except Exception as exc:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',error=type(exc).__name__+': '+str(exc));save();raise
    finally:os.close(fd);lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:run(args)
    else:
        record=args.output.with_suffix('.controller.json');log=args.output.with_suffix('.log')
        if record.exists() or args.output.exists():raise ValueError('Existing probe artifacts')
        with log.open('x',encoding='utf-8') as stream:
            child=launch_python(['-m','encoder_v3.geometry_cache_gpu_probe',*sys.argv[1:],'--worker'],stdout=stream);controller=dict(status='RUNNING',owned_worker_pid=child.pid,maximum_seconds=660)
            record.write_text(json.dumps(controller,indent=2),encoding='utf-8')
            try:code=child.wait(timeout=660)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                code=1
            controller.update(status='COMPLETED' if code==0 else 'FAILED_NO_AUTOMATIC_RETRY',worker_exit_code=child.returncode);record.write_text(json.dumps(controller,indent=2),encoding='utf-8');raise SystemExit(code)

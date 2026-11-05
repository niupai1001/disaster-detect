"""Bounded local/observation batching sweep at effective batch eight; no updates."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import torch
from .data import NativeOpticalDataset,sha256
from .geometry_cache import NativeGeometryCache
from .joint import accumulate_joint
from .model import Encoder,ReflectanceDecoder
from .pilot_protocol import load_study,prepare_update
from .plan_prefetch import move_views
from .processes import launch_python


def loss_map(result):
    return {(row['kind'],row['observation'],band['band']):(band.get('count'),band.get('unsupported'),band.get('nmse')) for row in result['rows'] for band in row['result']['bands']}


def run(args):
    if args.output.exists():raise ValueError('Existing measurements')
    root=args.run_root.resolve();lock=root/'pilot_run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.fsync(fd)
    report=dict(status='RUNNING',optimizer_updates=0,rows=[],limitations=['Same eight reviewed TRAIN observations; two fixed P60/P120 schedule entries','Timing includes cached Encoder geometry and complete forward/backward; excludes read/plan/transfer/update/save','Two repeats after per-configuration warmup; not full-experiment throughput','Loss/gradient checks rank candidates only; prediction/update/resume acceptance remains separate','Observation microbatches 2/4/8 are outside the earlier confirmed numerical-acceptance scope'])
    def save():args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        torch.set_num_threads(2);torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();cap=min(24*1024**3,max(1,free-2*1024**3));torch.cuda.set_per_process_memory_fraction(cap/total,0)
        report.update(allocated_memory_cap_bytes=cap,free_bytes_at_start=free,total_bytes=total,source_hashes={name:sha256(Path(__file__).with_name(name)) for name in ('model.py','layout.py','joint.py','geometry_cache.py','execution_batch_probe.py')});save()
        study=load_study(args.study);dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN');cache=NativeGeometryCache(data_version=study['source_identities'])
        if dataset.identities!=study['source_identities']:raise ValueError('Different data identities')
        checkpoint=root/'fitting/AdamW/u000100.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['frozen_ids']['study']!=study['candidate_sha256']:raise ValueError('Different checkpoint')
        encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').cuda().eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True).cuda().eval()
        encoder.load_state_dict(state['modules']['encoder']);decoder.load_state_dict(state['modules']['decoder']);del state
        report.update(study_sha256=study['candidate_sha256'],checkpoint_sha256=sha256(checkpoint));reference={};started=time.perf_counter()
        with cache.installed():
            prepared={index:prepare_update(study,dataset,index,'cpu',cell_indexed_geometry=True) for index in (0,1)}
            gpu={index:move_views(views,'cuda:0') for index,(views,_) in prepared.items()};torch.cuda.synchronize()
            def measure(local,micro):
                encoder.local_batch_size=local;aggregate=dict(local_batch_size=local,microbatch_observations=micro,effective_batch=8,cases=[])
                try:
                    for index,views in gpu.items():
                        kwargs=dict(logical_update=index,chunk_pixels=512,precision='fp32',clip_norm=3,microbatch_observations=micro)
                        accumulate_joint(encoder,decoder,views,**kwargs);torch.cuda.synchronize();durations=[];torch.cuda.reset_peak_memory_stats()
                        for _ in range(2):
                            start=time.perf_counter();result=accumulate_joint(encoder,decoder,views,**kwargs);torch.cuda.synchronize();durations.append(time.perf_counter()-start)
                        gradient=torch.cat([p.grad.detach().cpu().double().flatten() for module in (encoder,decoder) for p in module.parameters()]);losses=loss_map(result)
                        if index not in reference:reference[index]=(gradient,losses,result)
                        old_grad,old_losses,old_result=reference[index]
                        if losses.keys()!=old_losses.keys() or any(v[:2]!=old_losses[k][:2] for k,v in losses.items()) or result['global_tokens']!=old_result['global_tokens']:raise ValueError('Batching changed exposure/support/tokens')
                        grad_diff=float((gradient-old_grad).norm()/old_grad.norm().clamp_min(1e-24));nmse_diff=max(abs(v[2]-old_losses[k][2]) for k,v in losses.items() if v[2] is not None);task_diff=max(abs(result[k]-old_result[k]) for k in ('A_loss','B_loss'))
                        aggregate['cases'].append(dict(index=index,patch_m=study['schedule'][index]['patch_m'],input_signature=prepared[index][1],seconds=durations,median_seconds=statistics.median(durations),peak_allocated_bytes=torch.cuda.max_memory_allocated(),global_tokens=result['global_tokens'],padding_fraction=result['global_padding_fraction'],gradient_relative_l2=grad_diff,max_band_nmse_difference=nmse_diff,max_task_loss_difference=task_diff,meets_existing_numeric_values_as_diagnostic_only=grad_diff<=2e-6 and nmse_diff<=1e-5 and task_diff<=1e-6))
                        print('local',local,'micro',micro,'input',index,'seconds',durations,flush=True)
                        if time.perf_counter()-started>1800:raise RuntimeError('Diagnostic time bound reached')
                    aggregate.update(status='MEASURED_NOT_UPDATE_ACCEPTANCE',combined_median_seconds=sum(c['median_seconds'] for c in aggregate['cases']))
                except torch.OutOfMemoryError as exc:
                    aggregate.update(status='ALLOCATOR_CAP_REACHED',error=str(exc));encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                report['rows'].append(aggregate);save();return aggregate
            for local in (64,128,256,512):measure(local,1)
            eligible=[r for r in report['rows'] if r['status']=='MEASURED_NOT_UPDATE_ACCEPTANCE' and all(c['meets_existing_numeric_values_as_diagnostic_only'] for c in r['cases'])]
            if not eligible:raise ValueError('No numerically matching local execution candidate')
            best=min(eligible,key=lambda r:r['combined_median_seconds'])['local_batch_size'];report['local_candidate_for_further_checks']=best;save()
            for micro in (2,4,8):measure(best,micro)
        report.update(status='EXECUTION_BATCH_CANDIDATE_MEASUREMENTS_COMPLETE',cache=cache.summary(),seconds=time.perf_counter()-started);save()
    except Exception as exc:report.update(status='FAILED_NO_AUTOMATIC_RETRY',error=type(exc).__name__+': '+str(exc));save();raise
    finally:os.close(fd);lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:run(args)
    else:
        record=args.output.with_suffix('.controller.json');log=args.output.with_suffix('.log')
        if record.exists() or args.output.exists():raise ValueError('Existing probe artifacts')
        with log.open('x',encoding='utf-8') as stream:
            child=launch_python(['-m','encoder_v3.execution_batch_probe',*sys.argv[1:],'--worker'],stdout=stream);controller=dict(status='RUNNING',owned_worker_pid=child.pid,maximum_seconds=1860);record.write_text(json.dumps(controller,indent=2),encoding='utf-8')
            try:code=child.wait(timeout=1860)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                code=1
            controller.update(status='COMPLETED' if code==0 else 'FAILED_NO_AUTOMATIC_RETRY',worker_exit_code=child.returncode);record.write_text(json.dumps(controller,indent=2),encoding='utf-8');raise SystemExit(code)

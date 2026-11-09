"""Matched, resumable optimizer fitting inside the remaining original pilot budget."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import torch
from .budget import PilotBudget
from .data import NativeOpticalDataset,sha256
from .equivalence import require_cell_equivalence
from .joint import accumulate_joint
from .model import Encoder,ReflectanceDecoder
from .pilot_protocol import load_study,require_update_review,prepare_update,prepare_validation,optimizer_groups,LinearWarmup,canonical_sha
from .plan_prefetch import PlanPrefetch,move_views
from .streaming import stream_task
from .training import CAdamW,save_checkpoint,restore_checkpoint
from .processes import launch_python
from .cuda_probe import probe_cuda


def validation_means(rows):
    kinds=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in rows:
        if row['result']['loss'] is not None:kinds[row['panel']][row['source']][row['observation']].append(row['result']['loss'])
    return {kind:sum(sum(sum(values)/len(values) for values in observations.values())/len(observations) for observations in sources.values())/len(sources) for kind,sources in kinds.items()}


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())


def reconcile_pending(budget,fit,frozen):
    _,reservations,completed=budget.read()
    for token in set(reservations)-completed:
        reservation=reservations[token];step=reservation['logical_update']+1
        path=Path(fit)/reservation['candidate']/f'u{step:06d}.pt'
        if not path.exists():raise ValueError('Uncertain update lacks a matching durable checkpoint; no blind replay')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['frozen_ids']!=frozen or state['logical_update']!=step or state['sampler']['next_update']!=step or state['sampler']['reservation']!=token:raise ValueError('Pending reservation/checkpoint mismatch')
        budget.complete(token)


def run(args,study,config,equivalence):
    root=args.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/'pilot_run.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.fsync(descriptor)
    started=time.perf_counter();device='cuda:0';checkpoint_bytes_hint=64*1024**2
    try:
        torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        budget=PilotBudget(root/'budget.json',study_sha256=study['candidate_sha256'])
        fit=root/'fitting';fit.mkdir(exist_ok=True)
        core=('model.py','layout.py','spatial_index.py','streaming.py','tasks.py','training.py','joint.py','data.py','pilot_protocol.py','plan_prefetch.py','pilot_fit.py','equivalence.py','budget.py','processes.py','cuda_probe.py')
        frozen=dict(study=study['candidate_sha256'],source_identities=study['source_identities'],optimizer_conditions=config['candidate'],device=device,precision='fp32',effective_batch=8,execution=dict(microbatch_observations=1,local_batch_size=64,chunk_pixels=512,native_mode='grouped',layout_mode='indexed',cell_indexed=True,cell_indexed_support_count=True,prefetch_geometry=args.prefetch),equivalence=equivalence,code={name:sha256(Path(__file__).parent/name) for name in core})
        identity=fit/'identity.json'
        if identity.exists():
            if json.loads(identity.read_text(encoding='utf-8'))!=frozen:raise ValueError('Fitting source/code/config changed; no silent resume')
        else:write_new(identity,frozen)
        run_record=fit/('run_'+uuid.uuid4().hex[:12]+'.json')
        report=dict(status='RUNNING',through_update=args.through_update,optimizer_updates_before=budget.summary(),resource_limits=dict(wall_seconds=14400,max_allocated_bytes=12*1024**3,max_output_bytes=8*1024**3),optimizers={},formal_training_started=False)
        def persist():run_record.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        def limits(extra_storage=0):
            if time.perf_counter()-started>14400:raise RuntimeError('Fitting wall-time cap reached')
            if torch.cuda.max_memory_allocated(device)>12*1024**3:raise RuntimeError('Fitting allocated-memory cap reached')
            if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())+extra_storage>8*1024**3:raise RuntimeError('Fitting storage cap reached')
        persist();reconcile_pending(budget,fit,frozen)
        train=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN');val=NativeOpticalDataset(study['release'],study['contracts'],role='VAL')
        before=time.perf_counter();panels=prepare_validation(study,val,cell_indexed_geometry=True);report['validation_geometry_prepare_seconds']=time.perf_counter()-before;persist()
        parameters=config['candidate']
        for name in ('AdamW','CAdamW'):
            directory=fit/name;directory.mkdir(exist_ok=True)
            torch.manual_seed(study['seed'])
            encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(device).eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True).to(device).eval()
            modules={'encoder':encoder,'decoder':decoder}
            optimizer=(torch.optim.AdamW if name=='AdamW' else CAdamW)(optimizer_groups(modules.values(),parameters['weight_decay']),lr=parameters['peak_lr'],betas=tuple(parameters['betas']),eps=parameters['eps'])
            scheduler=LinearWarmup(optimizer,peak=parameters['peak_lr'],warmup=parameters['warmup_updates'])
            checkpoints=sorted(directory.glob('u[0-9][0-9][0-9][0-9][0-9][0-9].pt'))
            next_update=0;last=None
            if checkpoints:
                last=restore_checkpoint(checkpoints[-1],modules=modules,optimizer=optimizer,scheduler=scheduler,frozen_ids=frozen)
                next_update=last['logical_update']
                if last['sampler']['next_update']!=next_update:raise ValueError('Checkpoint/replay progress differs')
            if args.through_update<next_update:raise ValueError('Fitting endpoint cannot roll progress backward')
            if args.through_update>next_update+budget.summary()['remaining_by_candidate'][name]:raise ValueError('Requested fitting endpoint exceeds original remaining budget')
            expected_inputs=[canonical_sha(dict(optimizer_conditions=parameters,entry=entry)) for entry in study['schedule'][:args.through_update]]
            report['optimizers'][name]=dict(start_update=next_update,next_update=next_update,matched_schedule_sha256=canonical_sha(expected_inputs));persist()
            def validate(point):
                path=directory/f'validation_u{point:06d}.json'
                if path.exists():
                    existing=json.loads(path.read_text(encoding='utf-8'))
                    if existing['frozen_identity_sha256']!=sha256(identity) or existing['checkpoint_sha256']!=sha256(directory/f'u{point:06d}.pt'):raise ValueError('Validation identity differs')
                    return
                rows=[];start=time.perf_counter()
                for label,patch,cpu_view in panels:
                    view=move_views([cpu_view],device)[0]
                    result=stream_task(encoder,decoder,view.task,view.plan,train_stds=view.train_stds,chunk_pixels=512,loss_weight=1.,backward=False,precision='fp32')
                    rows.append(dict(panel=label,patch_m=patch,observation=view.task.metadata['id'],source=view.loss_source,result=result));limits()
                torch.cuda.synchronize(device)
                write_new(path,dict(status='FIXED_VALIDATION_ENDPOINT',optimizer=name,logical_updates=point,frozen_identity_sha256=sha256(identity),checkpoint_sha256=sha256(directory/f'u{point:06d}.pt'),panel_sha256=canonical_sha(study['validation_panels']),rows=rows,means_by_panel=validation_means(rows),B2=None,seconds=time.perf_counter()-start,limitations=['TRAIN-only historical scaling; not formal V3 statistics','Baseline/segment endpoint diagnostics; does not change the formal 250-update scheduler interval','No B2, unseen-combination or frozen-representation evaluation']))
            try:
                if not checkpoints:
                    initial=directory/'u000000.pt';save_checkpoint(initial,modules=modules,optimizer=optimizer,scheduler=scheduler,logical_update=0,sampler={'next_update':0,'reservation':None},frozen_ids=frozen)
                if next_update==0:validate(0)
                stream=PlanPrefetch(lambda index:prepare_update(study,train,index,'cpu',cell_indexed_geometry=True),start=next_update,stop=args.through_update,identity=study['candidate_sha256'],prefetch=args.prefetch)
                iterator=iter(stream)
                try:
                    while stream.next_update<args.through_update:
                        iteration_start=time.perf_counter();wait_start=time.perf_counter();index,(cpu_views,input_signature)=next(iterator);wait=time.perf_counter()-wait_start
                        transfer_start=time.perf_counter();views=move_views(cpu_views,device);torch.cuda.synchronize(device);transfer=time.perf_counter()-transfer_start
                        scheduler.apply(index);compute_start=time.perf_counter()
                        result=accumulate_joint(encoder,decoder,views,logical_update=index,chunk_pixels=512,precision='fp32',clip_norm=parameters['clip_norm']);torch.cuda.synchronize(device);compute=time.perf_counter()-compute_start
                        limits(checkpoint_bytes_hint);path=directory/f'u{index+1:06d}.pt'
                        if path.exists():raise ValueError('Refusing to overwrite a checkpoint')
                        token=budget.reserve(name,index);optimizer_start=time.perf_counter();optimizer.step();torch.cuda.synchronize(device);optimizer_seconds=time.perf_counter()-optimizer_start
                        stream.commit(index)
                        metrics=dict(logical_update=index,optimizer_updates=1,result=result,input_signature=input_signature,data_wait_seconds=wait,transfer_seconds=transfer,forward_backward_seconds=compute,optimizer_seconds=optimizer_seconds,lr=[g['lr'] for g in optimizer.param_groups],reservation=token)
                        save_start=time.perf_counter();save_checkpoint(path,modules=modules,optimizer=optimizer,scheduler=scheduler,logical_update=index+1,sampler=dict(stream.state_dict(),reservation=token,last_metrics=metrics),frozen_ids=frozen);budget.complete(token)
                        metrics.update(checkpoint_sha256=sha256(path),checkpoint_seconds=time.perf_counter()-save_start,end_to_end_seconds=time.perf_counter()-iteration_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                        write_new(directory/f'metrics_u{index+1:06d}.json',metrics)
                        report['optimizers'][name].update(next_update=index+1,last_metrics=metrics);persist();limits()
                        print(name,index+1,'/',args.through_update,'A',result['A_loss'],'B',result['B_loss'],flush=True)
                        del views,cpu_views
                finally:iterator.close()
                validate(stream.next_update)
                report['optimizers'][name]['status']='SEGMENT_COMPLETE';persist()
            except Exception as exc:
                report.update(status='FAILED_WITH_BUDGET_PRESERVED',error=type(exc).__name__+': '+str(exc),budget=budget.summary());persist();raise
            del encoder,decoder,optimizer,scheduler,modules,iterator,stream;gc.collect()
        report.update(status='MATCHED_PILOT_SEGMENT_COMPLETE_NOT_FORMAL_TRAINING',seconds=time.perf_counter()-started,budget=budget.summary());persist()
        return run_record
    finally:os.close(descriptor);lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--through-update',type=int,required=True);parser.add_argument('--prefetch',action='store_true');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--check-only',action='store_true');mode.add_argument('--run',action='store_true');args=parser.parse_args()
    if not 1<=args.through_update<=196:raise ValueError('Fitting endpoint must remain inside the remaining 196-update ceiling')
    source=Path(__file__).parent;study=load_study(args.study);config=json.loads((source/'config.json').read_text(encoding='utf-8'))
    require_update_review(config,study,source);equivalence=require_cell_equivalence(config,source,study)
    if not args.worker and not probe_cuda()['available']:
        print('FITTING_BLOCKED_CUDA_RUNTIME_NO_NEW_UPDATES',json.dumps(probe_cuda()));raise SystemExit(3)
    if args.check_only:print('FITTING_GATES_PASS_NO_UPDATES');raise SystemExit(0)
    if not args.worker:
        args.run_root.mkdir(parents=True,exist_ok=True);record=args.run_root/('fit_controller_'+uuid.uuid4().hex[:12]+'.json')
        log=record.with_suffix('.log')
        with log.open('x',encoding='utf-8') as output:
            child=launch_python(['-m','encoder_v3.pilot_fit',*sys.argv[1:],'--worker'],stdout=output)
            controller=dict(status='RUNNING',owned_worker_pid=child.pid,worker_log=str(log.resolve()),through_update=args.through_update,maximum_wall_seconds=14400);record.write_text(json.dumps(controller,indent=2),encoding='utf-8')
            try:code=child.wait(timeout=14400)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                code=1;controller['status']='OWN_WORKER_DEADLINE_TERMINATED_NO_AUTOMATIC_REPLAY'
        controller.update(status=controller['status'] if code else 'COMPLETED',worker_exit_code=child.returncode)
        if code and controller['status']=='RUNNING':controller['status']='FAILED'
        record.write_text(json.dumps(controller,indent=2),encoding='utf-8');raise SystemExit(code)
    print(run(args,study,config,equivalence))

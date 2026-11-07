"""Budgeted reference/cell and resume diagnostics from the Warmup endpoint."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
import time
import torch
import numpy as np
from .budget import PilotBudget
from .data import NativeOpticalDataset,sha256
from .gpu_update_check import cpu_tree,difference
from .joint import accumulate_joint
from .model import Encoder,ReflectanceDecoder
from .pilot_protocol import load_study,prepare_update,optimizer_groups,LinearWarmup
from .training import CAdamW,restore_checkpoint,save_checkpoint
from .processes import launch_python
from .geometry_cache import NativeGeometryCache


def rng_equal(a,b):
    return a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['torch'],b['torch']) and (a['cuda'] or {}).keys()==(b['cuda'] or {}).keys() and all(torch.equal(value,b['cuda'][key]) for key,value in (a['cuda'] or {}).items())


def compare(first,second,limits):
    checks={key:difference(first[key],second[key]) for key in ('modules','optimizer','gradients','predictions','scheduler','support')}
    failures=[]
    for key,metric,limit in [('modules','max_absolute',limits['parameter_update_or_resume_max_absolute']),('optimizer','relative_l2',limits['optimizer_state_relative_l2']),('gradients','relative_l2',limits['gradient_relative_l2']),('predictions','max_absolute',limits['prediction_max_absolute'])]:
        value=checks[key]
        if value[metric]>limit or value['scalar_mismatches'] or value['dtype_mismatches']:failures.append(key)
    if not checks['scheduler']['bitwise_equal'] or not checks['support']['bitwise_equal']:failures.append('schedule/support')
    if first['signature']!=second['signature']:failures.append('exposure')
    if not rng_equal(first['rng'],second['rng']):failures.append('RNG')
    nmse=[];tasks=[]
    if len(first['result']['rows'])!=len(second['result']['rows']):raise ValueError('Task layout differs')
    for a,b in zip(first['result']['rows'],second['result']['rows']):
        if (a['kind'],a['observation'],a['source'],a['exposure'])!=(b['kind'],b['observation'],b['source'],b['exposure']):raise ValueError('Task exposure differs')
        ra,rb=a['result'],b['result']
        if ra['loss'] is None or rb['loss'] is None:
            if ra['loss']!=rb['loss']:failures.append('missing task')
            continue
        tasks.append(abs(ra['loss']-rb['loss']))
        if len(ra['bands'])!=len(rb['bands']):raise ValueError('Band layout differs')
        for x,y in zip(ra['bands'],rb['bands']):
            if (x['band'],x['status'],x.get('count'),x.get('unsupported'))!=(y['band'],y['status'],y.get('count'),y.get('unsupported')):raise ValueError('Band exposure differs')
            if x.get('nmse') is not None:nmse.append(abs(x['nmse']-y['nmse']))
    checks['max_task_loss_difference']=max(tasks,default=0.)
    checks['max_band_nmse_difference']=max(nmse,default=0.)
    if checks['max_task_loss_difference']>limits['per_task_loss_max_absolute']:failures.append('task loss')
    if checks['max_band_nmse_difference']>limits['per_band_nmse_max_absolute']:failures.append('band NMSE')
    return dict(passed=not failures,failures=failures,checks=checks)


def main(args):
    root=args.run_root.resolve();fit=root/'fitting';identity=json.loads((fit/'identity.json').read_text(encoding='utf-8'))
    study=load_study(args.study);config=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    limits=json.loads((Path(__file__).parent/config['confirmed_preflight']['numerical_equivalence']['review']).read_text(encoding='utf-8'))
    if limits['review_type']!='explicit_user_design_confirmation' or limits!=identity['equivalence']['limits'] or identity['study']!=study['candidate_sha256'] or identity['optimizer_conditions']!=config['candidate']:raise ValueError('Unreviewed or mismatched preflight')
    for name,expected in identity['code'].items():
        if sha256(Path(__file__).with_name(name))!=expected:raise ValueError('Fitting code identity changed')
    if args.geometry_cache:
        measured=json.loads((root.parent/'real_geometry_cache_pipeline_comparison.json').read_text(encoding='utf-8'))
        if measured['status']!='MATCHED_REAL_GPU_GEOMETRY_CACHE_MEASUREMENTS_PASS_NO_UPDATES' or measured['study_sha256']!=study['candidate_sha256'] or measured['speedup']<=1 or not all(row['passed'] for row in measured['comparisons']) or measured['source_hashes']['geometry_cache.py']!=sha256(Path(__file__).with_name('geometry_cache.py')):raise ValueError('Geometry cache GPU comparison missing or changed')
    for name in ('AdamW','CAdamW'):
        if not (fit/name/'u000100.pt').exists():raise ValueError('Both matched fitting trajectories must reach update 100')
    budget=PilotBudget(root/'budget.json',study_sha256=study['candidate_sha256']);before=budget.summary()
    if before['incomplete_reservations'] or any(v<4 for v in before['remaining_by_candidate'].values()):raise ValueError('Uncertain or insufficient original budget')
    directory=root/'peak_update_check'
    if directory.exists():raise ValueError('Peak diagnostics already attempted; inspect evidence before any retry')
    if args.check_only:print('PEAK_CHECK_GATES_PASS_NO_UPDATES');return
    lock=root/'pilot_run.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.fsync(fd)
    cache_enabled=args.geometry_cache
    shadow_identity=dict(identity,diagnostic_execution=dict(mode='peak_FP32',cell_indexed=True,geometry_cache=cache_enabled,check_code=sha256(Path(__file__)),geometry_cache_code=sha256(Path(__file__).with_name('geometry_cache.py')) if cache_enabled else None))
    started=time.perf_counter();report=dict(status='RUNNING',budget_before=before,geometry_cache=cache_enabled,diagnostic_identity=shadow_identity,steps=[],comparisons={},formal_training_started=False,limitations=['Only measured FP32 peak-LR inputs; no BF16, DDP or formal-training authorization','Shadow diagnostic updates consume original budget and never replace fitting trajectory'])
    try:
        directory.mkdir();torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
        def persist():(directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        def resources():
            if time.perf_counter()-started>1800 or torch.cuda.max_memory_allocated(0)>12*1024**3:raise RuntimeError('Peak diagnostic resource bound reached')
            if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>8*1024**3:raise RuntimeError('Pilot output bound reached')
        persist();dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
        cache=NativeGeometryCache(data_version=study['source_identities']) if cache_enabled else None
        with cache.installed() if cache is not None else nullcontext():
            prepared={i:prepare_update(study,dataset,i,'cuda:0',cell_indexed_geometry=True) for i in (100,101)}
        parameters=config['candidate']
        def create(name,cell,path):
            encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').cuda(0).eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=cell).cuda(0).eval()
            optimizer=(torch.optim.AdamW if name=='AdamW' else CAdamW)(optimizer_groups([encoder,decoder],parameters['weight_decay']),lr=parameters['peak_lr'],betas=tuple(parameters['betas']),eps=parameters['eps'])
            scheduler=LinearWarmup(optimizer,peak=parameters['peak_lr'],warmup=parameters['warmup_updates'])
            expected=identity if path.parent==fit/name else shadow_identity
            state=restore_checkpoint(path,modules=dict(encoder=encoder,decoder=decoder),optimizer=optimizer,scheduler=scheduler,frozen_ids=expected)
            if state['logical_update'] not in (100,101):raise ValueError('Wrong diagnostic checkpoint')
            return encoder,decoder,optimizer,scheduler
        def step(name,branch,objects,index):
            encoder,decoder,optimizer,scheduler=objects;views,signature=prepared[index];scheduler.apply(index)
            encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True);predictions=[];supports=[]
            def capture(module,inputs,result):predictions.append(result[0].detach().cpu());supports.append(result[1].detach().cpu())
            hook=decoder.register_forward_hook(capture)
            use_cache=cache is not None and not branch.startswith('reference')
            try:
                with cache.installed() if use_cache else nullcontext():
                    result=accumulate_joint(encoder,decoder,views,logical_update=index,chunk_pixels=512,precision='fp32',clip_norm=parameters['clip_norm'])
            finally:hook.remove()
            gradients=cpu_tree({prefix+'.'+key:p.grad for prefix,module in [('encoder',encoder),('decoder',decoder)] for key,p in module.named_parameters()})
            if any(v is None for v in gradients.values()):raise RuntimeError('Missing gradient')
            resources();path=directory/f'{name}_{branch}.pt'
            if path.exists():raise ValueError('Refusing to overwrite diagnostic checkpoint')
            token=budget.reserve(name,index);entry=dict(optimizer=name,branch=branch,reservation=token,logical_input=index,signature=signature,status='RESERVED');report['steps'].append(entry);persist()
            optimizer.step();torch.cuda.synchronize(0)
            frozen=dict(shadow_identity,diagnostic_execution=dict(shadow_identity['diagnostic_execution'],cell_indexed=False,geometry_cache=False)) if branch.startswith('reference') else shadow_identity
            save_checkpoint(path,modules=dict(encoder=encoder,decoder=decoder),optimizer=optimizer,scheduler=scheduler,logical_update=index+1,sampler=dict(next_update=index+1,reservation=token,branch=branch),frozen_ids=frozen)
            budget.complete(token);entry.update(status='COMPLETED',checkpoint=str(path),sha256=sha256(path),result=result,cache=cache.summary() if use_cache else None);persist();resources()
            support=torch.cat(supports)
            state=torch.load(path,map_location='cpu',weights_only=False)
            return dict(modules=cpu_tree(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict())),optimizer=cpu_tree(optimizer.state_dict()),scheduler=cpu_tree(scheduler.state_dict()),gradients=gradients,predictions=torch.cat(predictions)[support],support=support,signature=signature,result=result,checkpoint=path,rng=state['rng'])
        for name in ('AdamW','CAdamW'):
            initial=fit/name/'u000100.pt';objects=create(name,False,initial);reference=step(name,'reference_101',objects,100);del objects;gc.collect()
            objects=create(name,True,initial);first=step(name,'cell_101',objects,100)
            comparison=compare(reference,first,limits);report['comparisons'][name]={'reference_vs_cell':comparison};persist()
            if not comparison['passed']:raise ValueError('Peak-LR reference/cell comparison failed')
            second=step(name,'uninterrupted_102',objects,101);del objects;gc.collect()
            if cache_enabled:cache=NativeGeometryCache(data_version=study['source_identities'])
            objects=create(name,True,first['checkpoint']);recovered=step(name,'restored_102',objects,101)
            comparison=compare(second,recovered,limits);report['comparisons'][name]['uninterrupted_vs_restored']=comparison;persist()
            if not comparison['passed']:raise ValueError('Peak-LR resume comparison failed')
            del objects,reference,first,second,recovered;gc.collect()
        report.update(status='PEAK_FP32_MEASURED_COMPARISONS_PASS',budget=budget.summary(),seconds=time.perf_counter()-started);persist()
    except Exception as exc:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',error=type(exc).__name__+': '+str(exc),budget=budget.summary())
        if directory.exists():(directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    finally:os.close(fd);lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--geometry-cache',action='store_true')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--check-only',action='store_true');mode.add_argument('--run',action='store_true');args=p.parse_args()
    if args.check_only or args.worker:main(args)
    else:
        record=args.run_root/'peak_check_controller.json';log=record.with_suffix('.log')
        with record.open('x',encoding='utf-8') as stream:json.dump(dict(status='STARTING'),stream)
        with log.open('x',encoding='utf-8') as output:
            child=launch_python(['-m','encoder_v3.peak_update_check',*sys.argv[1:],'--worker'],stdout=output)
            controller=dict(status='RUNNING',owned_worker_pid=child.pid,log=str(log.resolve()))
            record.write_text(json.dumps(controller,indent=2),encoding='utf-8')
            try:code=child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                code=1
            controller.update(status='COMPLETED' if code==0 else 'FAILED_NO_AUTOMATIC_RETRY',worker_exit_code=child.returncode)
            record.write_text(json.dumps(controller,indent=2),encoding='utf-8');raise SystemExit(code)

"""Matched full-native TRAIN A/B geometry reuse, forward/backward only."""
import gc,json,time
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from .data import NativeOpticalDataset,sha256
from .pilot_protocol import load_study
from .prepare_pilot import member,bounds
from .tasks import spatial_task,spectral_task
from .participation_encoder import ParticipationEncoder,prepare_participation_task
from .participation_validation_probe import confirmed_curve
from .participation_decoder import MatrixADecoder
from .participation_observation_batch_probe import parameter_hash
from .plan_prefetch import move_views
from . import joint,streaming,participation_encoder


def retained_array_bytes(views):
    unique={}
    def visit(value):
        if isinstance(value,np.ndarray):
            base=value
            while isinstance(base.base,np.ndarray):base=base.base
            unique[id(base)]=base.nbytes
        elif isinstance(value,dict):
            for v in value.values():visit(v)
        elif isinstance(value,(tuple,list)):
            for v in value:visit(v)
    for view in views:visit(view.plan.retained_layouts if view.plan is not None else None)
    return sum(unique.values())


def main():
    root=Path(__file__).resolve().parent;r=root/'reports';output=r/'participation_geometry_reuse_sweep.json'
    if output.exists():raise ValueError('Preserve existing measurement')
    started=time.perf_counter();torch.set_num_threads(2);study=load_study(r/'pilot_study_candidate.json');dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
    if dataset.identities!=study['source_identities']:raise ValueError('Reviewed source identity changed')
    free,total=torch.cuda.mem_get_info();cap=min(24*1024**3,free-2*1024**3)
    if cap<=0:raise RuntimeError('No unoccupied diagnostic GPU memory')
    torch.cuda.set_per_process_memory_fraction(cap/total);curve=confirmed_curve(r/'merge_rate_confirmation.json');torch.manual_seed(study['seed'])
    encoder=ParticipationEncoder(rate_curve=curve,selection_seed=study['seed'],hyper_hidden=64,local_batch_size=1024,packed_local_output=True).cuda().train();decoder=MatrixADecoder(indexed=True,cell_indexed=True,bulk_geometry=True).cuda().train();modules=(encoder,decoder)
    initial=parameter_hash(modules);entry=study['schedule'][0];lookup={row['id']:i for i,row in enumerate(dataset.rows)};frozen={row['id']:row for row in study['training_members']}
    names=('participation_geometry_reuse_probe.py','participation_encoder.py','adaptive_streaming.py','joint.py','participation_layout.py','adaptive_window.py','streaming.py','spatial_index.py','participation_decoder.py','participation_geometry.py')
    report=dict(status='RUNNING',optimizer_updates=0,effective_batch=8,device_uuid=str(torch.cuda.get_device_properties(0).uuid),memory_cap_bytes=cap,source_identities=dataset.identities,source_sha256={n:sha256(root/n) for n in names},study_sha256=study['candidate_sha256'],merge_recipe=encoder.merge_recipe,initial_parameter_sha256=initial,rows=[],default_adopted=False,limits=['Same current A rule and full approved schedule0 TRAIN windows, not all-member acceptance','No optimizer update or real resume comparison; stopped budget unchanged','End-to-end diagnostic includes read/decode/plan/transfer/forward/backward and predictions, excludes optimizer/validation/save','Forward/backward time includes internal CPU geometry and reporting, not CUDA kernel-only time','Retained-byte count is unique NumPy payload only, excludes Python containers and other plan storage'])
    def save():
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(output)
    def prepare(retain):
        views=[];load=planning=0.
        for spec in entry['tasks']:
            start=time.perf_counter();row=dataset.rows[lookup[spec['observation']]]
            if member(row)!=frozen[row['id']]:raise ValueError('Reviewed member changed')
            obs=dataset[lookup[row['id']]];stds=dataset.engineering_train_stds(obs);left,bottom,_,_=bounds(row);load+=time.perf_counter()-start
            tasks=(spatial_task(obs,spec['A_rectangle']),spectral_task(obs,seed=study['seed'],update=0))
            if tasks[1].metadata['hidden_band_ids']!=spec['B1_hidden']:raise ValueError('Hidden-band schedule differs')
            for task in tasks:
                start=time.perf_counter();plan=prepare_participation_task(task,rate_curve=curve,selection_seed=study['seed'],origin_xy=(left,bottom),chunk_pixels=16384,cell_indexed=True,retain_layouts=retain,physical_patch_m=entry['patch_m']);planning+=time.perf_counter()-start;views.append(joint.PlannedView(task,plan,stds,obs.metadata['sensor']))
        return views,load,planning
    original_roots=streaming.stream_task_roots;original_layout=participation_encoder.participation_layout;context={};predictions={};calls=[0]
    def track(*args,**kwargs):
        task=args[2];context.update(id=task.metadata['id'],kind=task.kind,ordinal=0);return original_roots(*args,**kwargs)
    def collect(module,inputs,output):
        key=(context['id'],context['kind'],context['ordinal']);context['ordinal']+=1
        if key in predictions:raise ValueError('Duplicate target chunk identity')
        predictions[key]=(output[0].detach().clone(),output[1].detach().clone(),inputs[1].detach().clone(),inputs[2].detach().clone())
    def count(*args,**kwargs):calls[0]+=1;return original_layout(*args,**kwargs)
    save();reference=None
    try:
        for retain in (False,True):
            warm,_,_=prepare(retain);warm=move_views(warm,'cuda:0');joint.accumulate_joint(encoder,decoder,warm,logical_update=0,chunk_pixels=16384,precision='fp32',clip_norm=3.);torch.cuda.synchronize();del warm;gc.collect()
            for repeat in range(2):
                report['phase']=dict(retain_layouts=retain,repeat=repeat,time=time.time());save();begin=time.time();start=time.perf_counter();views,load,planning=prepare(retain);nbytes=retained_array_bytes(views)
                transfer_start=time.perf_counter();views=move_views(views,'cuda:0');torch.cuda.synchronize();transfer=time.perf_counter()-transfer_start;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();predictions={};calls[0]=0
                handle=decoder.register_forward_hook(collect);compute_start=time.perf_counter()
                try:
                    with patch.object(streaming,'stream_task_roots',track),patch.object(participation_encoder,'participation_layout',count):result=joint.accumulate_joint(encoder,decoder,views,logical_update=0,chunk_pixels=16384,precision='fp32',clip_norm=3.)
                    torch.cuda.synchronize();compute=time.perf_counter()-compute_start
                finally:handle.remove()
                end_to_end=time.perf_counter()-start;end_unix=time.time();peak=torch.cuda.max_memory_allocated();current={key:tuple(v.cpu() for v in values) for key,values in predictions.items()};predictions.clear()
                gradients=torch.cat([torch.zeros_like(p).detach().cpu().double().flatten() if p.grad is None else p.grad.detach().cpu().double().flatten() for m in modules for p in m.parameters()]);presence=[p.grad is not None for m in modules for p in m.parameters()]
                if reference is None:reference=(current,gradients,presence,result)
                previous,old_gradient,old_presence,old_result=reference
                if previous.keys()!=current.keys() or presence!=old_presence or result['global_tokens']!=old_result['global_tokens'] or result['global_padding_fraction']!=0:raise ValueError('Supervision, gradient presence or tokens changed')
                maximum=0.;pixels=0
                for key,(prediction,supported,boxes,condition) in current.items():
                    old_prediction,old_supported,old_boxes,old_condition=previous[key]
                    if not torch.equal(supported,old_supported) or not torch.equal(boxes,old_boxes) or not torch.equal(condition,old_condition):raise ValueError('Support, targets or target SRF condition changed')
                    pixels+=int(supported.sum())
                    if supported.any():maximum=max(maximum,float((prediction[supported]-old_prediction[supported]).abs().max()))
                identity=lambda x:{(r['observation'],r['kind']):dict(coefficient=r['coefficient'],exposure=r['exposure'],support=[(b['band'],b['count'],b['unsupported']) for b in r['result']['bands']]) for r in x['rows']}
                if identity(result)!=identity(old_result):raise ValueError('Exposure or normalization changed')
                old_bands={(r['observation'],r['kind'],b['band']):b['nmse'] for r in old_result['rows'] for b in r['result']['bands']}
                nmse=max(abs(b['nmse']-old_bands[(r['observation'],r['kind'],b['band'])]) for r in result['rows'] for b in r['result']['bands'] if b['nmse'] is not None)
                row=dict(retain_layouts=retain,repeat=repeat,forward_backward_seconds=compute,diagnostic_end_to_end_seconds=end_to_end,CPU_read_seconds=load,CPU_plan_seconds=planning,transfer_seconds=transfer,retained_numpy_bytes=nbytes,forward_layout_calls=calls[0],global_tokens=result['global_tokens'],supported_pixels=pixels,padding_fraction=0.,peak_allocated_bytes=peak,prediction_max_abs=maximum,gradient_relative_l2=float((gradients-old_gradient).norm()/old_gradient.norm().clamp_min(1e-24)),max_band_nmse_abs=nmse,max_task_loss_abs=max(abs(result[k]-old_result[k]) for k in ('A_loss','B_loss')),timed_start_unix=begin,timed_end_unix=end_unix)
                row['within_existing_FP32_values_diagnostic_only']=row['prediction_max_abs']<=2e-6 and row['gradient_relative_l2']<=2e-6 and row['max_band_nmse_abs']<=1e-5 and row['max_task_loss_abs']<=1e-6
                if not row['within_existing_FP32_values_diagnostic_only'] or retain and row['forward_layout_calls']!=0:raise ValueError('Geometry reuse changed numerical result or recomputed layouts')
                report['rows'].append(row);save();print(json.dumps(row),flush=True);del views,current;gc.collect()
        report.update(status='GEOMETRY_REUSE_MATCHED_DIAGNOSTICS_COMPLETE_NOT_FORMAL_ACCEPTANCE',final_parameter_sha256=parameter_hash(modules));report['parameters_unchanged']=report['final_parameter_sha256']==initial
        if not report['parameters_unchanged']:raise RuntimeError('Parameters changed without an optimizer')
    except BaseException as exc:report.update(status='FAILED',error=dict(type=type(exc).__name__,message=str(exc)));raise
    finally:report['seconds']=time.perf_counter()-started;save()


if __name__=='__main__':main()

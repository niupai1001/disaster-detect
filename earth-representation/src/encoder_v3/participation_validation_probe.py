"""Run existing expanded B1 cases with confirmed preflight Merge, without updates."""
import argparse,json,math,time
from pathlib import Path
import torch
from .adaptive_census import spectral_panel
from .adaptive_streaming import stream_adaptive_task
from .data import NativeOpticalDataset,sha256
from .tasks import fixed_spectral_task
from .joint import PlannedView
from .model import ReflectanceDecoder
from .participation_encoder import ParticipationEncoder,prepare_participation_task
from .participation_layout import AreaEquivalentHillRate
from .participation_observation_batch_probe import parameter_hash
from .pilot_protocol import load_study
from .plan_prefetch import move_views
from .prepare_pilot import member,bounds


def confirmed_curve(path):
    review=json.loads(Path(path).read_text(encoding='utf-8'))
    if review.get('status')!='USER_CONFIRMED_PREFLIGHT_ONLY' or review.get('curve',{}).get('effective_gsd')!='sqrt(gx*gy)':raise ValueError('Confirmed preflight participation curve required')
    return AreaEquivalentHillRate(**{k:review['curve'][k] for k in ('amplitude','scale_m','power')})


def evaluate_case(observation,hidden,*,encoder,decoder,train_stds,origin_xy,chunk_pixels):
    task=fixed_spectral_task(observation,hidden);start=time.perf_counter()
    plan=prepare_participation_task(task,rate_curve=encoder.rate_curve,selection_seed=encoder.selection_seed,origin_xy=origin_xy,chunk_pixels=chunk_pixels,cell_indexed=True)
    planning=time.perf_counter()-start;device=next(encoder.parameters()).device;start=time.perf_counter()
    moved=move_views([PlannedView(task,plan,train_stds,observation.metadata['sensor'])],device)[0]
    if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    transfer=time.perf_counter()-start;audit=[]
    def capture(module,inputs,result):audit.extend(result.participation_audit)
    handle=encoder.register_forward_hook(capture);start=time.perf_counter()
    try:
        with torch.no_grad():result=stream_adaptive_task(encoder,decoder,moved.task,plan,train_stds=train_stds,chunk_pixels=chunk_pixels,loss_weight=1.,backward=False,precision='fp32')
        if device.type=='cuda':torch.cuda.synchronize(device)
    finally:handle.remove()
    if result['status']=='OK' and (not math.isfinite(result['loss']) or {b['band'] for b in result['bands']}!=set(hidden)):raise ValueError('Nonfinite result or changed target identities')
    if any(p.grad is not None for module in (encoder,decoder) for p in module.parameters()):raise RuntimeError('Forward-only diagnostic produced gradients')
    return dict(hidden=list(hidden),task_signature=plan.signature if plan is not None else None,CPU_planning_seconds=planning,transfer_seconds=transfer,forward_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,participation_audit=audit,**result)


def main(args):
    root=Path(__file__).resolve().parent;r=root/'reports';output=args.output;rows_path=output.with_suffix('.jsonl')
    if output.exists() or rows_path.exists():raise ValueError('Preserve existing diagnostic artifacts')
    started=time.perf_counter();torch.set_num_threads(2);study=load_study(r/'pilot_study_candidate.json');census=json.loads((r/'adaptive_member_census.json').read_text(encoding='utf-8'))
    dataset=NativeOpticalDataset(study['release'],study['contracts'],role='VAL')
    if dataset.identities!=study['source_identities'] or census['source_identities']!=dataset.identities or census['study_sha256']!=sha256(r/'pilot_study_candidate.json'):raise ValueError('Frozen panel source identity differs')
    if not torch.cuda.is_available():raise RuntimeError('Real GPU panel requires CUDA')
    actual=str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix('GPU-')!=args.device_uuid.removeprefix('GPU-'):raise ValueError('GPU identity differs')
    free,total=torch.cuda.mem_get_info();cap=min(24*1024**3,free-2*1024**3)
    if cap<=0:raise RuntimeError('No unoccupied diagnostic memory')
    torch.cuda.set_per_process_memory_fraction(cap/total);curve=confirmed_curve(r/'merge_rate_confirmation.json');torch.manual_seed(study['seed'])
    encoder=ParticipationEncoder(rate_curve=curve,selection_seed=study['seed'],hyper_hidden=64,local_batch_size=1024,packed_local_output=True).cuda().eval();decoder=ReflectanceDecoder(indexed=True,cell_indexed=True).cuda().eval();modules=(encoder,decoder)
    initial=parameter_hash(modules);names=('participation_validation_probe.py','participation_encoder.py','participation_layout.py','adaptive_window.py','adaptive_streaming.py','spatial_index.py','streaming.py','model.py','adaptive_census.py','tasks.py')
    report=dict(status='RUNNING',scope='Random-initialization B1 execution diagnostics, not trained quality or formal validation adoption',optimizer_updates=0,formal_panel_replaced=False,training_sampling_changed=False,source_identities=dataset.identities,study_sha256=study['candidate_sha256'],panel_census_sha256=sha256(r/'adaptive_member_census.json'),curve_confirmation_sha256=sha256(r/'merge_rate_confirmation.json'),merge_recipe=encoder.merge_recipe,source_sha256={n:sha256(root/n) for n in names},device_uuid=actual,memory_cap_bytes=cap,initial_parameter_sha256=initial,completed_cases=0,expected_cases=sum(len(p['hidden_band_combinations']) for p in census['validation_panels']),members=[],rows_path=str(rows_path),limits=['Random initialization: metrics check wiring, not learned information preservation','Expanded panel does not replace the scheduler validation metric','No optimizer, backward, training update, EMA update or parameter search','Only eight reviewed VAL members; no B2 or all-machine data admission'])
    def save():
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(output)
    save();lookup={row['id']:i for i,row in enumerate(dataset.rows)};frozen={row['id']:row for row in study['validation_members']}
    try:
        with rows_path.open('w',encoding='utf-8') as stream:
            for panel in census['validation_panels']:
                identifier=panel['observation'];index=lookup[identifier]
                if member(dataset.rows[index])!=frozen[identifier]:raise ValueError('Reviewed validation member changed')
                obs=dataset[index];generated=spectral_panel(obs.bands)
                if obs.metadata['cache_sha256']!=panel['cache_sha256'] or generated['hidden_band_combinations']!=panel['hidden_band_combinations'] or generated['spectral_order']!=panel['spectral_order']:raise ValueError('Expanded panel/cache changed')
                left,bottom,_,_=bounds(dataset.rows[index]);stds=dataset.engineering_train_stds(obs)
                report['current_member']=identifier;report['members'].append(dict(observation=identifier,cache_sha256=panel['cache_sha256'],spectral_order=panel['spectral_order'],cases=len(panel['hidden_band_combinations']),counts_by_hidden_number=generated['counts_by_hidden_number']))
                for ordinal,hidden in enumerate(panel['hidden_band_combinations']):
                    begin=time.time();row=evaluate_case(obs,hidden,encoder=encoder,decoder=decoder,train_stds=stds,origin_xy=(left,bottom),chunk_pixels=16384)
                    row.update(observation=identifier,case_ordinal=ordinal,begin_unix=begin,end_unix=time.time());stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    report['completed_cases']+=1;report['current_hidden']=hidden;save()
                if sha256(obs.metadata['cache'])!=obs.metadata['cache_sha256']:raise ValueError('Validation cache changed')
                print(json.dumps(dict(observation=identifier,completed_cases=report['completed_cases'])),flush=True)
        report['final_parameter_sha256']=parameter_hash(modules);report['parameters_unchanged']=report['final_parameter_sha256']==initial
        if not report['parameters_unchanged'] or report['completed_cases']!=report['expected_cases']:raise RuntimeError('Incomplete diagnostic or changed parameters')
        report['rows_sha256']=sha256(rows_path);report['status']='B1_FULL_PANEL_FORWARD_DIAGNOSTICS_COMPLETE_NOT_QUALITY_ACCEPTANCE'
    except BaseException as exc:
        report['status']='FAILED';report['error']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:report['seconds']=time.perf_counter()-started;save()
    print(json.dumps({key:report[key] for key in ('status','completed_cases','parameters_unchanged','seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device-uuid',required=True);main(parser.parse_args())

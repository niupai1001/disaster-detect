"""Apply explicitly reviewed FP32 limits to immutable measurement evidence."""
import json
from pathlib import Path
from .data import sha256


def require_cell_equivalence(config,root,study):
    root=Path(root);entry=config.get('confirmed_preflight',{}).get('numerical_equivalence',{})
    if not entry.get('review'):raise ValueError('Numerical equivalence parameters not confirmed')
    limits=json.loads((root/entry['review']).read_text(encoding='utf-8'))
    if limits.get('review_type')!='explicit_user_design_confirmation' or limits.get('scope')!='FP32_single_GPU_native_cell_readout_preflight_only':raise ValueError('Wrong numerical review scope')
    marker=json.loads((root/'reports/pilot_runs/update_check_completed.json').read_text(encoding='utf-8'))
    report_path=Path(marker['report'])
    if sha256(report_path)!=marker['sha256']:raise ValueError('Update evidence changed')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report['frozen_ids']['study']!=study['candidate_sha256'] or len(report['check_steps'])!=8 or report['budget']['completed']!=8:raise ValueError('Update evidence does not match this study')
    checkpoint=json.loads((root/'reports/gpu_update_checkpoint_review.json').read_text(encoding='utf-8'))
    for step in report['check_steps']:
        if step['status']!='COMPLETED' or sha256(step['checkpoint'])!=step['checkpoint_sha256']:raise ValueError('Incomplete or changed check checkpoint')
        if step['signature']!=report['update_input_signatures'][step['logical_input']] or step['result']['effective_batch']!=8:raise ValueError('Update exposure differs')
    failures=[]
    for name,comparisons in report['comparisons'].items():
        for kind,values in comparisons.items():
            for field,key,measure in [('parameters','parameter_update_or_resume_max_absolute','max_absolute'),('optimizer_state','optimizer_state_relative_l2','relative_l2'),('predictions','prediction_max_absolute','max_absolute')]:
                value=values[field]
                if value[measure]>limits[key] or value['scalar_mismatches'] or value['dtype_mismatches']:failures.append(name+'/'+kind+'/'+field)
            if kind=='reference_vs_cell' and values['gradients']['relative_l2']>limits['gradient_relative_l2']:failures.append(name+'/gradient')
            if kind=='uninterrupted_vs_restored' and not (values['schedule_equal'] and values['next_input_signature_equal']):failures.append(name+'/resume exposure')
        extra=checkpoint['results'][name]
        if extra['max_per_band_nmse_difference']>limits['per_band_nmse_max_absolute'] or max(extra['A_loss_difference'],extra['B_loss_difference'])>limits['per_task_loss_max_absolute']:failures.append(name+'/loss')
        if not all(extra[key] for key in ('logical_update_equal','next_update_equal','LR_equal')) or not all(extra['rng'][key] for key in ('torch_cpu_equal','cuda_rng_equal','python_equal','numpy_equal')) or extra['rng']['cuda_devices']!=['0']:failures.append(name+'/exact state')
    if failures:raise ValueError('FP32 equivalence failed: '+', '.join(failures))
    return dict(status='REVIEWED_FP32_LIMITS_PASS_ON_MEASURED_INPUTS',limits=limits,measurement_sha256=marker['sha256'],scope='Warmup inputs 0/1, P60/P120, single GPU; not peak LR, BF16, multi-GPU or formal authorization')

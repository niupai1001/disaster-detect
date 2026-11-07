"""Summarize evidence without promoting partial checks into stage acceptance."""
import json
from pathlib import Path
from .data import sha256
from .budget import PilotBudget


if __name__=='__main__':
    root=Path(__file__).resolve().parent;reports=root/'reports'
    names=['engineering_tests.json','native_cache_audit.json','inventory_metadata.json','inventory_scope_review.json','real_s2_probe.json','real_landsat_indexed_layout_probe.json','real_full_s2_indexed_probe.json','real_decoder_index_probe.json','real_precision_probe.json','real_full_target_streaming.json','real_prefetch_probe.json','dual_gpu_communication_unrestricted.json','hypernetwork_preflight_review.json','spatial_mask_preflight_review.json','real_joint_batch8.json','independent_gpu_batch8_unrestricted.json','pilot_study_candidate.json','pilot_study_review.md']
    names+=['mounted_root_discovery.json','real_microbatch_p120.json','real_cell_index_comparison.json','real_cell_index_p60_comparison.json','real_task_python_profile.json','real_task_cuda_profiler_aborted.json','real_task_cpu_operator_profile.json','p60_gpu_sampling_summary.json','gpu_efficiency_review.md']
    names+=['native_lineage_audit.json','native_lineage_audit_source_files.json','lineage_qa_header_review.json','inventory_release_reconciliation.json','pilot_members_panels_review.json','gpu_update_execution_review.json','pilot_first_inputs_check.json','gpu_update_checkpoint_review.json','fp32_equivalence_tolerance_proposal.json','fp32_equivalence_tolerance_review.md','pilot_runs/update_check_completed.json','pilot_runs/update_check_a624c7d0c860/report.json','pilot_runs/budget.json']
    names+=['fp32_equivalence_tolerance_review.json','fp32_equivalence_acceptance.json','cuda_runtime_recovery.json','gpu_restart_checkpoint_comparison.json','fast_geometry_pipeline_probe.json','cpu_support_index_comparison.json','cross_source_pair_candidates.json','cross_source_pair_QA_support.json','local_QA_candidate_inventory.json']
    names+=['failed_prefetch_reproduction.json','prefetch_concurrency_reproduction.json','prefetch_failure_incident.json','local_uav_hyperspectral_header_audit.json','discovery_scope_notes.json']
    evidence={name:dict(path=str((reports/name).resolve()),sha256=sha256(reports/name)) for name in names}
    tests=json.loads((reports/'engineering_tests.json').read_text());audit=json.loads((reports/'native_cache_audit.json').read_text());precision=json.loads((reports/'real_precision_probe.json').read_text())
    assert tests['status'] in ('ENGINEERING_TESTS_PASS','ENGINEERING_TESTS_PASS_WITH_SKIPS')
    assert audit['status']=='CACHE_REREAD_PASS_NOT_V3_ADMISSION'
    assert -1.00000001<=precision['gradient_cosine']<=1.00000001
    config=json.loads((root/'config.json').read_text())
    study=json.loads((reports/'pilot_study_candidate.json').read_text(encoding='utf-8'))
    budget=PilotBudget(reports/'pilot_runs/budget.json',study_sha256=study['candidate_sha256']).summary()
    progress={}
    for name in ('AdamW','CAdamW'):
        paths=sorted((reports/'pilot_runs/fitting'/name).glob('u[0-9][0-9][0-9][0-9][0-9][0-9].pt'))
        progress[name]=int(paths[-1].stem[1:]) if paths else 0
    report=dict(status='IN_PROGRESS_NOT_READY_FOR_FORMAL_TRAINING',engineering_tests=tests['tests'],engineering_test_skips=tests.get('skipped',[]),existing_native_members_reread=audit['read_members'],real_training_updates=budget['completed'],optimizer_pilot_updates=budget['reserved'],pilot_budget=budget,fitting_pilot_updates=sum(progress.values()),fitting_progress=progress,formal_training_started=False,
                stages={'M0':'7,778 existing cache lineages exactly reproduced; 1,721 original QA recalculations and 6,057 traced parent masks; all-machine admission incomplete','M1':'Hypernetwork confirmed for preflight; source GSD/SRF/support metadata exported without additional embeddings','M2':'grouped/tiled synthetic equivalence pass; real broad acceptance incomplete','M3':'indexed/reference geometry equivalence pass; full-grid acceptance incomplete','M4':'local receptive regions and shared complete-input union after global attention retained; isolated observation batches tested','M5':'cell-indexed readout faster on matched P60/P120; reviewed FP32 limits passed on first two warmup inputs only','M6':'budgeted checks complete; bounded matched fitting in progress; actual progress and remaining budget reported separately; no DDP acceptance','M7':'not started','M8':'predictor/EMA helpers tested; cross-source training not started','M9':'task-head budget unset; training not started'},
                confirmed_preflight=config.get('confirmed_preflight'),required_before_training=config['required_before_training'],evidence=evidence,next_work=['Continue same fitting checkpoints to warmup endpoint; reserve budget for peak-LR/update/resume diagnostics','Continue full optical inventory/source association and V3 pair geometry/QA/change-risk audit','Track parent-only QA limitations and categorical QA alignment warning; do not equate reproduction with admission','Current recovered runtime exposes one GPU; multi-GPU checks require two available devices','Measure full end-to-end data preparation/wait/utilization and complete precision/validation/stability report before formal training'])
    report['cell_index_performance_candidates']=[]
    report['unresolved_runtime_incident']=json.loads((reports/'prefetch_failure_incident.json').read_text(encoding='utf-8'))
    report['mounted_filename_scan_summary']=str((reports/'mounted_raster_discovery/summary.json').resolve())
    for name in ('real_cell_index_comparison.json','real_cell_index_p60_comparison.json'):
        measurement=json.loads((reports/name).read_text());reference,candidate=measurement['rows']
        assert measurement['status']=='REAL_MICROBATCH_MEASUREMENTS_NOT_ACCEPTANCE'
        assert reference['supported_supervision_pixels']==candidate['supported_supervision_pixels']
        report['cell_index_performance_candidates'].append(dict(patch_m=measurement['patch_m'],reference_seconds=reference['median_seconds'],candidate_seconds=candidate['median_seconds'],forward_backward_speedup=candidate['speedup_vs_microbatch_1'],relative_gradient_difference=candidate['relative_gradient_difference'],default_adopted=False))
    (reports/'execution_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','engineering_tests','existing_native_members_reread','real_training_updates','optimizer_pilot_updates')},indent=2))

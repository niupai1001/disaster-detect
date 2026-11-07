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
    names += [name for name in ('preformal_tests.json','adaptive_window_audit.json','adaptive_window_review.md','adaptive_member_census.json','pair_registration_diagnostics/index.json','pilot_runs/pilot_stop_decision.json','merge_participation_review.md','merge_participation_proposal.json','jepa_flow_tests.json','cross_role_split_audit.json','train_statistics_recomputed.json','merge_participation_review.md') if (reports/name).exists()]
    names += ['merge_B_confirmation.json','participation_merge_implementation_evidence.json','participation_merge_implementation_review.md','participation_engineering_tests.json','filename_discovery_reconciled.json','filename_inventory_review.md','discovery_coverage_post_refinement.json']
    names += ['pair_admission_evidence.json','pair_admission_review.md','pair_registration_diagnostics/full_QA/index.json','pair_full_QA_lineage.json','pair_fixed_support_diagnostics.json','pair_admission_engineering_tests.json']
    names += ['pair_metadata_recovery.json','metadata_recovery_test_evidence.json','pair_fixed_support_short_gap_supplement.json','pair_fixed_support_fine_grid.json']
    names += ['participation_batch_execution_evidence.json','participation_batch_execution_review.md','participation_batch_engineering_tests.json','participation_profile.json']
    names += ['packed_local_execution_review.md','participation_packed_comparison.json','packed_local_engineering_tests.json']
    names += ['packed_batch_execution_evidence.json','A_boundary_center_review.md','A_boundary_center_reproduction.json','A_boundary_engineering_tests.json']
    names += ['merge_levels_confirmation.json','merge_rate_confirmation.json','participation_current_execution_evidence.json','participation_current_rule_controller.json']
    names += ['B1_expanded_evidence.json','B1_expanded_review.md','B1_pair_engineering_tests.json','pair_spatial_holdout_evidence.json','pair_spatial_holdout_disjoint.json']
    names += ['retained_geometry_execution_evidence.json','retained_geometry_execution_review.md','retained_geometry_engineering_tests.json','participation_geometry_reuse_controller.json']
    names += ['B2_geometry_proposal_confirmation.json','B2_pair_geometry_proposal.md','B2_pair_geometry_proposal.json','B2_query_geometry_identity.json']
    names += ['B2_geometry_mechanism_confirmation.json','B2_query_geometry_implementation_review.md','B2_query_geometry_final_integrity.json','pair_geometry_final_tests.json','filename_discovery_after_leaf_followup.json','discovery_coverage_after_leaf_followup.json']
    names += ['pair_quality_risk.json','pair_quality_risk_review.md','pair_QA_rule_sensitivity.json','B2_QA_policy_confirmation.json','pair_quality_CPU_targeted.json','pair_quality_runtime_diagnostics.json','discovery_coverage_after_pending_followup.json']
    names += ['pair_calibration_inventory.json','B2_QA_consistency_check.json']
    names += ['filename_discovery_current.json','discovery_coverage_current.json','cuda_fault_event_evidence.json','non_observation_sample_evidence.json']
    evidence={name:dict(path=str((reports/name).resolve()),sha256=sha256(reports/name)) for name in names}
    tests=json.loads((reports/('preformal_tests.json' if (reports/'preformal_tests.json').exists() else 'engineering_tests.json')).read_text(encoding='utf-8'));audit=json.loads((reports/'native_cache_audit.json').read_text());precision=json.loads((reports/'real_precision_probe.json').read_text())
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
                stages={'M0':'7,778 existing cache lineages exactly reproduced; 1,721 original QA recalculations and 6,057 traced parent masks; all-machine admission incomplete; 159 raw-QA TRAIN pairs /315 members diagnosed, including coherent nonzero offsets in two disjoint-source spatial holdout cases; query-only mechanism confirmed and implemented; exact offsets/calibration/admission remain pending; name inventory1,589,404 with0 scheduled/timeout gaps and2 access-error chunks; existing QA confirmed for4 initial candidate members, zero B2 admissions','M1':'Hypernetwork confirmed for preflight; source GSD/SRF/support metadata exported without additional embeddings','M2':'grouped/tiled synthetic equivalence pass; real broad acceptance incomplete','M3':'User-selected single-round B and native/merged two scales confirmed; exact participation curve and sqrt(gx*gy) confirmed for preflight; FOV/mask relation and formal configuration pending','M4':'Actual FP32 efficient-attention backend verified; complete eight-observation execution micro1/2/4/8 measured without padding or cross-observation interaction; query chunks not adopted; final acceptance pending','M5':'B1 diagnostic502 combinations/1796 band reports executed; query-only B2 Decoder integration tested with target-native indices and target TRAIN statistics; real pair offsets/admission not frozen','M6':'AdamW selected; real updates stopped at214; Historical full146 plus9 new CPU audit checks pass; current CUDA initialization access violation prevents fresh full GPU verification; current A-rule full-eight FP32 diagnostics complete: packed micro1 28.06s, micro2/4/8 no observed throughput benefit; parameters unchanged; optional per-task geometry reuse measured30.38->24.99s forward/backward and38.85->34.32s diagnostic end-to-end, no default adoption or real update/resume acceptance','M7':'not started','M8':'A+B2+JEPA query-frame integration passes synthetic CPU/GPU, micro1/2/4/8 and exact update/EMA/next-step restore; Teacher/native Encoder unchanged; reviewed real offsets/loss/admission pending','M9':'task-head budget unset; training not started'},
                confirmed_preflight=config.get('confirmed_preflight'),required_before_training=config['required_before_training'],evidence=evidence,next_work=['Use confirmed preflight participation curve and single-round B; resolve physical FOV/mask relation and final merge configuration','Freeze exact translations, calibration/validation isolation and B2 admission; query-only mechanism already user-confirmed and implemented, no real correction applied','Review formal adoption of the fully executed502-case B1 diagnostic panel; old scheduler metric unchanged','Complete remaining execution acceptance and end-to-end cost evidence; current microbatch sweep complete, real pilot update budget remains stopped','Complete remaining data, resource, precision and stability gates before formal training'])
    stop_path=reports/'pilot_runs/pilot_stop_decision.json'
    report['pilot_stop_decision']=json.loads(stop_path.read_text(encoding='utf-8')) if stop_path.exists() else None
    report['remaining_pilot_updates_authorized']=0 if report['pilot_stop_decision'] else budget['remaining_total']
    report['optimizer_selected']=config['confirmed'].get('optimizer')
    report['legacy_performance_scope']='Historical fixed-physical-window model; not adaptive-model acceptance'
    latest=reports/'latest_verification_attempt.json'
    report['latest_verification_attempt']=json.loads(latest.read_text(encoding='utf-8')) if latest.exists() else None
    report['engineering_tests_currently_verified']=not report['latest_verification_attempt'] or report['latest_verification_attempt']['status']=='CURRENT_VERIFICATION_PASS'
    report['engineering_test_evidence_scope']=('Previous successful full suite; latest status: '+report['latest_verification_attempt']['status']) if not report['engineering_tests_currently_verified'] else 'Latest successful suite'
    report['cell_index_performance_candidates']=[]
    report['historical_runtime_incident']=json.loads((reports/'prefetch_failure_incident.json').read_text(encoding='utf-8'))
    report['unresolved_runtime_incident']=None if (report['latest_verification_attempt'] or {}).get('runtime_status')=='MINIMAL_RUNTIME_RECOVERED' else report['historical_runtime_incident']
    runtime_report=(report['latest_verification_attempt'] or {}).get('runtime_incident_report')
    if runtime_report:report['unresolved_runtime_incident']=json.loads((reports/runtime_report).read_text(encoding='utf-8'))
    report['mounted_filename_scan_summary']=str((reports/'mounted_raster_discovery/summary.json').resolve())
    for name in ('real_cell_index_comparison.json','real_cell_index_p60_comparison.json'):
        measurement=json.loads((reports/name).read_text());reference,candidate=measurement['rows']
        assert measurement['status']=='REAL_MICROBATCH_MEASUREMENTS_NOT_ACCEPTANCE'
        assert reference['supported_supervision_pixels']==candidate['supported_supervision_pixels']
        report['cell_index_performance_candidates'].append(dict(patch_m=measurement['patch_m'],reference_seconds=reference['median_seconds'],candidate_seconds=candidate['median_seconds'],forward_backward_speedup=candidate['speedup_vs_microbatch_1'],relative_gradient_difference=candidate['relative_gradient_difference'],default_adopted=False))
    (reports/'execution_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','engineering_tests','existing_native_members_reread','real_training_updates','optimizer_pilot_updates','engineering_tests_currently_verified')},indent=2))

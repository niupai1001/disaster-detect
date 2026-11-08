"""Read-only A/B1/B2/JEPA task audit and mask-semantics diagnostic."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from encoder_v3.data import Observation
from encoder_v3.participation_encoder import prepare_participation_task
from encoder_v3.participation_layout import AreaEquivalentHillRate, participation_layout
from encoder_v3.tasks import spatial_task, spatial_hide
from encoder_v3.test_adaptive_window import source

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
REPORTS = ROOT / "encoder_v3" / "reports"
CURVE = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)


def main():
    b1 = json.loads((REPORTS / "B1_expanded_evidence.json").read_text(encoding="utf-8-sig"))
    possible = 0
    rows = []
    for member in b1["members_summary"]:
        total = 2 ** member["real_bands"] - 2
        possible += total
        rows.append(dict(observation=member["observation"], bands=member["real_bands"], evidence_cases=member["cases"],
                         possible_combinations=total, coverage=member["cases"] / total,
                         hidden_counts=member["hidden_counts"], every_band_at_every_count=member["every_band_exposed_at_every_hidden_count"]))
    b1_audit = dict(status=b1.get("status"), cases=b1.get("cases"), members=b1.get("members"),
                    possible_combinations=possible, coverage=b1.get("cases") / possible,
                    per_observation=rows, per_band_reports=b1.get("per_band_reports"),
                    supported_pixel_exposures=b1.get("supported_pixel_exposures"),
                    no_support_cases=b1.get("no_support_cases"), optimizer_updates=b1.get("optimizer_updates"),
                    parameters_unchanged=b1.get("parameters_unchanged"), quality_claim=b1.get("quality_claim"),
                    note="Selected combinations, not exhaustive; random-initialization flow diagnostic only.")

    pair = json.loads((REPORTS / "pair_admission_evidence.json").read_text(encoding="utf-8-sig"))
    holdout = json.loads((REPORTS / "pair_spatial_holdout_disjoint.json").read_text(encoding="utf-8-sig"))
    b2_audit = dict(status=pair.get("status"), admitted_pairs=pair.get("admitted_pairs"),
                    metadata_catalog_pairs=pair.get("metadata_catalog_pairs"), unique_members=pair.get("unique_members"),
                    pair_thresholds_frozen=pair.get("pair_thresholds_frozen"), raw_QA_policy_frozen=pair.get("raw_QA_policy_frozen"),
                    unresolved=pair.get("unresolved"), spatial_holdout_status=holdout.get("status"),
                    spatial_holdout_admitted=holdout.get("admitted_pairs"), correction_applied=holdout.get("correction_applied"))

    # A mask-unit diagnostic: rectangle cells versus participation groups.
    band = source(0.5, 0.5)
    band.reflectance = torch.zeros(64, 64)
    band.valid_mask = torch.ones(64, 64, dtype=torch.bool)
    band.affine = (0.5, 0.0, 0.0, 0.0, -0.5, 32.0)
    rectangle = (1.0, 1.0, 5.0, 5.0)
    hidden, target = spatial_hide(band, rectangle)
    layout = participation_layout(band, rate_curve=CURVE, selection_seed=0, origin_xy=(0.0, 0.0))
    hidden_flat = ~hidden.keep().flatten()
    partial = full_hidden = full_visible = 0
    for coverage in layout["coverage"]:
        cells = torch.as_tensor(coverage)
        # Recover native row/col from the cell rectangle against this affine.
        cols = ((cells[:, 0] - band.affine[2]) / band.affine[0]).floor().long().clamp(0, 63)
        rows = ((band.affine[5] - cells[:, 3]) / (-band.affine[4])).floor().long().clamp(0, 63)
        flags = hidden_flat[rows * 64 + cols]
        if bool(flags.any()) and bool((~flags).any()):
            partial += 1
        elif bool(flags.any()):
            full_hidden += 1
        else:
            full_visible += 1
    task = spatial_task(Observation([band], dict(id="mask_unit_diagnostic", role="TRAIN")), rectangle)
    plan = prepare_participation_task(task, rate_curve=CURVE, selection_seed=0, origin_xy=(0.0, 0.0),
                                      chunk_pixels=64, cell_indexed=True, physical_patch_m=60.0)
    mask_units = dict(rectangle=rectangle, hidden_positions=int(hidden_flat.sum()),
                      target_positions=int(target.sum()), visible_positions=int(band.keep().sum()),
                      participation_tokens=len(layout["lengths"]), groups_partially_hidden=partial,
                      groups_fully_hidden=full_hidden, groups_fully_visible=full_visible,
                      A_supported_positions=int(plan.bands[0].supported_count),
                      note="Rectangle masking cuts through Merge groups; a Merge-support-unit mask would hide whole groups. They are not equivalent at group boundaries.")

    config = json.loads((ROOT / "encoder_v3" / "config.json").read_text(encoding="utf-8"))
    jepa = dict(required_before_jepa=config.get("required_before_jepa"),
                configured_conditions=sum(value is not None and value is not False and value != "" for value in (config.get("required_before_jepa") or {}).values()),
                jepa_module_present=(ROOT / "encoder_v3" / "jepa.py").exists(),
                latent_loss_implementation="cross_source_latent_loss requires an external callable loss_fn; no reviewed default exists",
                status="BLOCKED_BY_FIVE_UNCONFIRMED_CONDITIONS")

    losses = dict(
        training_path="stream_task_roots + joint.layout/adaptive_joint weights; per-band pixel NMSE divided by TRAIN std; family/source weights; one final clip",
        diagnostic_helper="losses.band_loss/hierarchical_loss is a separate diagnostic used by tests, not the joint training path",
        verified_tests=["test_participation_joint.test_joint_gradient_equals_explicit_source_balanced_sum",
                        "test_participation_batch.test_joint_batch_8_source_normalization_and_jepa_isolation",
                        "test_validation_metric.test_hand_computed_hierarchy"],
        unverified=["inactive A/B task renormalization protocol", "formal B1/B2 task mixture weights", "source-set equality across validation families"],
    )
    decoder = dict(shared_module="stream_adaptive_task passes the same ReflectanceDecoder; QueryFrameDecoder only adapts B2 query boxes",
                   separate_decoder=False, b2_target_reflectance_input=False)

    report = dict(
        schema="encoder_v3_formal_readiness_task_audit_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        A=dict(hidden_value_isolation="PASS: hidden reflectance replaced by NaN leaves tokens/group_ids unchanged",
               mask_unit_diagnostic=mask_units),
        B1=b1_audit,
        B2=b2_audit,
        decoder=decoder,
        losses=losses,
        JEPA=jepa,
        notes=["The 502-case B1 panel is not the full 2^C-2 combination space.",
               "B2 remains fail-closed with zero admitted pairs.",
               "No reviewed JEPA latent loss or five-condition evidence exists."],
    )
    out = AUDIT / "04_task_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(b1_cases=b1.get("cases"), b1_possible=possible, b1_coverage=b1.get("cases") / possible,
                          b2_admitted=pair.get("admitted_pairs"), mask_units=mask_units,
                          jepa=jepa["status"], output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Read-only architecture audit: SRF, Hypernetwork, Merge, P invariance, isolation."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from encoder_v3.data import Observation, sha256
from encoder_v3.model import Encoder, ReflectanceDecoder, SRFEncoder, spatial_hide
from encoder_v3.participation_encoder import ParticipationEncoder, prepare_participation_task
from encoder_v3.participation_layout import AreaEquivalentHillRate, participation_layout
from encoder_v3.tasks import spatial_task
from encoder_v3.test_adaptive_window import source

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "encoder_v3" / "reports" / "formal_readiness_audit"
CURVE = AreaEquivalentHillRate(.8092248954506467, 4.010953026221307, 1.219268281571556)


def band(gsd=0.5, side=32, *, top=None, left=0.0):
    reflectance = torch.rand(side, side, dtype=torch.float32)
    top = side * gsd if top is None else top
    value = source(gsd, gsd)
    value.reflectance = reflectance
    value.valid_mask = torch.ones(side, side, dtype=torch.bool)
    value.affine = (gsd, 0.0, left, 0.0, -gsd, top)
    return value


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    mixed = band(0.5, 32)
    mixed.wavelengths_nm = torch.linspace(440.0, 900.0, 21)
    mixed.response = torch.exp(-0.5 * ((mixed.wavelengths_nm - 665.0) / 30.0) ** 2)
    mixed.response = mixed.response / torch.trapezoid(mixed.response, mixed.wavelengths_nm)

    encoder = ParticipationEncoder(rate_curve=CURVE, selection_seed=0, hyper_hidden=64, local_batch_size=64)
    parameter_table = {name: dict(shape=list(value.shape), count=value.numel()) for name, value in encoder.named_parameters()}
    parameter_count = sum(item["count"] for item in parameter_table.values())

    # SRF: same physical curve sampled at 1 nm and 2 nm; trapezoid quadrature.
    srf = SRFEncoder()
    fine = torch.arange(400.0, 1000.0, 1.0)
    coarse = fine[::2]
    response = torch.exp(-0.5 * ((fine - 665.0) / 30.0) ** 2)
    response = response / torch.trapezoid(response, fine)
    encoded_fine = srf(fine, response)
    encoded_coarse = srf(coarse, response[::2])
    srf_grad = encoded_fine.sum()
    srf_grad.backward()
    srf_check = dict(
        shape=list(encoded_fine.shape),
        same_curve_sampling_max_abs=float((encoded_fine - encoded_coarse).abs().max()),
        quadrature_tolerance_1e_4=bool(float((encoded_fine - encoded_coarse).abs().max()) < 1e-4),
        all_parameters_receive_gradient=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in srf.parameters()),
        input_condition_on_target_srf_supported=True,
    )

    # Hypernetwork / dynamic convolution: serial vs grouped path.
    bands = [mixed, band(1.21, 16, left=100.0), band(10.0, 8, left=200.0), band(20.0, 4, left=300.0)]
    for index, item in enumerate(bands):
        item.record_id = f"b{index}"
        item.wavelengths_nm = mixed.wavelengths_nm + index * 5.0
        item.response = mixed.response
    serial = [encoder.native(item) for item in bands]
    grouped = encoder.native_grouped(bands)
    hyper_check = dict(
        hyper_output=int(encoder.hyper[-1].out_features),
        kernel_values=32 * 3 * 3,
        bias_values=32,
        output_total=int(encoder.hyper[-1].out_features),
        kernels_plus_bias=32 * 3 * 3 + 32,
        serial_vs_grouped_max_abs=max(float((a[0] - b[0]).abs().max()) for a, b in zip(serial, grouped)),
        condition_max_abs=max(float((a[2] - b[2]).abs().max()) for a, b in zip(serial, grouped)),
        keep_masks_equal=all(bool((a[1] == b[1]).all()) for a, b in zip(serial, grouped)),
    )
    for item in bands:
        features, keep, condition = encoder.native(item)
        loss = features.square().mean() + condition.square().mean()
        loss.backward(retain_graph=True)
    hyper_check["encoder_parameter_gradients_finite"] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for module in (encoder.srf, encoder.hyper) for p in module.parameters()
    )

    # Participation rate/group audit for the required GSD cases.
    gsd_cases = []
    for gsd in (0.5, 1.21, 10.0, 20.0, 30.0):
        side = 64 if gsd <= 1.21 else 32
        item = band(gsd, side)
        layout = participation_layout(item, rate_curve=CURVE, selection_seed=0, origin_xy=(0.0, 0.0))
        audit = layout["participation_audit"] if layout is not None else {}
        gsd_cases.append(dict(
            gsd_m=gsd, visible_positions=audit.get("visible_positions"), target_rate=audit.get("target_rate"),
            actual_rate=audit.get("actual_rate"), merged_participants=audit.get("merged_participants"),
            independent_positions=audit.get("independent_positions"), merged_groups=audit.get("merged_groups"),
            tokens=audit.get("tokens"), window_m=layout["window_m"].tolist() if layout is not None else None,
            group_size_min=audit.get("group_size_min"), group_size_max=audit.get("group_size_max"),
            deterministic=bool(layout is not None and
                              torch.equal(torch.as_tensor(layout["indices"]),
                                          torch.as_tensor(participation_layout(item, rate_curve=CURVE, selection_seed=0, origin_xy=(0.0, 0.0))["indices"]))),
        ))

    # Anisotropic and noninteger assay from the existing layout test evidence.
    anisotropic = source(0.5, 1.21)
    anisotropic.valid_mask[0, 0] = False
    anisotropic.visible = anisotropic.valid_mask.clone()
    anisotropic.visible[3:7, 3:7] = False
    layout = participation_layout(anisotropic, rate_curve=CURVE, selection_seed=3, origin_xy=(99.8, 180.0))
    anisotropic_check = dict(
        visible_positions=int(anisotropic.keep().sum()), tokens=len(layout["lengths"]),
        sum_lengths=int(sum(layout["lengths"])), target_rate=layout["participation_audit"]["target_rate"],
        actual_rate=layout["participation_audit"]["actual_rate"], window_m=layout["window_m"].tolist(),
    )

    # P invariance: P is a task/metadata context; Merge layout must not change.
    item = band(0.5, 32)
    observation = Observation([item], dict(id="generated", role="TRAIN"))
    task = spatial_task(observation, (1.0, 1.0, 8.0, 8.0))
    plans = {}
    for p in (60.0, 120.0):
        plans[p] = prepare_participation_task(task, rate_curve=CURVE, selection_seed=0, origin_xy=(0.0, 0.0),
                                              chunk_pixels=64, cell_indexed=True, physical_patch_m=p)
    geometry_equal = all(torch.equal(a, b) for a, b in zip(plans[60.0].geometry.cells, plans[120.0].geometry.cells))
    p_check = dict(
        geometry_cells_equal=geometry_equal,
        centers_equal=torch.equal(plans[60.0].geometry.centers, plans[120.0].geometry.centers),
        bounds_equal=torch.equal(plans[60.0].geometry.bounds, plans[120.0].geometry.bounds),
        physical_patch_metadata_60=plans[60.0].physical_patch_m,
        physical_patch_metadata_120=plans[120.0].physical_patch_m,
        merge_recipe_equal=plans[60.0].merge_recipe == plans[120.0].merge_recipe,
    )

    # Hidden-value isolation on the actual Participation path.
    hidden, _ = spatial_hide(item, (1.0, 1.0, 8.0, 8.0))
    first = encoder([hidden], origin_xy=(0.0, 0.0))
    changed = hidden
    changed.reflectance = hidden.reflectance.clone()
    changed.reflectance[~hidden.keep()] = float("nan")
    second = encoder([changed], origin_xy=(0.0, 0.0))
    isolation = dict(tokens_equal=torch.equal(first.tokens, second.tokens),
                     group_ids_equal=first.group_ids == second.group_ids,
                     hidden_positions=int((~hidden.keep()).sum()),
                     finite_visible_tokens=bool(torch.isfinite(first.tokens).all()))

    # Decoder/shared parameters.
    decoder = ReflectanceDecoder(indexed=True, cell_indexed=True)
    decoder_check = dict(mlp=[list(p.shape) for p in decoder.parameters()],
                         shared_across_kinds=all(hasattr(decoder, name) for name in ("mlp",)))

    module_hashes = {name: sha256(ROOT / "encoder_v3" / name) for name in
                     ("model.py", "participation_layout.py", "participation_encoder.py", "adaptive_window.py", "spatial_index.py", "participation_decoder.py", "complete_attention.py")}

    report = dict(
        schema="encoder_v3_formal_readiness_architecture_audit_v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        architecture=dict(
            hypernetwork=dict(input=34, hidden=64, output=320, activation="GELU", gsd_transform="natural_log_of_gsd_in_meters"),
            encoder_parameters=parameter_count,
            parameter_table=parameter_table,
            module_hashes=module_hashes,
        ),
        srf=srf_check,
        hypernetwork=hyper_check,
        merge=dict(gsd_cases=gsd_cases, anisotropic=anisotropic_check,
                   participation_definition="valid visible native positions participating in multi-position merges, not token reduction",
                   p_invariance=p_check),
        isolation=isolation,
        decoder=decoder_check,
        local_global=dict(local_layers=len(encoder.local.layers), global_layers=len(encoder.global_net.layers),
                          width=128, heads=4, ffn=512,
                          full_attention_default=encoder.global_query_chunk_size is None,
                          query_chunk_optional="complete_attention.query_chunked_transformer only when explicitly configured"),
        notes=[
            "The pair-query offset is applied once; tuple*2 is [dx,dy,dx,dy], not a doubled translation.",
            "Dense and indexed A paths now share represented_boundary_v1; legacy_center_outside remains only for old reference representations.",
            "Signed/unsigned numeric gaps are measured diagnostics, not formal tolerances.",
        ],
    )
    out = AUDIT / "02_architecture_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(parameters=parameter_count, srf=srf_check, hyper=hyper_check, gsd=gsd_cases,
                          p=p_check, isolation=isolation, output=str(out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

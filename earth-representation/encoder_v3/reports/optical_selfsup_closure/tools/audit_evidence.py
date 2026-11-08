"""P0 E0 evidence: group partition, hidden-value isolation, batch independence, support.

Uses two real TRAIN members (a 128-pixel four-band Sentinel-2 scene and a small
Landsat scene) and never steps an optimizer.

    python reports/optical_selfsup_closure/tools/audit_evidence.py --config encoder_v3/config_optical_selfsup_v1.json --output ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from encoder_v3.formal_schedule import _mask_rule, _rectangle_in, observation_bounds  # noqa: E402
from encoder_v3.model import BandObservation, ReflectanceDecoder  # noqa: E402
from encoder_v3.participation_encoder import ParticipationEncoder, prepare_participation_task  # noqa: E402
from encoder_v3.participation_layout import participation_layout  # noqa: E402
from encoder_v3.selfsup_preflight import build_context, load_config  # noqa: E402
from encoder_v3.tasks import fixed_spectral_task, hidden_band_indices, spatial_task  # noqa: E402


def to_device(band, device):
    return BandObservation(band.reflectance.to(device), band.valid_mask.to(device), band.wavelengths_nm.to(device),
                           band.response.to(device), band.affine, band.crs, band.record_id, band.visible)


def masking(band, rectangle):
    cells = band.cells()
    mask = cells.new_tensor(rectangle)
    intersects = (torch.minimum(cells[:, 2:], mask[2:]) - torch.maximum(cells[:, :2], mask[:2])).gt(0).all(1)
    contained = (cells[:, :2] >= mask[:2]).all(1) & (cells[:, 2:] <= mask[2:]).all(1)
    return intersects, contained


def partition(band, *, curve, seed):
    layout = participation_layout(band, rate_curve=curve, selection_seed=seed, origin_xy=(0.0, 0.0))
    keep = band.keep().flatten()
    indices = torch.as_tensor(layout["indices"])
    return dict(visible_positions=int(keep.sum()), groups=int(len(layout["lengths"])),
                group_positions=int(len(indices)), unique_positions=int(len(set(indices.tolist()))),
                merged_groups=int(sum(1 for key in layout["group_ids"] if key[0] == "merge")),
                singleton_groups=int(sum(1 for key in layout["group_ids"] if key[0] == "native")),
                tokens=int(len(layout["centers"])),
                every_visible_position_exactly_once=(len(indices) == len(set(indices.tolist())) == int(keep.sum())),
                covers_only_visible_positions=bool(keep[indices].all()), max_group=int(max(layout["lengths"])))


def set_equals_hidden(task):
    """A: every supervised target lies inside the masked rectangle. B1: no target band is visible."""
    if task.kind == "B1":
        visible = {band.record_id for band in task.visible}
        return not (visible & {target.band.record_id for target in task.targets})
    found = []
    for target in task.targets:
        positions = torch.where((target.mask & target.band.valid_mask).flatten())[0]
        found.append(positions)
    return bool(found)


def decoder_support(plan, task, representation, encoder, decoder, *, chunk_pixels):
    """Mirror stream_task_roots: count decoder support per band without backward."""
    readout = type(representation)(tokens=representation.tokens.detach(), cells=representation.cells,
                                   centers=representation.centers, bounds=representation.bounds, valid=representation.valid,
                                   crs=representation.crs, spatial_index=None)
    readout.a_adjacency_rule = representation.a_adjacency_rule
    mask = None if task.rectangle is None else torch.tensor(task.rectangle, dtype=torch.float64,
                                                            device=representation.tokens.device)
    rows = []
    for band_plan, target in zip(plan.bands, task.targets):
        band = target.band
        condition = encoder.srf(band.wavelengths_nm, band.response).detach()
        bounds = band.cells()
        supported = 0
        for ids in band_plan.indices.split(chunk_pixels):
            prediction, keep = decoder(readout, bounds[ids.to(bounds.device)], condition, kind=task.kind, mask_bounds=mask)
            supported += int(keep.sum())
        rows.append(dict(band=band.record_id, plan=int(band_plan.supported_count), decoder=int(supported),
                         matches_plan=int(supported) == int(band_plan.supported_count)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="E0 architecture evidence on real TRAIN members")
    parser.add_argument("--config", type=Path, default=ROOT / "encoder_v3" / "config_optical_selfsup_v1.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    context = build_context(args.config, device=args.device)
    dataset, curve = context["train"], context["curve"]
    rule = _mask_rule(config["sampling"]["mask_rule"])
    seed = int(config["sampling"]["seed"])
    picks = []
    for sensor, bands, side in (("S2", 4, 128), ("L8", 6, 44)):
        found = [index for index, row in enumerate(dataset.rows) if row["sensor"] == sensor and len(row["bands"]) == bands
                 and max(max(band["shape"]) for band in row["bands"]) == side]
        picks.append((sensor, found[0], len(found)))
    torch.manual_seed(int(config["model"]["init_seed"]))
    encoder = ParticipationEncoder(rate_curve=curve, selection_seed=context["selection_seed"],
                                   hyper_hidden=int(config["model"]["hypernetwork_hidden"]),
                                   local_batch_size=int(config["model"]["local_batch_size"]), packed_local_output=False)
    encoder.to(args.device).eval()
    decoder = ReflectanceDecoder(indexed=True).to(args.device).eval()
    chunk = int(config["formal_step"]["chunk_pixels"])
    records = []
    for sensor, index, pool in picks:
        observation = dataset[index]
        left, bottom, right, top = observation_bounds(observation)
        rectangle = _rectangle_in(left, bottom, right, top, 60.0, seed=seed, update=0, identity=observation.identifier,
                                  stream="A_rectangle_optical_selfsup_v1", rule=rule)
        hidden = sorted(hidden_band_indices(len(observation.bands), seed=seed, update=0, observation_id=observation.identifier))
        hidden_ids = [observation.bands[position].record_id for position in hidden]
        spatial = spatial_task(observation, rectangle)
        spectral = fixed_spectral_task(observation, hidden_ids)
        record = dict(sensor=sensor, observation=observation.identifier, member_pool=pool, bands=len(observation.bands),
                      shape=list(observation.bands[0].reflectance.shape), hidden_band_ids=hidden_ids, P_m=60.0)
        record["partition"] = [partition(band, curve=curve, seed=context["selection_seed"]) for band in observation.bands[:2]]
        visible_ids = [band.record_id for band in spectral.visible]
        target_ids = [target.band.record_id for target in spectral.targets]
        record["B1_band_isolation"] = dict(visible=visible_ids, targets=target_ids, disjoint=not (set(visible_ids) & set(target_ids)),
                                           leaves_visible=bool(visible_ids),
                                           every_band_accounted=(set(visible_ids) | set(target_ids)) == {band.record_id for band in observation.bands})
        intersects, _ = masking(observation.bands[0], rectangle)
        poisoned = []
        for band in spatial.visible:
            reflectance = band.reflectance.clone()
            reflectance.flatten()[intersects & band.valid_mask.flatten()] = float("nan")
            poisoned.append(BandObservation(reflectance, band.valid_mask, band.wavelengths_nm, band.response, band.affine,
                                            band.crs, band.record_id, band.visible))
        moved = [to_device(band, args.device) for band in observation.bands]
        moved_poisoned = [to_device(band, args.device) for band in poisoned]
        moved_spatial = [to_device(band, args.device) for band in spatial.visible]
        moved_spectral = [to_device(band, args.device) for band in spectral.visible]
        with torch.no_grad():
            base = encoder.forward(moved_spatial, origin_xy=(left, bottom))
            injected = encoder.forward(moved_poisoned, origin_xy=(left, bottom))
            single = encoder.forward(moved, origin_xy=(left, bottom))
            paired = encoder.forward_batch([(moved, (left, bottom)), (moved, (left, bottom))])
            spectral_representation = encoder.forward(moved_spectral, origin_xy=(left, bottom))
        record["A_hidden_value_isolation"] = dict(
            hidden_positions=int((intersects & observation.bands[0].valid_mask.flatten()).sum()),
            tokens_equal=bool(torch.equal(base.tokens, injected.tokens)),
            group_ids_equal=bool(base.group_ids == injected.group_ids),
            finite_visible_tokens=bool(torch.isfinite(injected.tokens).all()),
        )
        record["batch_independence"] = dict(
            duplicate_pair_token_max_abs_difference=float((paired[0].tokens - paired[1].tokens).abs().max()),
            vs_single_token_max_abs_difference=float((paired[0].tokens - single.tokens).abs().max()),
            duplicate_pair_group_ids_equal=bool(paired[0].group_ids == paired[1].group_ids),
        )
        support = {}
        plans = {}
        for label, task, representation in (("A", spatial, base), ("B1", spectral, spectral_representation)):
            plan = prepare_participation_task(task, rate_curve=curve, selection_seed=context["selection_seed"],
                                              origin_xy=(left, bottom), chunk_pixels=chunk, cell_indexed=False,
                                              retain_layouts=False, physical_patch_m=60.0)
            plans[label] = plan
            if plan is None:
                support[label] = dict(status="NO_PLAN")
                continue
            groups = [torch.as_tensor(band.indices) for band in plan.bands]
            support[label] = dict(
                status="OK", planned=int(sum(len(band.indices) for band in plan.bands)),
                supported=int(sum(band.supported_count for band in plan.bands)),
                supervised_targets_are_the_hidden_positions=set_equals_hidden(task),
                supervised_positions_disjoint_per_band=all(len(group) == len(set(group.tolist())) for group in groups),
            )
            if label == "A":
                support[label]["decoder_equals_plan"] = decoder_support(plan, task, representation, encoder, decoder, chunk_pixels=chunk)
            else:
                support[label]["decoder_equals_plan"] = decoder_support(plan, task, representation, encoder, decoder, chunk_pixels=chunk)
        record["support"] = support
        records.append(record)
        del observation, base, injected, single, paired, spectral_representation, moved, moved_poisoned
    report = dict(status="REAL_DATA_ARCHITECTURE_EVIDENCE_NO_OPTIMIZER_STEP", device=args.device, members=records,
                  optimizer_updates=0, parameter_updates=0,
                  scope="two real TRAIN members: partition uniqueness, hidden-value isolation, B1 band isolation, batch independence, plan/decoder support equality")
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload if args.output is None else json.dumps({key: report[key] for key in ("status", "optimizer_updates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

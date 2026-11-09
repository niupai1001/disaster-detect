"""Exact exposure/coverage accounting for the frozen E0 sampling recipe.

Never trains and never reads VAL. Prints (and optionally writes) the report used
by 02_data_sampling_report.json: member exposure counts over every logical
update, source totals, P counts, A/B1 task counts, B1 hidden-count distribution,
token-size percentiles over the pool, and a bounded real supervision-ratio probe.

    python -m encoder_v3.sampling_coverage --config encoder_v3/config_optical_selfsup_v1.json --output ...
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .formal_schedule import FAMILIES, optical_train_pool
from .participation_layout import AreaEquivalentHillRate
from .selfsup_preflight import build_context, configuration_sha256
from .tasks import hidden_band_indices


def token_estimate(row, curve, merge_sample_pixels=1.0):
    """Analytic token estimate: (1-r)*pixels + r*pixels/group_size, per band."""
    total = 0.0
    for band in row["bands"]:
        h, w = band["shape"]
        a, _, _, _, e, _ = band["transform"]
        gx, gy = abs(a), abs(e)
        rate = float(curve(gx, gy))
        span = 1.0 / (1.0 - rate)
        group = max(1.0, span * span * merge_sample_pixels)
        total += h * w * ((1.0 - rate) + rate / group)
    return int(round(total))


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def distribution(values):
    values = list(values)
    return dict(count=len(values), min=min(values) if values else None, p1=percentile(values, 0.01),
                p25=percentile(values, 0.25), median=percentile(values, 0.5), p75=percentile(values, 0.75),
                p99=percentile(values, 0.99), max=max(values) if values else None,
                mean=(sum(values) / len(values)) if values else None, zero=int(sum(1 for value in values if value == 0)))


def coverage(context, *, supervision_updates=6):
    config, schedule, pool = context["config"], context["schedule"], context["pool"]
    curve = context["curve"]
    rows = {row["id"]: row for row in context["train"].rows}
    family_of = {member[0]: member[1] for member in pool["members"]}
    band_of = {member[0]: member[2] for member in pool["members"]}
    exposures = Counter()
    by_family = Counter()
    P_counts = Counter()
    task_counts = Counter()
    hidden_counts = Counter()
    single_vs_multi = Counter()
    hidden_by_band_count = defaultdict(Counter)
    for index in range(len(schedule)):
        patch, slots = schedule.members(index)
        P_counts[str(patch)] += 1
        for family, identifier in slots:
            exposures[identifier] += 1
            by_family[family] += 1
            band_count = band_of[identifier]
            hidden = hidden_band_indices(band_count, seed=schedule.seed, update=index, observation_id=identifier)
            hidden_counts[len(hidden)] += 1
            single_vs_multi["single" if len(hidden) == 1 else "multi"] += 1
            hidden_by_band_count[band_count][len(hidden)] += 1
            task_counts["A"] += 1
            task_counts["B1"] += 1
    values = [exposures.get(identifier, 0) for identifier in family_of]
    tokens = {identifier: token_estimate(rows[identifier], curve) for identifier in family_of}
    supervision = supervision_probe(context, updates=supervision_updates)
    return dict(
        status="EXACT_RECIPE_ACCOUNTING",
        experiment=config["experiment"],
        configuration_sha256=configuration_sha256(context["config_path"]),
        recipe_sha256=context["recipe_sha256"],
        updates=len(schedule),
        release_members=len(context["train"].rows),
        pool=dict(members=len(pool["members"]), sha256=pool["sha256"],
                  families={family: len(values) for family, values in pool["families"].items()},
                  band_counts=dict(sorted(Counter(band_of.values()).items())),
                  excluded=pool["excluded"], excluded_by_sensor=pool["excluded_by_sensor"],
                  max_native_pixels=pool["max_native_pixels"]),
        exposures=dict(per_member=distribution(values), by_family=dict(sorted(by_family.items())),
                       totals=dict(slots=int(sum(by_family.values())), expected_unique_members=len(family_of)),
                       covered_members=int(sum(1 for value in values if value > 0)),
                       by_sensor=dict(sorted(Counter(family_of[identifier] for identifier in family_of).items()))),
        P_counts=dict(sorted(P_counts.items())),
        task_counts=dict(sorted(task_counts.items())),
        B1_hidden=dict(single_vs_multi=dict(sorted(single_vs_multi.items())),
                       by_hidden_count=dict(sorted(hidden_counts.items())),
                       by_band_count={str(key): dict(sorted(value.items())) for key, value in sorted(hidden_by_band_count.items())}),
        token_estimate=dict(per_member=distribution(list(tokens.values())), max=int(max(tokens.values())),
                            sum_squares_upper_bound=int(sum(value * value for value in tokens.values())),
                            method="analytic (1-r)+r/group per band from the frozen curve; measured sample in supervision probe"),
        supervision=supervision,
        optimizer_updates=0,
    )


def supervision_probe(context, *, updates):
    """Bounded real geometry probe: supervised target support and token counts."""
    config = context["config"]
    curve = context["curve"]
    schedule = context["schedule"]
    records = []
    for index in range(min(updates, len(schedule))):
        packet = schedule.prepare(index, cell_indexed=config["model"]["cell_indexed_decoder"], retain_layouts=False)
        for view in packet["views"]:
            task = view.task
            if view.plan is None:
                records.append(dict(kind=task.kind, status="NO_PLAN", P_m=packet["physical_patch_m"],
                                    planned=0, supported=0))
                continue
            planned = sum(len(band.indices) for band in view.plan.bands)
            supported = sum(band.supported_count for band in view.plan.bands)
            records.append(dict(kind=task.kind, status="OK", P_m=packet["physical_patch_m"],
                                planned=int(planned), supported=int(supported),
                                ratio=(supported / planned) if planned else None,
                                bands=len(task.targets)))
    by_kind = {}
    for kind in ("A", "B1"):
        group = [record for record in records if record["kind"] == kind and record["ratio"] is not None]
        by_kind[kind] = dict(cases=len(group),
                             ratio=distribution([record["ratio"] for record in group]),
                             planned=sum(record["planned"] for record in group),
                             supported=sum(record["supported"] for record in group))
    return dict(updates=min(updates, len(schedule)), cases=len(records), by_kind=by_kind,
                method="real native geometry on the first frozen updates; not repeated over all updates",
                records=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exact E0 sampling exposure and coverage accounting")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_optical_selfsup_v1.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--supervision-updates", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    context = build_context(args.config, device=args.device)
    report = coverage(context, supervision_updates=args.supervision_updates)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload if args.output is None else json.dumps({key: report[key] for key in ("status", "updates", "pool", "exposures", "P_counts", "task_counts")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

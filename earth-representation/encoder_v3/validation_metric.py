"""Fixed hierarchical validation scalar; refuses silently redefined weights.

Confirmed form: S=(A+B)/2, B=(B1+B2)/2. Per case: mean over available target
bands (the caller supplies that case loss). A: mean cases per observation,
mean observations per source, equal sources. B1: single-hidden and multi-hidden
halves; multi-hidden is an equal mean over hidden counts 2..C-1 after a mean
over cases within each count. B2: same source hierarchy as A. P repeats are
cases. A missing task, count, source or nonfinite loss makes the scalar
unavailable instead of substituting zero or dropping a cell.
"""
from collections import defaultdict


class ValidationUnavailable(ValueError):
    pass


def _number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationUnavailable(f"{label}: finite loss required") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValidationUnavailable(f"{label}: nonfinite loss")
    return number


def _mean(values, label):
    if not values:
        raise ValidationUnavailable(f"{label}: no cases")
    return sum(values) / len(values)


def _validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValidationUnavailable("nonempty validation cases required")
    seen = set()
    normalized = []
    for case in cases:
        identifier = case.get("case_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValidationUnavailable("explicit case_id required")
        if identifier in seen:
            raise ValidationUnavailable(f"duplicate case_id: {identifier}")
        seen.add(identifier)
        kind = case.get("kind")
        if kind not in ("A", "B1", "B2"):
            raise ValidationUnavailable(f"{identifier}: unknown kind")
        source = case.get("source")
        observation = case.get("observation")
        if not isinstance(source, str) or not source or not isinstance(observation, str) or not observation:
            raise ValidationUnavailable(f"{identifier}: explicit source/observation required")
        status = case.get("status", "OK")
        if status != "OK":
            raise ValidationUnavailable(f"{identifier}: case status {status}")
        loss = _number(case.get("loss"), identifier)
        hidden = case.get("hidden_count")
        band_count = case.get("band_count")
        if kind == "B1":
            if type(hidden) is not int or hidden < 1:
                raise ValidationUnavailable(f"{identifier}: B1 hidden_count required")
            if type(band_count) is not int or band_count < 3:
                raise ValidationUnavailable(f"{identifier}: B1 formal metric needs band_count>=3 for multi-hidden")
            if hidden > band_count - 1:
                raise ValidationUnavailable(f"{identifier}: B1 must leave a visible band")
        else:
            hidden = None
            band_count = None
        normalized.append(dict(
            case_id=identifier, kind=kind, source=source, observation=observation,
            loss=loss, hidden_count=hidden, band_count=band_count, P_m=case.get("P_m"),
        ))
    return sorted(normalized, key=lambda item: item["case_id"])


def _observation_losses(cases, kind):
    by_source = defaultdict(lambda: defaultdict(list))
    for case in cases:
        if case["kind"] == kind:
            by_source[case["source"]][case["observation"]].append(case)
    result = {}
    for source, observations in by_source.items():
        result[source] = {}
        for observation, items in observations.items():
            losses = [item["loss"] for item in items]
            if kind == "A":
                result[source][observation] = _mean(losses, f"A {source}/{observation}")
                continue
            if kind == "B2":
                result[source][observation] = _mean(losses, f"B2 {source}/{observation}")
                continue
            band_counts = {item["band_count"] for item in items}
            if len(band_counts) != 1:
                raise ValidationUnavailable(f"B1 {source}/{observation}: inconsistent band_count")
            band_count = next(iter(band_counts))
            single = [item["loss"] for item in items if item["hidden_count"] == 1]
            multi = [item for item in items if item["hidden_count"] >= 2]
            if not single:
                raise ValidationUnavailable(f"B1 {source}/{observation}: single-hidden cases missing")
            if not multi:
                raise ValidationUnavailable(f"B1 {source}/{observation}: multi-hidden cases missing")
            expected = list(range(2, band_count))
            counts = defaultdict(list)
            for item in multi:
                counts[item["hidden_count"]].append(item["loss"])
            missing = sorted(set(expected) - set(counts))
            if missing:
                raise ValidationUnavailable(f"B1 {source}/{observation}: missing hidden counts {missing}")
            count_means = [_mean(counts[count], f"B1 {source}/{observation}/h{count}") for count in expected]
            result[source][observation] = 0.5 * _mean(single, f"B1 {source}/{observation}/single") + 0.5 * _mean(count_means, f"B1 {source}/{observation}/multi")
    return result


def _source_means(observation_losses):
    return {source: _mean(list(observations.values()), f"source {source}") for source, observations in observation_losses.items()}


def aggregate_validation(cases, *, require_b2=True):
    normalized = _validate_cases(cases)
    family_observations = {kind: _observation_losses(normalized, kind) for kind in ("A", "B1", "B2")}
    family = {kind: _source_means(values) if values else {} for kind, values in family_observations.items()}
    for kind in ("A", "B1"):
        if not family[kind]:
            raise ValidationUnavailable(f"{kind}: no evaluation source")
    if not family["B2"]:
        if require_b2:
            raise ValidationUnavailable("B2: no admitted/evaluated source; formal scalar unavailable")
        b = None
    else:
        if set(family["B1"]) != set(family["B2"]):
            raise ValidationUnavailable("B1/B2 source sets differ; B family unavailable")
        b = 0.5 * _mean(list(family["B1"].values()), "B1") + 0.5 * _mean(list(family["B2"].values()), "B2")
    a = _mean(list(family["A"].values()), "A")
    score = None if b is None else 0.5 * (a + b)
    return dict(
        status="OK" if score is not None else "B2_UNAVAILABLE",
        S=score,
        A=a,
        B=b,
        B1=None if not family["B1"] else _mean(list(family["B1"].values()), "B1"),
        B2=None if not family["B2"] else _mean(list(family["B2"].values()), "B2"),
        counts={kind: sum(len(items) for items in observations.values()) for kind, observations in family_observations.items()},
        sources={kind: sorted(values) for kind, values in family.items()},
    )

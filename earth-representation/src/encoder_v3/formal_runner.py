"""Gated single-route runner: schedule -> views -> update transaction -> next cursor.

It composes the existing frozen-data loaders, route_step transaction and fixed
validation collector. It never starts by itself, never invents control values
and refuses the JEPA phase until the latent views/signatures are supplied by a
reviewed caller.

The next schedule entry is prepared exactly once (one-step lookahead) and the
early-stop state is committed inside every checkpoint, so a resumed run makes
the same decisions as an uninterrupted one.
"""
from __future__ import annotations

import math
import time

from .gates import require_training
from .plan_prefetch import move_views
from .route_step import execute_route_update
from .validation_metric import ValidationUnavailable

_EARLY_KEYS = ("best_validation", "stale_validations", "validation_evaluations")


def panel_validator(panels, *, cell_indexed=False, retain_layouts=False, metric_key="S"):
    """Adapt FrozenValidationPanels.collect to the route-step validation hook."""
    def validate(modules, completed_update):
        result = panels.collect(modules["encoder"], modules["decoder"], cell_indexed=cell_indexed, retain_layouts=retain_layouts)
        metric = result.get("metric")
        if metric is None:
            raise ValidationUnavailable(result.get("metric_status") or "fixed validation scalar unavailable")
        value = metric.get(metric_key)
        if value is None:
            raise ValidationUnavailable(f"fixed validation scalar {metric_key} unavailable")
        return float(value)
    return validate


def early_stop_state(state, *, min_delta):
    """Validate a restored runner state or return the empty starting state."""
    if state is None:
        state = dict(best_validation=float("inf"), stale_validations=0, validation_evaluations=0)
    if not isinstance(state, dict) or any(key not in state for key in _EARLY_KEYS):
        raise ValueError("Complete runner early-stop state required")
    best = state["best_validation"]
    if isinstance(best, bool) or not isinstance(best, (int, float)) or not (math.isfinite(best) or best == float("inf")):
        raise ValueError("Finite best validation value required")
    stale, evaluations = state["stale_validations"], state["validation_evaluations"]
    if type(stale) is not int or stale < 0 or type(evaluations) is not int or evaluations < 0:
        raise ValueError("Nonnegative early-stop counters required")
    if not math.isfinite(float(min_delta)) or min_delta < 0:
        raise ValueError("Nonnegative early-stop min_delta required")
    return dict(best_validation=float(best), stale_validations=stale, validation_evaluations=evaluations, min_delta=float(min_delta))


def run_route(*, config, frozen_ids, progress, modules, optimizer, scheduler, schedule,
              directory, device, control, validation=None, expected_input_signature=None,
              gate=None, restored_early_stop=None):
    """Run one already-authorized route until a frozen limit or the schedule end."""
    if gate is None:
        gate = require_training
    gate(config)
    if progress.phase not in ("shared", "A+B"):
        raise ValueError("JEPA route requires reviewed latent views and signatures; use the gated branch initializer first")
    for module in modules.values():
        if type(module).__module__.split(".")[0] in ("unittest", "mock"):
            raise ValueError("Test doubles cannot enter a production route")
    required = {"max_updates", "max_wall_seconds", "cell_indexed", "retain_layouts"}
    if not isinstance(control, dict) or not required <= set(control) or set(control) - required - {"early_stop"}:
        raise ValueError("Explicit frozen runner controls required")
    early = control.get("early_stop")
    if early is not None and (not isinstance(early, dict) or set(early) != {"patience", "min_delta"}
                              or type(early["patience"]) is not int or early["patience"] < 1
                              or not isinstance(early["min_delta"], (int, float)) or early["min_delta"] < 0):
        raise ValueError("Explicit early-stop patience and nonnegative min_delta required")
    max_updates = control["max_updates"]
    max_wall = control["max_wall_seconds"]
    if type(max_updates) is not int or max_updates < 1 or max_updates > 10000 or type(max_wall) not in (int, float):
        raise ValueError("Positive update limit within the route cap and wall limit required")
    if max_wall <= 0 or type(control["cell_indexed"]) is not bool or type(control["retain_layouts"]) is not bool:
        raise ValueError("Positive wall limit and explicit geometry flags required")
    if not isinstance(schedule, object) or not hasattr(schedule, "prepare") or not hasattr(schedule, "__len__"):
        raise ValueError("Frozen schedule loader required")
    if early is not None and not callable(validation):
        raise ValueError("Early stopping requires the fixed validation callback")
    state = early_stop_state(restored_early_stop, min_delta=0.0 if early is None else early["min_delta"])
    delta = state["min_delta"]
    start_update = progress.logical_update
    started = time.perf_counter()

    def tracked_validation(modules_, completed_update):
        metric = validation(modules_, completed_update)
        if not isinstance(metric, (int, float)) or isinstance(metric, bool) or not math.isfinite(float(metric)):
            raise ValidationUnavailable("Finite fixed validation scalar required")
        state["validation_evaluations"] += 1
        if float(metric) < state["best_validation"] - delta:
            state["best_validation"] = float(metric)
            state["stale_validations"] = 0
        else:
            state["stale_validations"] += 1
        return float(metric)

    def stop(reason):
        return dict(status="ROUTE_STOPPED", progress=progress, updates=len(history), history=history,
                    wall_seconds=time.perf_counter() - started, reason=reason,
                    early_stop_state=dict(state))

    history = []
    packet = None
    while progress.logical_update < max_updates and progress.logical_update < len(schedule):
        if time.perf_counter() - started > max_wall:
            return stop("wall_limit")
        if early is not None and state["stale_validations"] >= early["patience"]:
            return stop("early_stop")
        if packet is None:
            packet = schedule.prepare(progress.logical_update, cell_indexed=control["cell_indexed"], retain_layouts=control["retain_layouts"])
        index = progress.logical_update
        signature = packet["AB_input_signature"]
        if index == start_update and expected_input_signature is not None and signature != expected_input_signature:
            raise ValueError("First schedule input differs from the checkpoint expectation")
        views = move_views(packet["views"], device)
        next_update = index + 1
        next_packet = schedule.prepare(next_update, cell_indexed=control["cell_indexed"], retain_layouts=control["retain_layouts"]) if next_update < len(schedule) else None
        next_signature = signature if next_packet is None else next_packet["AB_input_signature"]
        progress, metrics = execute_route_update(
            config=config, progress=progress, frozen_ids=frozen_ids, modules=modules,
            optimizer=optimizer, scheduler=scheduler, views=views,
            expected_input_signature=signature,
            sampler_after=dict(next_update=next_update, next_input_signature=next_signature),
            directory=directory, validation=None if validation is None else tracked_validation,
            gate=gate, runner_state=dict(state) if validation is not None else None,
        )
        history.append(metrics)
        if early is not None and state["stale_validations"] >= early["patience"]:
            return stop("early_stop")
        if time.perf_counter() - started > max_wall:
            return stop("wall_limit")
        if next_packet is None:
            break
        packet = next_packet
    return stop("update_limit" if progress.logical_update >= max_updates else "schedule_end")

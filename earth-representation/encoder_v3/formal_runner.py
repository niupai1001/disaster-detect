"""Gated single-route runner: schedule -> views -> update transaction -> next cursor.

It composes the existing frozen-data loaders, route_step transaction and fixed
validation collector. It never starts by itself, never invents control values
and refuses the JEPA phase until the latent views/signatures are supplied by a
reviewed caller.
"""
from __future__ import annotations

import time

from .gates import require_training
from .plan_prefetch import move_views
from .route_step import execute_route_update
from .validation_metric import ValidationUnavailable


def panel_validator(panels, *, cell_indexed=False, retain_layouts=False):
    """Adapt FrozenValidationPanels.collect to the route-step validation hook."""
    def validate(modules, completed_update):
        result = panels.collect(modules["encoder"], modules["decoder"], cell_indexed=cell_indexed, retain_layouts=retain_layouts)
        metric = result.get("metric")
        if metric is None:
            raise ValidationUnavailable(result.get("metric_status") or "fixed validation scalar unavailable")
        return float(metric["S"])
    return validate


def run_route(*, config, frozen_ids, progress, modules, optimizer, scheduler, schedule,
              directory, device, control, validation=None, expected_input_signature=None):
    """Run one already-authorized route until a frozen limit or the schedule end."""
    require_training(config)
    if progress.phase not in ("shared", "A+B"):
        raise ValueError("JEPA route requires reviewed latent views and signatures; use the gated branch initializer first")
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
    start_update = progress.logical_update
    started = time.perf_counter()
    history = []
    best_validation = float("inf")
    stale_validations = 0
    while progress.logical_update < max_updates and progress.logical_update < len(schedule):
        index = progress.logical_update
        packet = schedule.prepare(index, cell_indexed=control["cell_indexed"], retain_layouts=control["retain_layouts"])
        signature = packet["AB_input_signature"]
        if index == start_update and expected_input_signature is not None and signature != expected_input_signature:
            raise ValueError("First schedule input differs from the checkpoint expectation")
        views = move_views(packet["views"], device)
        next_update = index + 1
        if next_update < len(schedule):
            next_packet = schedule.prepare(next_update, cell_indexed=control["cell_indexed"], retain_layouts=control["retain_layouts"])
            next_signature = next_packet["AB_input_signature"]
        else:
            next_signature = signature
        progress, metrics = execute_route_update(
            config=config, progress=progress, frozen_ids=frozen_ids, modules=modules,
            optimizer=optimizer, scheduler=scheduler, views=views,
            expected_input_signature=signature,
            sampler_after=dict(next_update=next_update, next_input_signature=next_signature),
            directory=directory, validation=validation,
        )
        history.append(metrics)
        validation_result = metrics.get("validation") if isinstance(metrics, dict) else None
        if early is not None and isinstance(validation_result, dict) and validation_result.get("metric") is not None:
            metric = float(validation_result["metric"])
            if metric < best_validation - early["min_delta"]:
                best_validation = metric
                stale_validations = 0
            else:
                stale_validations += 1
                if stale_validations >= early["patience"]:
                    return dict(status="ROUTE_STOPPED", progress=progress, updates=len(history), history=history,
                                wall_seconds=time.perf_counter() - started, reason="early_stop")
        if time.perf_counter() - started > max_wall:
            return dict(status="ROUTE_STOPPED", progress=progress, updates=len(history), history=history,
                        wall_seconds=time.perf_counter() - started, reason="wall_limit")
    return dict(status="ROUTE_STOPPED", progress=progress, updates=len(history), history=history,
                wall_seconds=time.perf_counter() - started,
                reason="update_limit" if progress.logical_update >= max_updates else "schedule_end")

"""更新制训练的预算解析：把"停止步数"与"学习率调度长度"显式分开。

历史配置只有一个 ``segformer.train.max_steps``，它同时充当停止上限和 poly 日程
总长。直接把它改成 2000 会把学习率日程一起压缩（预热从 180 降到 100），
得到的不是旧 3600 步日程的前 2000 步。这里统一解析三个量：

- ``stop_updates``：成功更新到该步数就停（缺省回落到 ``max_steps``）；
- ``schedule_updates``：poly 日程总长（缺省等于停止步数，即历史行为）；
- ``report_steps``：额外保存权重的步（缺省只存停止步）。

两个键都写在配置里时必须一致地表达同一件事：``max_steps`` 若同时存在，只表示
停止上限，必须等于 ``stop_updates``，否则拒绝，避免"配置写着 2000、实际跑 3600"。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UpdateBudget:
    stop_updates: int
    schedule_updates: int
    eval_every_updates: int
    report_steps: tuple[int, ...]

    def warmup_updates(self, warmup_frac: float) -> int:
        """与历史实现同式：``max(1, int(schedule * warmup_frac))``。"""
        return max(1, int(self.schedule_updates * float(warmup_frac)))

    def as_dict(self) -> dict:
        return {"stop_updates": self.stop_updates,
                "schedule_updates": self.schedule_updates,
                "eval_every_updates": self.eval_every_updates,
                "report_steps": list(self.report_steps)}


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是正整数，得到布尔值 {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是正整数，得到 {value!r}") from None
    if number < 1:
        raise ValueError(f"{name} 必须是正整数，得到 {value!r}")
    return number


def resolve_update_budget(train_cfg: dict) -> UpdateBudget:
    """从 ``segformer.train`` 解析更新制预算；不合法就拒绝，不静默兜底。"""
    max_steps = train_cfg.get("max_steps")
    stop_raw = train_cfg.get("stop_updates")
    if stop_raw is None:
        if max_steps is None:
            raise ValueError("更新制训练需要 train.max_steps 或 train.stop_updates")
        stop = _positive_int(max_steps, "train.max_steps")
    else:
        stop = _positive_int(stop_raw, "train.stop_updates")
        if max_steps is not None and _positive_int(max_steps, "train.max_steps") != stop:
            raise ValueError(
                f"train.max_steps={max_steps} 与 train.stop_updates={stop} 不一致；"
                "更新制只把 stop_updates 当停止上限，两者写同一数值或删掉 max_steps"
            )

    schedule_raw = train_cfg.get("schedule_updates")
    schedule = stop if schedule_raw is None else _positive_int(schedule_raw, "train.schedule_updates")
    if schedule < stop:
        raise ValueError(
            f"train.schedule_updates={schedule} 小于停止步数 {stop}；"
            "调度长度必须覆盖全部更新，否则学习率会在中途归零"
        )

    interval = _positive_int(train_cfg.get("eval_every_updates", 100), "train.eval_every_updates")

    raw_steps = train_cfg.get("report_steps")
    if raw_steps is None:
        steps = (stop,)
    else:
        if not isinstance(raw_steps, (list, tuple)) or not raw_steps:
            raise ValueError("train.report_steps 必须是非空列表")
        steps = tuple(sorted({_positive_int(x, "train.report_steps") for x in raw_steps}))
        bad = [x for x in steps if x % interval or x > stop]
        if bad:
            raise ValueError(
                f"train.report_steps={list(steps)} 必须是 eval_every_updates={interval} 的整数倍"
                f"且不超过停止步数 {stop}；不合法项 {bad}"
            )
        if stop not in steps:
            steps = tuple(sorted({*steps, stop}))
    return UpdateBudget(stop_updates=stop, schedule_updates=schedule,
                        eval_every_updates=interval, report_steps=steps)


def poly_lr_scale(step: int, *, schedule_updates: int, warmup_updates: int,
                  power: float = 1.0) -> float:
    """poly 日程的乘子；``step`` 为 0 起算的更新序号，与 LambdaLR 的 epoch 计数一致。"""
    if step < warmup_updates:
        return (step + 1) / warmup_updates
    progress = (step - warmup_updates) / max(1, schedule_updates - warmup_updates)
    return max(0.0, (1.0 - progress) ** float(power))

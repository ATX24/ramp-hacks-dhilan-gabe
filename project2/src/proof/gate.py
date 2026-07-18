"""The proof gate — platform-owned, recipes cannot override it.

Evaluates a candidate on the frozen holdout: field accuracy, schema validity,
cost ratio vs teacher, and break-even request count. Below target => BLOCKED.
"""
from __future__ import annotations

from proof.backends.base import Backend, TrainedModel
from proof.metrics import aggregate
from proof.resources import Record, Target


async def prove(backend: Backend, candidate: TrainedModel, teacher: str,
                holdout: list[Record], target: Target, experiment_cost: float) -> dict:
    golds = []
    unl = [r for r in holdout if r.output is None]
    if unl:  # holdout golds come from the teacher, once, then frozen
        outs = await backend.sample(teacher, [r.input for r in unl])
        for r, o in zip(unl, outs):
            r.output = o
    golds = [r.output for r in holdout if r.output is not None]
    inputs = [r.input for r in holdout if r.output is not None]

    preds = await backend.sample(candidate.ref, inputs)
    metrics = aggregate(preds, golds)

    t_cost, s_cost = backend.cost_per_call(teacher), backend.cost_per_call(candidate.ref)
    cost_ratio = s_cost / t_cost if t_cost else 1.0
    saving = t_cost - s_cost
    break_even = int(experiment_cost / saving) + 1 if saving > 0 else None

    passed = metrics["field_accuracy"] >= target.quality and cost_ratio <= target.cost
    return {
        **metrics,
        "cost_ratio": round(cost_ratio, 4),
        "teacher_cost_per_call": t_cost,
        "student_cost_per_call": s_cost,
        "experiment_cost_usd": round(experiment_cost, 4),
        "break_even_requests": break_even,
        "target": target.model_dump(),
        "passed": passed,
    }

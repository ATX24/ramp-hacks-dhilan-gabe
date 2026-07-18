import asyncio

import pytest

import proof  # noqa: F401  (registers built-ins)
from proof.backends.base import RecipeIncompatible
from proof.backends.mock import MockBackend, oracle
from proof.data import descriptors
from proof.metrics import schema_valid
from proof.resources import DistillationStatus, Target
from proof.service import ProofService


async def run_distillation(svc, recipe, n=80, target=None):
    ds = svc.create_dataset([{"input": d} for d in descriptors(n)])
    dist = svc.create_distillation("mock-teacher", "mock-student", ds.id, recipe,
                                   target or Target(quality=0.9, cost=0.2))
    for _ in range(400):
        if dist.status in (DistillationStatus.READY, DistillationStatus.BLOCKED,
                           DistillationStatus.FAILED):
            break
        await asyncio.sleep(0.02)
    return dist


async def test_managed_recipe_reaches_ready():
    svc = ProofService(MockBackend())
    dist = await run_distillation(svc, "managed-distillation")
    assert dist.status == DistillationStatus.READY, dist.error
    assert dist.report["passed"]
    assert dist.report["break_even_requests"] > 0


async def test_gate_blocks_underpowered_candidate():
    svc = ProofService(MockBackend())
    # Tiny dataset -> weak student -> must be BLOCKED, never silently deployed.
    dist = await run_distillation(svc, "managed-distillation", n=20,
                                  target=Target(quality=0.99, cost=0.2))
    assert dist.status == DistillationStatus.BLOCKED
    assert dist.model_id is None


async def test_promotion_refused_without_passing_report():
    svc = ProofService(MockBackend())
    dist = await run_distillation(svc, "managed-distillation")
    model_id = dist.model_id
    svc.models[model_id].report["passed"] = False
    with pytest.raises(ValueError):
        svc.promote(model_id, "merchant-normalizer", "mock-teacher")


async def test_deploy_invoke_fallback_rollback():
    svc = ProofService(MockBackend())
    d1 = await run_distillation(svc, "managed-distillation")
    svc.promote(d1.model_id, "merchant-normalizer", "mock-teacher")

    out = await svc.invoke("merchant-normalizer", "GITHUB INC* 4821")
    assert schema_valid(out["output"])

    forced = await svc.invoke("merchant-normalizer", "GITHUB INC* 4821", force_fallback=True)
    assert forced["fallback"] and forced["output"] == oracle("GITHUB INC* 4821")

    d2 = await run_distillation(svc, "rejection-sampling")
    dep = svc.promote(d2.model_id, "merchant-normalizer", "mock-teacher")
    assert dep.model_id == d2.model_id and dep.history == [d1.model_id]
    assert svc.rollback("merchant-normalizer").model_id == d1.model_id


async def test_custom_recipe_registration_and_run():
    from proof.recipes.base import RecipeContext, register
    from proof.resources import Record

    class KeepAll:
        name = "keep-all-test"

        async def run(self, ctx: RecipeContext, train):
            outs = await ctx.sample(ctx.teacher, [r.input for r in train])
            kept = [Record(input=r.input, output=o) for r, o in zip(train, outs) if o]
            return await ctx.train(kept, {})

    register(KeepAll())
    svc = ProofService(MockBackend())
    dist = await run_distillation(svc, "keep-all-test")
    assert dist.status == DistillationStatus.READY, dist.error


async def test_logit_recipe_fails_loudly():
    from proof.recipes.base import register

    class LogitKD:
        name = "logit-kd-test"

        async def run(self, ctx, train):
            await ctx.logprobs(ctx.teacher, train[0].input, {})

    register(LogitKD())
    svc = ProofService(MockBackend())
    dist = await run_distillation(svc, "logit-kd-test", n=10)
    assert dist.status == DistillationStatus.FAILED
    assert "RECIPE_INCOMPATIBLE" in dist.error

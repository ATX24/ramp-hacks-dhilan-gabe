"""The live-demo beat: a user-defined distillation method in ~20 lines.

Self-consistency distillation — sample the teacher twice per input and keep
only records where it agrees with itself. Registered like any built-in, run
through the same proof gate. Usage:

    PYTHONPATH=src python examples/custom_recipe.py
"""
import asyncio

from proof.backends.mock import MockBackend
from proof.recipes.base import RecipeContext, register
from proof.resources import Record, Target
from proof.service import ProofService
from proof.data import descriptors


class SelfConsistency:
    name = "self-consistency"

    async def run(self, ctx: RecipeContext, train):
        a = await ctx.sample(ctx.teacher, [r.input for r in train])
        b = await ctx.sample(ctx.teacher, [r.input + " " for r in train])  # perturbed re-ask
        kept = [Record(input=r.input, output=x)
                for r, x, y in zip(train, a, b) if x is not None and x == y]
        ctx.emit("self_consistency_kept", kept=len(kept), total=len(train))
        return await ctx.train(kept, {"teacher": ctx.teacher})


register(SelfConsistency())


async def main():
    svc = ProofService(MockBackend())
    ds = svc.create_dataset([{"input": d} for d in descriptors(60)])
    dist = svc.create_distillation("mock-teacher", "mock-student", ds.id,
                                   "self-consistency", Target(quality=0.9, cost=0.2))
    while dist.status.value not in ("READY", "BLOCKED", "FAILED"):
        await asyncio.sleep(0.05)
    print(dist.status.value, dist.report)


if __name__ == "__main__":
    asyncio.run(main())

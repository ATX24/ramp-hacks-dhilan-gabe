"""Built-in recipe: rejection sampling. Label with the teacher, then keep only
records a judge accepts — smaller but cleaner training set. This is the
live-demo recipe: same primitives, different algorithm, measurably different
result through the same proof gate."""
from __future__ import annotations

from proof.backends.base import TrainedModel
from proof.recipes.base import RecipeContext, register
from proof.resources import Record


class RejectionSampling:
    name = "rejection-sampling"

    RUBRIC = ("The output correctly normalizes the descriptor: right merchant, "
              "right category, correct subscription status and policy flags.")

    async def run(self, ctx: RecipeContext, train: list[Record]) -> TrainedModel:
        outputs = await ctx.sample(ctx.teacher, [r.input for r in train])
        labeled = [Record(input=r.input, output=o)
                   for r, o in zip(train, outputs) if o is not None]
        ctx.emit("labeled", count=len(labeled))

        verdicts = await ctx.judge(self.RUBRIC, [r.model_dump() for r in labeled])
        kept = [r for r, ok in zip(labeled, verdicts) if ok]
        ctx.emit("filtered", kept=len(kept), rejected=len(labeled) - len(kept))

        return await ctx.train(kept, {"teacher": ctx.teacher})


register(RejectionSampling())

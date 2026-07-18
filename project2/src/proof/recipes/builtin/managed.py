"""Built-in recipe: straight teacher-labeled SFT (the Bedrock managed
distillation shape). Label every training input with the teacher, train the
student on all of it. ~15 lines — the baseline every custom recipe fights."""
from __future__ import annotations

from proof.backends.base import TrainedModel
from proof.recipes.base import RecipeContext, register
from proof.resources import Record


class ManagedDistillation:
    name = "managed-distillation"

    async def run(self, ctx: RecipeContext, train: list[Record]) -> TrainedModel:
        unlabeled = [r for r in train if r.output is None]
        if unlabeled:
            ctx.emit("labeling", count=len(unlabeled), teacher=ctx.teacher)
            outputs = await ctx.sample(ctx.teacher, [r.input for r in unlabeled])
            for r, o in zip(unlabeled, outputs):
                r.output = o
        labeled = [r for r in train if r.output is not None]
        ctx.emit("training", records=len(labeled))
        return await ctx.train(labeled, {"teacher": ctx.teacher})


register(ManagedDistillation())

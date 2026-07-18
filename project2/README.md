# Proof — quality-gated AI spend optimizer

Distill expensive repetitive model calls into cheap students — with **user-defined
distillation methods** (Tinker-style: you write the algorithm as plain Python
against a small primitives API; the platform owns execution and the quality gate).

Four durable resources: immutable `Dataset`, async `Distillation`, immutable
`Model`, callable `Deployment`. A failed evaluation gate produces `BLOCKED`,
never a silently deployed model.

## Recipes (the differentiator)

```python
from proof.recipes.base import RecipeContext, register
from proof.resources import Record

class SelfConsistency:
    name = "self-consistency"
    async def run(self, ctx: RecipeContext, train):
        a = await ctx.sample(ctx.teacher, [r.input for r in train])
        b = await ctx.sample(ctx.teacher, [r.input + " " for r in train])
        kept = [Record(input=r.input, output=x)
                for r, x, y in zip(train, a, b) if x is not None and x == y]
        return await ctx.train(kept, {"teacher": ctx.teacher})

register(SelfConsistency())
```

Primitives: `ctx.sample` / `ctx.judge` / `ctx.logprobs` / `ctx.train` / `ctx.emit`.
Built-ins (`managed-distillation`, `rejection-sampling`) use the same API.
Recipes never see the holdout; the platform-owned proof gate (frozen split,
field accuracy, schema validity, cost ratio, break-even) decides READY vs BLOCKED.
Incompatible demands (e.g. logit KD on a black-box teacher) fail loudly with
`RECIPE_INCOMPATIBLE`.

## Backends

- `mock` (default): deterministic oracle teacher — full loop runs offline.
- `bedrock`: Nova Pro teacher → Nova Micro/Lite students; `train()` submits a
  managed DISTILLATION customization job. Set `PROOF_BACKEND=bedrock`,
  `PROOF_BEDROCK_ROLE_ARN`, `PROOF_BUCKET`, `AWS_PROFILE=ramp-hackathon`.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
PYTHONPATH=src:packages/sdk .venv/bin/pytest -q
PYTHONPATH=src .venv/bin/python examples/custom_recipe.py
PYTHONPATH=src:apps/api .venv/bin/uvicorn proof_api.main:app --reload
```

SDK happy path:

```python
proof = Proof(base_url="http://localhost:8000")
job = await proof.distill(teacher="amazon:nova-pro", data="./traces.jsonl",
                          target={"quality": 0.92, "cost": 0.1})
model = await job.deploy("merchant-normalizer")   # raises unless gate == READY
```

## AWS state (us-east-1, profile `ramp-hackathon`)

- Bucket: `proof-ramp-hackathon-225989358036`
- Role: `ProofBedrockDistillationRole`
- All resources tagged `Project=RampHackathon, Owner=Dhilan, TTL=2026-07-20`.

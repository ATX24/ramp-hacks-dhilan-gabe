# Distillery — Smaller models. Proven economics.

High-level distillation API (Curate -> Synthesize -> Train -> Prove) plus **TinyFable**,
a tiny finance-generalist student distilled from Claude Opus teacher responses via
`sequence.v1` (offline sequence-level KD). See the shared plan for locked decisions.

- API: FastAPI control plane (`apps/api`), deployed on Fly.io
- SDK: `packages/sdk/distillery_sdk` — three-line happy path
- Data: deterministic synthetic finance world + executable latent oracle (`src/distillery/data`)
- Synthesis: programmatic Claude Opus (`claude-opus-4-8`) fills only missing train/val labels
- Trainer: `python -m distillery.training.entrypoint --manifest ... --data ... --out ...` (QLoRA SFT; run on a GPU host — not started by default)
- `logit.v1` is implemented at the contract level but is incompatible with an API black-box teacher and fails loudly (`RECIPE_INCOMPATIBLE`) — no silent downgrade.

## User-defined distillation recipes

Anyone can write a distillation method as plain Python against
`SynthesisContext` primitives (`teacher_label`, `oracle_label`, `is_valid`,
`agrees_with_oracle`, `emit`). Recipes run at the synthesis/curation stage,
always produce a NEW immutable dataset, never see test splits, and train via
`sequence.v1` on their output — the proof gate is not overridable.

```python
from distillery.recipes.custom import SynthesisContext, register

class RejectHard:
    name = "reject_hard.v1"
    requires = frozenset({"teacher"})           # gated by the resolver
    description = "Teacher-label, then drop hard examples the teacher got wrong."

    def run(self, ctx: SynthesisContext, examples):
        ctx.teacher_label(examples)
        return [ex for ex in examples
                if ctx.is_valid(ex) and (ex.difficulty != "hard" or ctx.agrees_with_oracle(ex))]

register(RejectHard())
```

Load via `DISTILLERY_RECIPES=my_pkg.my_recipes` (imported at API startup), then:

```bash
curl -X POST localhost:8000/v1/datasets/<ds>/recipes/reject_hard.v1 -d '{"dry_run": false}'
# -> {"new_dataset_id": "ds_...", ...}   then plan/distill that dataset as usual
```

Built-ins written against the same primitives: `rejection_sampling.v1`,
`oracle_curriculum.v1` (zero teacher cost). SDK: `distillery.datasets.run_recipe(ds, name)`.

## Quickstart
```bash
uv venv && uv pip install -e ".[dev]"
PYTHONPATH=.:src:packages/sdk pytest
PYTHONPATH=.:src:packages/sdk uvicorn apps.api.distillery_api.main:app --reload
python examples/finance_generalist.py http://localhost:8000
```

## Deploy
```bash
fly deploy
fly secrets set ANTHROPIC_API_KEY=...   # required for real Opus teacher synthesis
```

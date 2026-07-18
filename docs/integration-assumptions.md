# Integration assumptions (Workstream 8)

Path-isolated demos and e2e fakes target these interfaces. If reality diverges,
update adapters/tests—do not silently weaken safety gates.

## Python SDK (Workstream 1)

Expected import (either form acceptable):

```python
from distillery import Distillery
# or
from distillery.client import Distillery
```

Expected methods / resources:

```python
client = Distillery(api_key=...)
dataset = client.datasets.create(path)           # -> Dataset
plan = client.plan_distillation(dataset, recipe="auto")
# or client.distillation_runs.plan(...)
run = client.distill(dataset, recipe="auto")     # launches training
run.wait()
run.model_artifact
run.proof_report
```

`plan_distillation` must not create a backend job.

## HTTP API

See [runbook-aws-api-deploy.md](./runbook-aws-api-deploy.md). Plan endpoint is
mandatory for demos; create-run is out of scope for default scripts.

## Contracts package (foundation / W1)

Already present under `src/distillery/contracts/**`. Examples optionally import:

- `distillery.contracts.recipes.resolve_requested_recipe`
- `distillery.contracts.recipes.AutoResolverInput`
- Resource models for typing in tests

## Fixtures (Workstream 2 / foundation)

- `tests/fixtures/finance_world_v1/golden.jsonl`
- `tests/fixtures/finance_world_v1/fixture_manifest.json`

Held-out coverage in the golden set is sparse; selector tests accept the
available `iid_test` / `ood_test` / `test` rows and skip-or-fail clearly when a
primary task lacks held-out rows.

## Infra (Workstream 6)

API deploy commands and SageMaker submit live elsewhere. This stream documents
API-only deploy and never calls training submit helpers.

## What is intentionally unimplemented here

- Model download / Hugging Face snapshot pin resolution
- SageMaker `CreateTrainingJob`
- Real proof metric computation (Workstream 5)
- Web UI (Workstream 7)

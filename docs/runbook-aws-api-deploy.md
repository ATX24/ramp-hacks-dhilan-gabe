# AWS API deployment runbook (control plane only)

Deploy or verify the Distillery **API control plane**. This runbook does **not**
launch SageMaker Training Jobs, download teacher/student weights, or start
distillation. Training remains a separate, explicitly acknowledged path outside
this workstream's demo defaults.

## Scope

| In scope | Out of scope |
| --- | --- |
| FastAPI (or Mangum) API packaging | SageMaker Training Job submit |
| IAM role for API execution | Model weight download / HF cache |
| Env config for metadata store + S3 read of immutable artifacts | Endpoints for inference serving |
| Health + OpenAPI smoke | Autopilot, Observe, traffic routing |

## Expected infra / API tooling (integration assumptions)

Workstream 6 owns `infra/sagemaker/**` and backends. Workstream 1 owns
`apps/api/distillery_api/**`. Until those land, treat the following as the
contract this runbook will call:

```text
POST /v1/datasets
GET  /v1/datasets/{dataset_id}
POST /v1/distillation-runs/plan          # pure; no job
POST /v1/distillation-runs              # seals + submits (DO NOT call here)
GET  /v1/distillation-runs/{run_id}
POST /v1/distillation-runs/{run_id}/cancel
GET  /v1/model-artifacts/{artifact_id}
GET  /v1/proof-reports/{report_id}
GET  /v1/recipes
```

Local packaging entrypoints expected once API code merges:

```bash
# Illustrative — exact script names come from apps/api + infra owners
uv sync
uv run uvicorn distillery_api.main:app --host 0.0.0.0 --port 8000
```

AWS Lambda + API Gateway (Mangum) when `infra/` provides a SAM/CDK/Terraform
module — use that module's documented deploy command. Do not invent a second
cloud architecture mid-hackathon.

## Deploy checklist (API only)

1. Confirm region (`us-east-1` planning default) and account budget alerts.
2. Deploy **API** stack only. Refuse any template parameter that creates
   `ml.*` training instances or SageMaker endpoints.
3. Set secrets via the platform secret store (never commit keys):
   - API auth key material for `DISTILLERY_API_KEY` clients
   - Read credentials for metadata store / artifact bucket
4. Smoke (plan only):

```bash
curl -sS -X POST "$API_BASE/v1/distillation-runs/plan" \
  -H "Authorization: Bearer $DISTILLERY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"ds_smoke","recipe":"auto"}' | jq .
```

Assert the response does **not** include a backend job ARN / Training Job name.

5. Negative check: do **not** call `POST /v1/distillation-runs` from this runbook.

## Demo client against a live API

```bash
export DISTILLERY_API_KEY=...   # from secret store
python examples/finance_generalist.py --mode plan --api-key "$DISTILLERY_API_KEY"
```

If the SDK package is not importable yet, the example falls back to the local
protocol adapter and still refuses training.

## Rollback

- Redeploy previous API image/version.
- Leave S3 artifact prefixes immutable; never rewrite successful run objects.
- Cancel is for runs, not for this API-only deploy path.

## Explicit non-goals reminder

No deployment of TinyFable as a hosted model. No proxy. No notifications.
Distillation proof may conclude `do_not_distill`.

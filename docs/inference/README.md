# Live inference serving plane (Demo / Playground)

This tree owns the SageMaker real-time inference container and local serving
package. It does **not** yet wire `apps/api` or `apps/web`.

## Layout

| Path | Role |
| --- | --- |
| `apps/inference/` | Typed serving package (`/ping`, `/invocations`, `/v1/demo/*`) |
| `containers/inference/` | Digest-pinned non-root inference image |
| `infra/inference/` | CloudFormation for Model / EndpointConfig / Endpoint (plan-only default) |
| `scripts/inference/` | Build/publish helpers (dry-run default) |
| `tests/inference/` | Deterministic fake-runtime tests (no model downloads) |

## Model bundle contract

Mount an immutable bundle at `/opt/ml/model` (SageMaker `ModelDataUrl`):

```text
serving_registry.json
base/                          # pinned Qwen2.5-0.5B snapshot
adapters/<arm>/                # PEFT adapters or merged dirs
integrity/SHA256SUMS
```

`serving_registry.json` enumerates servable `model_id` / `artifact_id` pairs
(`student_base`, `oracle_sft`, `sequence_kd`, `logit_kd`, `ce_ablation`,
`promoted_winner`). Switching is registry-driven; missing artifacts fail loud.

## Later API wiring (do not implement here)

Add to `apps/api` later:

| Route | Behavior |
| --- | --- |
| `GET /v1/demo/health` | Proxy readiness from SageMaker Runtime or cached health |
| `GET /v1/demo/models` | Return sealed registry/stats (or invoke a lightweight metadata path) |
| `POST /v1/demo/infer` | Validate web contract body, then `sagemaker-runtime:InvokeEndpoint` |

Required IAM on the API role (scoped to one endpoint ARN):

- `sagemaker:InvokeEndpoint` on `arn:aws:sagemaker:<region>:<account>:endpoint/<name>`
- No training permissions

Web already calls `NEXT_PUBLIC_DISTILLERY_INFERENCE_URL` + `/v1/demo/infer`.
Point that base URL at the API once the proxy exists (not directly at SageMaker
unless CORS/auth are handled).

## Safety defaults

- Offline / network isolation enforced (`HF_HUB_OFFLINE`, template
  `EnableNetworkIsolation`)
- Checksums + pinned revisions validated before ready
- Bounded tokens, timeouts, concurrent requests
- Never silently substitute another model
- Image build/publish and stack deploy default to dry-run / plan-only

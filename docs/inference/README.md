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

## Real oracle-SFT baseline

The exact `aws-smoke-oracle-sft-79db97a954be` registration is sealed at
`infra/inference/baselines/aws-smoke-oracle-sft-79db97a954be.json`.

- Model: `model_oracle_sft`
- Artifact: `artifact_oracle_sft_0021b7d6cdfd`
- Supported tasks: `variance_analysis`, `cash_reconciliation`
- Manifest: `79db97a954be7bd3395cf6856ad2d1b74e3038d022d3aa0e8ff7a5f6b357edaf`
- Adapter weights: `0021b7d6cdfd86e6a255e9367bc53607055937f2cc249a418f030d3935b13bfc`
- Bundle: `s3://distillery-225989358036-us-east-1/inference/baselines/aws-smoke-oracle-sft-79db97a954be/model.tar.gz`

The 16-example sealed smoke comparison measured a `0.0` primary index for both
base and adapter, with `0.0` delta and no confidence interval. The registry
therefore reports `insufficient_evidence` and `not_promoted`; it does not claim
an improvement. The emergency split contains no transaction-review examples.

`scripts/inference/materialize_serving_bundle.py` fails closed on the training
manifest seal, tar hashes, output `SHA256SUMS`, base snapshot manifest, adapter
copy equality, reload/tokenizer evidence, and measured comparison hashes before
building the immutable SageMaker model tar.

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

For the deployed baseline, set the API's endpoint name to
`distillery-demo-inference`. The API should expose the inference service's exact
`/v1/demo/health`, `/v1/demo/models`, and `/v1/demo/infer` payloads unchanged.
The SageMaker endpoint is private AWS Runtime API, not a browser URL.

## Safety defaults

- Offline / network isolation enforced (`HF_HUB_OFFLINE`, template
  `EnableNetworkIsolation`)
- Checksums + pinned revisions validated before ready
- Bounded tokens, timeouts, concurrent requests
- Never silently substitute another model
- Strict task schemas validate nested types and accounting arithmetic
- One `ml.g5.xlarge` endpoint, with a one-time auto-delete schedule and a
  `$4.224` maximum three-hour hosting budget
- Image build/publish and stack deploy default to dry-run / plan-only

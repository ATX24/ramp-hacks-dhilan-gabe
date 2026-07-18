# `distillery_inference`

SageMaker-shaped inference API for the Demo/Playground.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/ping` | SageMaker liveness (`200` / `503`) |
| POST | `/invocations` | SageMaker invoke body = live infer request |
| GET | `/v1/demo/health` | Web/API health contract |
| GET | `/v1/demo/models` | Registry + stats (`null`/unknown when absent) |
| POST | `/v1/demo/infer` | Same body/response as `/invocations` |

## Request / response (web-compatible)

Request:

```json
{
  "model_id": "model_sequence_kd",
  "artifact_id": "artifact_sequence_kd",
  "task": "transaction_review",
  "example_id": "ex_txn_hard_001",
  "input": {"amount_minor": 5000000}
}
```

Success includes `structured_output`, `raw_output`, `latency_ms`,
`prompt_tokens`, `completion_tokens`, and `provenance: "live"`.

## Local fake runtime (tests)

```bash
DISTILLERY_INFERENCE_RUNTIME=fake \
DISTILLERY_INFERENCE_MODEL_ROOT=/path/to/bundle \
DISTILLERY_INFERENCE_REQUIRE_OFFLINE=0 \
uv run pytest tests/inference -q
```

Production containers set `DISTILLERY_INFERENCE_RUNTIME=torch` and require
offline env vars.

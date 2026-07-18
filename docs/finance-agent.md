# Finance Agent task

Isolated conversational finance Q&A task with sandboxed synthetic tools.

## Package

`src/distillery/finance_agent/` — contracts, sandbox, oracle trajectories, corpus
generation, metrics, and an isolated `agent_trajectory.v1` plan adapter.

Does **not** extend TinyFable `TaskId` / `finance_world.*` envelopes.

## Corpora

| Corpus | Count | Splits |
| --- | --- | --- |
| smoke | 48 | train 24 / validation 8 / test 16 |
| planned | 2200 | train 1200 / validation 200 / iid_test 400 / ood_test 400 |

OOD holds out tools `variance_drill_down`, `transaction_matching` and domain `payroll`.

## Metrics (`finance_agent.metrics.v1`)

- tool_selection_accuracy
- argument_exactness
- tool_result_use
- final_answer_correctness
- unnecessary_calls
- latency_ms / cost_usd_micros
- end_to_end_success

## Integration steps (after review)

1. Keep active UI/API/training paths untouched until review.
2. Seal/register `agent_trajectory.v1` through BYODT (`examples/byodt/agent_trajectory_v1/`).
3. Point Demo mode at `examples/finance_agent/chat_demo_contract.json` and
   `examples/finance_agent/model_registry_finance_agent.json`.
4. Generate smoke/planned corpora; seal manifests; wire proof arms to agent metrics.
5. Distill 72B teacher trajectories into TinyFable Generalist + specialists via the
   isolated adapter (never alias to `sequence.v1` / `logit.v1`).

## Commands

```bash
uv run pytest tests/finance_agent -q
uv run ruff check src/distillery/finance_agent tests/finance_agent
uv run python scripts/finance_agent/generate_smoke.py
```

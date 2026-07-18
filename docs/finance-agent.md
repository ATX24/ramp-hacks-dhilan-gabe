# Finance Agent task

Isolated conversational finance Q&A over deterministic synthetic tools. It does
not extend TinyFable `TaskId` / `finance_world.*`, call live finance systems, or
expose shell/network capabilities.

## Data boundary

Every `finance_agent.v1` envelope has two sealed parts:

- `model_input`: system prompt, public world facts/IDs, user turns, canonical tool
  JSON Schemas, and component hashes.
- `gold`: ordered oracle trajectory, exact tool results/provenance, expected answer,
  and latent-world hash.

Materialization writes `model/`, `gold/`, and `oracle/` separately. Model and
inference processes receive only `model/`. Validation requires the private world,
replays every call, and compares canonical result bytes including provenance.

## Corpora

- Smoke: 48 episodes. Train 24, validation 8, test 8, OOD test 8.
- Planned: 2,200 episodes. Train 1,200, validation 200, IID test 400,
  OOD test 400.

`transaction_matching`, `variance_drill_down`, and payroll semantics occur only in
OOD. Payroll changes the exposed COA, ledger account/memo, policy, and merchant.
Prompt leakage checks cover identity, model-input hash, normalized text, template
family, and semantic fingerprint across every split pair.

## Metrics and proof

`finance_agent.metrics.v2` scores ordered tool names, arguments, result bytes,
result-to-answer bindings, final answers, skipped/extra calls, and end-to-end
success. Latency and cost remain null until measured.

`finance-agent-proof.v1` binds corpus seed/content/order, system prompts, tool
schemas, render template, model, tokenizer, chat template, license/output-use
disposition, and measured cost. Generated corpora have honest `not_ready` proof
state because model/tokenizer/license/cost artifacts do not exist.

## agent_trajectory.v1

The isolated objective supervises assistant messages, tool calls, and final
answers. System, user, tool-result, and padding tokens use `ignore_index`.
Current labels are oracle labels, not teacher labels. No model, teacher rollout,
specialist, or training artifact is claimed.

Do not register through BYODT or wire UI/API/training until independent re-review.

## Checks

```bash
uv run pytest tests/finance_agent -q
uv run ruff check src/distillery/finance_agent tests/finance_agent scripts/finance_agent
uv run python scripts/finance_agent/generate_smoke.py
```

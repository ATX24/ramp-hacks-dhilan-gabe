# Context Map

## Contexts

- [Managed Tinker](./CONTEXT.md) — product language for managed post-training (optional root glossary when present)
- [Finance Agent](./src/distillery/finance_agent/CONTEXT.md) — conversational finance Q&A with sandboxed tools
- TinyFable finance world (`src/distillery/contracts/tasks.py`, `src/distillery/data/`) — single-turn synthetic finance tasks (`transaction_review`, `variance_analysis`, `cash_reconciliation`)

## Relationships

- **Finance Agent → TinyFable finance world**: Shared finance vocabulary (accounts, policies, ledgers) but **separate envelopes**. Finance Agent must not extend `TaskId` or `FinanceTaskEnvelope`.
- **Finance Agent → BYODT**: The planned role-masked objective uses technique id
  `agent_trajectory.v1`. Integration into `TechniqueRegistry.with_builtins()` waits
  for BYODT review. Until then the objective/collator and plan adapter stay under
  `src/distillery/finance_agent/technique/`.
- **Finance Agent ↛ live systems**: No real Ramp/ERP/bank APIs, shell, or network tool I/O.

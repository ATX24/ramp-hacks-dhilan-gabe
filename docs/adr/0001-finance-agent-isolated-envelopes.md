# Isolate Finance Agent envelopes from TinyFable tasks

TinyFable's locked task model is three single-turn finance tasks under `finance_world.*`. Finance Agent is multi-turn tool use with different evaluation axes, so it gets its own `finance_agent.v1` envelope and package instead of extending `TaskId` / `FinanceTaskEnvelope`.

**Considered Options**: Extend `TaskId` with `finance_agent`; parallel schema under `finance_agent.v1`.

**Consequences**: Existing corpora, proof metrics, and Demo task pickers stay stable. Finance Agent UI mode must opt in via its own demo contract later.

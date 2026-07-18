# Finance Agent

Conversational finance Q&A over a sandboxed synthetic tool environment. Labels and trajectories come from a deterministic oracle, not live finance systems.

## Language

**Finance Agent Episode**:
One multi-turn conversation that ends in a final answer or an explicit refusal.
_Avoid_: Session, chat, ticket

**Trajectory**:
The ordered turn sequence for an Episode, including user messages, assistant tool calls, tool results, and the final assistant message.
_Avoid_: Trace (unless referring to observability), sequence.v1 example

**Tool Call**:
A named sandbox tool invocation with bounded, schema-valid arguments.
_Avoid_: Function call, action, API request

**Tool Result**:
The deterministic JSON payload returned by the sandbox for one Tool Call, including provenance.
_Avoid_: Observation, side effect

**Sandbox**:
The in-process synthetic tool environment that executes Tool Calls without shell or network access.
_Avoid_: Runtime, plugin, MCP server

**Latent World**:
Hidden synthetic finance state (accounts, policies, ledger rows, merchants) from which oracle labels are derived.
_Avoid_: Database, ERP, company books

**Oracle Trajectory**:
The gold Trajectory produced by deterministic solvers over Latent World state for one Episode.
_Avoid_: Teacher sample, human label

**Hard Case Family**:
A labeled failure-mode template used to generate adversarial Episodes (wrong tool, stale policy, refusal, and similar).
_Avoid_: Difficulty, slice

**Held-Out Split**:
A corpus partition that withholds tools and/or domain facets from training to measure generalization.
_Avoid_: Test set (when tool/domain hold-out is the intent)

**agent_trajectory.v1**:
The distillation technique that supervises student models on Oracle Trajectories from a large teacher. It is not `sequence.v1` and not `logit.v1`.
_Avoid_: sequence KD, SFT recipe (as technique identity)

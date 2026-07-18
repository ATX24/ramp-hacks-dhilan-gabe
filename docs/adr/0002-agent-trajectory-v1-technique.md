# Distill Finance Agent via agent_trajectory.v1, not sequence.v1

Sequence-level trajectory supervision from a 72B teacher looks like response SFT, but the unit of supervision is a full Trajectory (tool calls + results + final answer), with agent-specific metrics. Calling it `sequence.v1` would silently collapse technique identity and auto-resolver behavior.

Decision: label the technique `agent_trajectory.v1` and keep its plan adapter under `src/distillery/finance_agent/technique/` until BYODT review clears registry integration. Do not register it in `TechniqueRegistry.with_builtins()`.

**Consequences**: Trainers must opt into the adapter explicitly. Catalog recipe stub `agent_trajectory` remains unimplemented; the sealed technique id is `agent_trajectory.v1`.

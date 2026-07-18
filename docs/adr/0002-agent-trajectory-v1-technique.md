# Keep role-masked trajectories distinct as agent_trajectory.v1

Finance Agent supervision spans interleaved user, assistant, and tool turns. A
prompt/completion mask cannot express the required ownership boundary.

Decision: `agent_trajectory.v1` uses role-masked causal CE. Assistant messages,
tool calls, and final answers are supervised. System, user, tool-result, and
padding tokens use `ignore_index`. The objective and collator remain under
`src/distillery/finance_agent/technique/` and are not registered in BYODT
builtins until review clears.

Current labels are deterministic oracle trajectories. Teacher-labeled
trajectories require a distinct rollout artifact with exact model revision,
license text hash, output-use review, and attribution disposition. No teacher or
specialist claim is made without that evidence.

Consequences: the catalog recipe remains an unimplemented stub; the isolated
plan is not training-ready and cannot silently route through another objective.

# agent_trajectory.v1 (isolated)

Role-masked causal supervision over Finance Agent trajectories. System, user, tool
result, and padding tokens use `ignore_index`; assistant messages, tool calls, and
final answers are supervised.

Current labels are deterministic oracle trajectories. No teacher rollout, model
artifact, specialist route, measured cost, or training readiness is claimed.

**Status:** objective/collator and plan adapter only. Not registered in
`TechniqueRegistry.with_builtins()`.

## Inspect the sealed plan

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from distillery.finance_agent.technique import AgentTrajectoryPlanAdapter

config = json.loads(
    Path("examples/byodt/agent_trajectory_v1/sample_config.json").read_text()
)
plan = AgentTrajectoryPlanAdapter().plan(config)
print(plan.model_dump_json(indent=2))
PY
```

The sample binds the deterministic smoke corpus and intentionally leaves model,
tokenizer, chat-template, license, and cost evidence null. The resulting plan is
`not_materialized` and `training_ready=false`.

## Later integration gates

1. Independent review of the objective and exact role masks.
2. Pinned model/tokenizer/template artifacts and reviewed license/output use.
3. Measured cost evidence and a ready paired proof protocol.
4. Explicit BYODT registration. Never alias this objective to another technique.

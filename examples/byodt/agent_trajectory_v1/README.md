# agent_trajectory.v1 (isolated)

Finance Agent trajectory distillation from a 72B teacher into TinyFable Generalist
and task specialists.

**Status:** plan adapter only. Not registered in `TechniqueRegistry.with_builtins()`.
Do not treat this as `sequence.v1` or `logit.v1`.

## Local plan

```bash
uv run python -c "
from distillery.finance_agent.technique import AgentTrajectoryPlanAdapter
plan = AgentTrajectoryPlanAdapter().plan({
  'max_length': 4096,
  'max_completion': 1024,
  'seed': 17,
  'teacher_model_id': 'Qwen/Qwen2.5-72B-Instruct',
  'teacher_revision': 'a' * 40,
  'student_model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
  'student_revision': 'b' * 40,
  'trajectory_corpus_sha256': 'c' * 64,
  'specialist_task': 'generalist',
})
print(plan.technique_id, plan.plan_hash())
"
```

## BYODT integration (later)

After review:

1. Seal an external or builtin descriptor under technique id `agent_trajectory.v1`
2. Register via `TechniqueRegistry.register` / `byodt_ctl.py register`
3. Wire Demo mode using `examples/finance_agent/chat_demo_contract.json`

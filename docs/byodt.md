# Bring Your Own Distillation Technique (BYODT)

Hackathon users (for example Dhilan) can add a distillation technique without
editing control-plane orchestration or hardcoding recipe branches.

## Seam

Callers and tests use one deep module interface:

```python
from distillery.techniques import (
    TechniqueRegistry,
    TechniqueRequest,
    CompatibilityContext,
)

registry = TechniqueRegistry.with_builtins()
registry.register_from_path(Path("technique.json"))
plan = registry.plan(
    TechniqueRequest(technique_id="...", version="1.0.0", config={...}),
    CompatibilityContext(...),
)
```

Built-in adapters wrap `sequence.v1` / `logit.v1` and yield the existing
training load plan + loss contract. External adapters yield a sealed
`ExternalExecutionPlan`; the CLI can materialize its content-addressed channel
envelope. There is no external trainer/backend consumer yet, so this is
plan-only and does not claim runtime isolation enforcement.

## What a contributor seals

1. **Technique descriptor** (`distillery.technique.v1`) — immutable, versioned,
   content-hashed (`descriptor_sha256`).
2. **Config JSON Schema** — Draft 2020-12, `type: object`,
   `additionalProperties: false`, fully inline (no references/retrieval or
   defaults), and no secret-like fields.
3. **Capabilities + evidence requirements** — closed vocabulary only.
4. **Teacher signal, tokenizer constraint, artifact contract, metrics,
   hardware, cost model**.
5. For externals: **digest-pinned plugin image** (`@sha256:...`, never a tag)
   and **reviewed source binding** (commit + tree hash + review record hash).

## Local workflow

```bash
# Seal / refresh an example descriptor
uv run python examples/byodt/reverse_kl_v1/build_descriptor.py

# Validate the descriptor/schema (not environment compatibility)
uv run python scripts/techniques/byodt_ctl.py validate-descriptor \
  examples/byodt/reverse_kl_v1/technique.json

# Register into a local descriptor store
uv run python scripts/techniques/byodt_ctl.py register \
  examples/byodt/reverse_kl_v1/technique.json \
  --registry-dir /tmp/distillery-techniques

# Plan only (no training submit)
uv run python scripts/techniques/byodt_ctl.py plan \
  --technique-id hackathon.dhilan.reverse_kl \
  --version 1.0.0 \
  --descriptor examples/byodt/reverse_kl_v1/technique.json \
  --config examples/byodt/reverse_kl_v1/sample_config.json \
  --context examples/byodt/reverse_kl_v1/sample_context.json \
  --channel-dir /tmp/byodt-channel
```

## How it later appears in API / UI

MVP integration is intentionally staged (this stream does not patch live API/UI):

| Surface | Later wiring |
| --- | --- |
| API `DistillationRun` create | Accept `technique_id` + `version` + `config` beside today's `recipe` field; resolve via `TechniqueRegistry` before sealing the run manifest |
| Manifest | Persist `descriptor_sha256`, `config_sha256`, `protocol_sha256`, and external channel identity |
| Train stage UI | Technique picker lists registered descriptors (`list_techniques()`), shows capabilities/cost model, and plan-only preview of `TechniquePlan` |
| Prove stage | Metrics declared in the descriptor feed the proof evaluator arm labels |

Until that wiring lands, hackathon demos use `byodt_ctl.py plan` and the
Python seam above. Built-ins already expose parity with `sequence.v1` /
`logit.v1` objective fields through the same `registry.plan(...)` path.
`plan` is the only compatibility preflight; descriptor validation alone makes
no claim about model, tokenizer, hardware, or backend compatibility.

## Non-goals (enforced)

- No silent fallback to another technique when capabilities fail
- No mutable descriptors after sealing
- No tag-based plugin images
- No in-process import of external technique code
- No training launch from the BYODT CLI
- No claim that channel isolation is enforced before backend wiring exists

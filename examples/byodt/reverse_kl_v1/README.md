# Example BYODT technique: `hackathon.dhilan.reverse_kl@1.0.0`

This is a plan-only SDK example for Bring Your Own Distillation Technique.

## What is sealed

- `technique.json` — immutable versioned descriptor (content-hashed)
- `config.schema.json` — JSON Schema for technique config (`additionalProperties: false`)
- `sample_config.json` — one valid config instance
- `build_descriptor.py` — helper to re-seal the descriptor after schema edits

## Control-plane contract

External technique code is **not** imported into Distillery. Planning happens in
the control plane and can produce a content-addressed `technique_plan.json`.
No trainer/backend consumes that channel yet. Future integration must enforce
the sealed digest, source identity, artifact contract, and network isolation.

## Validate / register / plan-only

```bash
uv run python scripts/techniques/byodt_ctl.py validate-descriptor \
  examples/byodt/reverse_kl_v1/technique.json

uv run python scripts/techniques/byodt_ctl.py register \
  examples/byodt/reverse_kl_v1/technique.json \
  --registry-dir /tmp/distillery-techniques

uv run python scripts/techniques/byodt_ctl.py plan \
  --technique-id hackathon.dhilan.reverse_kl \
  --version 1.0.0 \
  --descriptor examples/byodt/reverse_kl_v1/technique.json \
  --config examples/byodt/reverse_kl_v1/sample_config.json \
  --context examples/byodt/reverse_kl_v1/sample_context.json
```

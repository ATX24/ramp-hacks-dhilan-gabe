# Sequence teachers

`TeacherGenerator` supports the existing local/open-weight Qwen path, injected
Bedrock Converse, and direct official Anthropic SDK clients. No model alias is
assumed: direct Anthropic model IDs are supplied by the caller and verified by
the model probe.

Default project policy:

- Apache-2.0 open-weight Qwen output may teach Qwen students.
- Claude and Nova are limited to non-retained evaluation/benchmark output.
- Nova output does not teach Qwen.
- Claude-to-Qwen sequence KD is blocked unless a current, hash-bound,
  model/student/use/storage/disposition-specific written authorization record is
  supplied. The record contains only opaque evidence metadata and an operator
  attestation. Permission documents and sensitive terms are never stored.
- Bedrock returns text/tool messages, not logits, so `logit.v1` is rejected.
- Direct Anthropic Messages also returns text/tool trajectories, never logits.
- Direct credentials are runtime-only. An injected resolver may read only
  `ANTHROPIC_API_KEY` from the environment or macOS Keychain; keys never enter
  requests, policy/config models, provenance, caches, logs, errors, or tests.

Policy, license, request, resource, and cache checks fail closed before client
invocation. Fable/Opus errors are terminal and never trigger silent fallback.

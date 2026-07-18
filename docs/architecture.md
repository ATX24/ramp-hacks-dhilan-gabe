# Architecture (demo-facing summary)

Distillery is a scientific experiment platform with a high-level run API—not a
serving product and not a low-level training RPC.

## One-sentence topology

`Python SDK → thin FastAPI control plane → immutable metadata + S3 → local subprocess or one SageMaker Training Job → S3 artifacts → proof evaluator / four-stage UI`

## Public resources

- **Dataset** — immutable content/split hashes, provenance, mixture counts
- **DistillationRun** — async state, sealed manifest hash, recipe resolution
- **ModelArtifact** — portable adapter (+ optional merged), checksums, load notes
- **ProofReport** — arms, gates, uncertainty, economics, `proof_status`

## Lifecycle

One sealed manifest → exactly one finite training job (or local entrypoint) →
artifacts + proof inputs → process exits. No persistent GPU worker, no
per-operation S3 RPC, no Step Functions in MVP.

## Recipes

| Recipe | Role |
| --- | --- |
| `sequence.v1` | Offline sequence distillation / completion-only QLoRA |
| `logit.v1` | Same-tokenizer white-box forward KL + hard CE |
| `auto` | Transparent resolver; may recommend `do_not_distill` |

Unsupported catalog methods return `RECIPE_NOT_IMPLEMENTED`. No silent downgrade.

## BYODT (techniques)

`src/distillery/techniques/` is a deep module at a clean seam: immutable
versioned technique descriptors, registry resolution without silent fallback,
and runtime adapters that yield the existing training plan/loss contract.
External techniques execute only in digest-pinned network-isolated containers.
See [byodt.md](./byodt.md).

## Four UI stages (exactly)

Curate → Synthesize → Train → Prove. `/` redirects to Curate. No deployment
language.

## Demo workstream placement

Workstream 8 owns `examples/**`, `docs/**`, and `tests/e2e/**`. It consumes
stable interfaces and fixtures; it does not patch trainers, backends, or
contracts to force a green demo.

## Safety defaults in examples

- Default mode is plan / dry-run.
- `plan_distillation` never launches training.
- `distill` requires an explicit acknowledgment flag + phrase.
- Offline mode requires checksum verification.

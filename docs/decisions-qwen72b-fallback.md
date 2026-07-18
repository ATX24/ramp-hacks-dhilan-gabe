# Decision: Qwen2.5-72B expensive quality fallback

Date: 2026-07-18  
Workstream: `feat/distillery-qwen72b-fallback`
Decision id: `decision_qwen72b_finance_world_v2_fallback_v2`

## Verdict

`Qwen/Qwen2.5-72B-Instruct` at
`495f39366efef23836d0cfae4fbe635880d2be31` has two closed roles:

1. `qwen72b_teacher`: a potential teacher for TinyFable students. This role is
   not ready until a non-empty, provenance-bound trajectory bundle exists.
2. `qwen72b_adapted_finance_fallback`: an expensive fallback adapted from
   executable `finance_world.v2` latent-oracle targets.

The adapted fallback has no identified larger teacher and is not a distilled
student. TinyFable remains the deployable small model.

## Identity and model license

- Architecture: `Qwen2ForCausalLM`, 80 layers, hidden size 8192, vocabulary
  size 152064.
- Weights: 37 safetensors shards, 145,412,407,296 bytes.
- `config.json` SHA-256:
  `14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29`.
- Tokenizer aggregate SHA-256:
  `8d9faaae11e51ab274be7eb785767c43a34f46b0c573632a6e2b8c0edbb90000`.
- Chat-template SHA-256:
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Model license: Qwen LICENSE AGREEMENT dated 2024-09-19, not Apache-2.0.
- Repo code license: MIT. It does not replace the model license.
- Distribution artifacts must include `QWEN_NOTICE.txt`. Base distributions use
  “Built with Qwen”; derived distributions use “Improved using Qwen”.
- Production use requires counsel review. The 100M-MAU commercial grant
  condition remains explicit.

Tokenizer compatibility is not represented by a boolean. Training readiness
must hash the four tokenizer files, derive the special-token IDs, and hash the
chat template for the 72B model and every registered target pair from live S3
object bodies.

## Training data

The rehearsal uses 24 `finance_world.v2` envelopes: six for each of transaction
review, variance analysis, merchant tagging, and cash reconciliation. Every
record binds the executable generator revision, `latent_state_hash`, full
envelope hash, prompt rendering, and oracle output rendering.

`finance_world.v1` generation remains unchanged. Hand-authored template targets
are not called oracle outputs.

Teacher trajectories are currently explicitly `absent`. No teacher-readiness
claim or TinyFable sequence-supervision artifact may seal until a non-empty
Qwen72B trajectory bundle is generated and verified.

## Precision and distribution

BF16 LoRA under DDP is infeasible because the base weights alone exceed one
A100-80GB. Four-bit NF4 QLoRA under DDP is the candidate, not an authorization.
Every rehearsal or full run requires a measured target-device probe bound to
the exact model identity, image digest, profile hash, tensor shape, and all
eight A100 ranks.

FlashAttention 2 is not packaged or claimed. The profile uses SDPA math,
disables flash and memory-efficient SDPA kernels, enables PyTorch deterministic
algorithms, and fixes sampler order and tensor shapes. Bitsandbytes NF4 is not
claimed bitwise reproducible across driver stacks.

Each DDP child sees one logical GPU. The parent launcher sets rank-local
visibility, enforces a 120-second NCCL timeout, kills peers when a rank dies,
and requires all-rank completion acknowledgement.

## Runtime and cost

- Materialization: one ephemeral `c5n.9xlarge`, at most three hours, $500 hard
  cap, one launch attempt, versioned S3 manifest, and partial-upload cleanup.
- Memory probe: one `ml.p4de.24xlarge`, at most 60 minutes,
  $100 hard cap, one launch attempt.
- Three-step rehearsal: one `ml.p4de.24xlarge`, 60-minute wall, $100 hard cap.
- Full run: one `ml.p4de.24xlarge`, 90-minute wall, $500 hard cap.

The rehearsal phase budget explicitly reserves channel verification, model
load, three updates, real PEFT save, fresh-base PEFT reload and forward probe,
artifact cleanup, and shutdown. Cost evidence includes active transfer and
p4de exposure. Retries are disabled.

## Current execution state

Execution is blocked. The committed bindings have no review-packet hashes, ECR
image, measured memory probe, transfer AMI, scoped instance profile, private
subnet, or no-ingress security group. The 72B S3 snapshot is not materialized.
This is the correct state until both independent reviews clear and every live
AWS probe succeeds.

The gate command returns nonzero when blocked and never accepts caller-supplied
readiness booleans:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/qwen72b/check_gates.py \
  --action rehearsal \
  --launch-name qwen72b-rehearsal-<id> \
  --confirm "EXECUTE QWEN72B REHEARSAL qwen72b-rehearsal-<id>"
```

Scoped tests:

```bash
PYTHONPATH=.:src .venv/bin/python -m pytest tests/qwen72b_fallback -q
```

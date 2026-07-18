# Decision record: Qwen2.5-72B expensive quality fallback

Date: 2026-07-18  
Workstream: isolated `feat/distillery-qwen72b-fallback`  
Decision id: `decision_qwen72b_oracle_sft_fallback_v1`

## Verdict

`Qwen/Qwen2.5-72B-Instruct` @ `495f39366efef23836d0cfae4fbe635880d2be31` is the
**expensive quality fallback** and a **powerful teacher for TinyFable tiers**.
**TinyFable remains the deployable small model.**

The 72B base adapted by synthetic oracle / sequence SFT is an
`oracle_sft_adapted_fallback`. It is **not** a distilled student unless a
separately identified larger teacher supplies its supervision. No such larger
teacher is used for the 72B adaptation path.

## Verified identity

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-72B-Instruct` |
| Revision | `495f39366efef23836d0cfae4fbe635880d2be31` |
| License | Qwen LICENSE AGREEMENT (2024-09-19), **not** Apache-2.0 |
| Output-use | Synthetic finance teaching + oracle-SFT for hackathon; counsel before prod; 100M-MAU commercial grant gate |
| `config.json` sha256 | `14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29` |
| Tokenizer aggregate sha256 | `8d9faaae11e51ab274be7eb785767c43a34f46b0c573632a6e2b8c0edbb90000` |
| Chat template sha256 | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` |
| Weight shards | 37 safetensors, ~135.4 GiB |
| Architecture | `Qwen2ForCausalLM` / `qwen2`, hidden=8192, layers=80, vocab=152064 |

Tokenizer / merges / vocab / chat-template hashes match the existing Qwen2.5
family pins used by 7B/14B/32B materialization. Compatible as a same-family
teacher for TinyFable students.

## Roles

1. **Teacher** — emits precomputed trajectories for smaller TinyFable tiers.
2. **Oracle/sequence-SFT adapted fallback** — post-trains the same 72B base on
   synthetic finance oracle completions. Scientific role:
   `oracle_sft_adapted_fallback`.

## Precision / distribution choice

Compared estimated peaks on one `ml.p4de.24xlarge` (8×A100 80GB):

- BF16 LoRA + DDP: does **not** fit (~145 GiB base weights alone).
- 4-bit QLoRA + DDP: fits under the 85% usable budget with FlashAttention 2,
  gradient checkpointing, packed completion-only sequences.
- FSDP2 / DeepSpeed ZeRO-3: **not** enabled by default. Only a sealed measured
  memory probe may authorize them.

Chosen: **4-bit QLoRA + DDP** for memory/throughput, not novelty.

## Budgets

| Action | Instance | Wall | Hard cap |
| --- | --- | --- | --- |
| S3 materialization | ephemeral `c5n.9xlarge` | ≤ 3 h projected | **$500** |
| 3-step rehearsal | `ml.p4de.24xlarge` | 20 min sealed | **$100** |
| Full run profile | `ml.p4de.24xlarge` | 30–90 min (sealed 90) | **$500** |

Teacher/tool trajectories are precomputed **outside** the warm training timer.

## Non-goals / safety

- Do not call adapted 72B a distilled student without a larger teacher.
- Synthetic finance examples only; no customer data.
- Do not interfere with active g5 smoke or 14B/32B materialization.
- Real AWS execute paths stay latch-blocked until integrity, ECR image, IAM,
  conflict, and cost gates all pass.

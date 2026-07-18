# Decision 0002: Tiered TinyFable portfolio

Status: superseded by Decision 0003
Date: 2026-07-18
Contract: `distillery.portfolio.plan.v2`

Decision 0003 replaces the unsafe pre-promotion registry, unmatched
sequence-specialist layout, and non-executable seed-23 language below. This
document remains only as the historical review input.

## Scope change

This version supersedes the original one-generalist-only Notion scope.
TinyFable remains a shared generalist by default, now independently within
each candidate tier. Task-specialist QLoRA adapters are explicit backups.
Neither specialists nor larger tiers may be silently routed. Test data cannot
select an arm, specialist, or tier.

Candidate names are metadata, not quality or marketing claims:

- Nano: fastest baseline, Qwen2.5-1.5B teacher to 0.5B student (494M label),
  screened on two `ml.g5.48xlarge` nodes.
- Core: priority larger candidate, Qwen2.5-7B teacher to 1.5B student,
  screened later on two `ml.p4de.24xlarge` nodes.
- Plus: stretch candidate, Qwen2.5-14B teacher to 3B student, screened later
  on two `ml.p4de.24xlarge` nodes.

The existing 1.5B snapshot can be reused as Core's student only after separate
student-role evidence is sealed. Role is part of the evidence hash. Revisions,
config, tokenizer, template, special-token, license, and output-use evidence
are bound. Logit KD requires tokenizer parity.

## Wave 1: exact Nano matrix

Seed is 17. Slot `s` maps to node `s % 2`, GPU `s // 2`.

```text
slot node gpu role        task(s)                       arm
0    0    0   generalist  all four                     oracle_sft
1    1    0   generalist  all four                     sequence_kd
2    0    1   generalist  all four                     logit_kd
3    1    1   generalist  all four                     ce_ablation
4    0    2   specialist transaction_review            sequence_kd
5    1    2   specialist transaction_review            logit_kd
6    0    3   specialist transaction_review            ce_ablation
7    1    3   specialist variance_analysis             sequence_kd
8    0    4   specialist variance_analysis             logit_kd
9    1    4   specialist variance_analysis             ce_ablation
10   0    5   specialist merchant_tagging              sequence_kd
11   1    5   specialist merchant_tagging              logit_kd
12   0    6   specialist merchant_tagging              ce_ablation
13   1    6   specialist cash_reconciliation           sequence_kd
14   0    7   specialist cash_reconciliation           logit_kd
15   1    7   specialist cash_reconciliation           ce_ablation
```

Each specialist `ce_ablation` is the same-task, same-seed, same-`logit.v1`
hard-CE control for `logit_kd`. Specialist `oracle_sft` is omitted because it
would duplicate that hard-target signal in the 12 specialist slots.

Core and Plus use the same 16-slot layout and seed on later A100 waves:
`wave_core_a100_screen_v1` and `wave_plus_a100_screen_v1`. Finalists replicate
at seed 23 with all other protocol inputs fixed.

## Exact gates

Every tier requires measured, image/model/protocol-bound probes for all four
generalist execution paths. Optional merged exports require a fifth probe.
Estimates cannot open the gate.

- QLoRA NF4 student; BF16 frozen/no-grad logit teacher.
- Length 512, completion 128, microbatch 1, accumulation 1.
- LoRA rank 8, alpha 16, dropout 0.05; vocab chunk 4096.
- Deterministic, network-isolated execution.
- A10G: measured 24 GiB capacity, peak no more than 85%, headroom at least
  4 GiB.
- A100: measured 80 GiB capacity, peak no more than 85%, headroom at least
  8 GiB.
- Tokenizer, license, output-use, and role-specific evidence must pass.

## Promotion

Specialists only become explicit backups when the 95% bootstrap interval's
lower bound for task gain is at least 0.02 against the same-tier generalist.
The generalist remains default.

Cross-tier comparison is generalist-to-generalist with no model-size prior:

- Quality-led: quality lower bound at least +0.02, throughput-ratio lower
  bound at least 0.80, cost-ratio upper bound at most 1.25.
- Efficiency-led: quality lower bound at least -0.005, throughput-ratio lower
  bound at least 1.10, cost-ratio upper bound at most 0.90.

Tokens/second and cost per 1,000 tokens stay `null` until measured with the
same harness and token count. There is no hidden scalar score.

## Launch-captain integration

1. Seal role-specific evidence for each teacher/student. Re-attest the 1.5B
   snapshot as a student for Core.
2. Supply sealed g5.48 and p4de regional pricing evidence, then call
   `build_plan(...)`; persist canonical bytes and `plan_sha256`.
3. Materialize each slot into a real `SealedRunManifest`, carrying its
   `protocol_sha256` and `manifest_binding_sha256`. The binding is not a
   substitute for the eventual manifest seal.
4. Keep each adapter and optional merge under its isolated bound URI. Publish
   the single registry bundle only after checksum verification.
5. Do not pass this plan directly to `stage_two_job_wave`: that helper is
   g5.48-only and requires one shared manifest identity. Add a versioned
   portfolio adapter that supports A100 profiles and typed per-task filtering
   before launch.
6. Preserve slot order and round-robin partitioning. Demo/inference must use a
   tier's generalist unless the user explicitly switches model and tier.

This contract does not authorize training and does not mutate v1 artifacts,
UI, API, inference, or containers.

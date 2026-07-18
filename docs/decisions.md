# Decision log (demo / reproducibility stream)

Source of truth: shared Distillery + TinyFable plan (2026-07-18). This file
records Workstream 8 interpretations only; product locks stay in the plan.

## Locked product decisions (inherited)

- Name/tagline: Distillery — "Smaller models. Proven economics."
- TinyFable is one finance generalist; primaries `transaction_review` and
  `variance_analysis`; backup `cash_reconciliation`.
- High-level run API; four public resources; one finite job per run.
- Implement `sequence.v1` and same-tokenizer `logit.v1`; `auto` is transparent.
- Qwen2.5 1.5B → 0.5B first; Qwen3 fallback only after failed teacher-gap pilot.
- Synthetic demo data; licensing/output-use still blocking.
- No deployment, Observe/proxy, multi-tenancy, or advanced-recipe execution.
- Proof may return `do_not_distill` or `insufficient_evidence`.

## Workstream 8 decisions (2026-07-18)

| Decision | Rationale |
| --- | --- |
| Example default is `--mode plan` | Prevent accidental GPU spend during rehearsal |
| Dual acknowledgment for `--mode train` (flag + phrase) | Fail loud; single flag is too easy to script by mistake |
| Local protocol adapter when SDK missing | Path-isolated delivery before Workstream 1 SDK merges |
| Adapter still refuses distill even with ack | This stream must not download models or submit jobs |
| Held-out selector seed default `17` | Matches locked screen seed; reproducible on stage |
| Public view strips `expected_output` / oracle | Avoid displaying gold labels as "model outputs" |
| Offline stub labeled `stub_not_a_benchmark` | Rehearsal packaging without fabricating proof |
| E2E tests use fakes, never real training | Contract coverage without GPU/AWS |

## Pending (not decided here)

- Exact model commit SHAs, tokenizer hashes, container digest
- Cheap off-the-shelf model ID/price pin
- Measured runtimes, VRAM, gross cost, teacher-gap, final `proof_status`

When those land, update pitch scripts to cite the report—not this file.

## Related isolated workstream

- [decisions-qwen72b-fallback.md](./decisions-qwen72b-fallback.md) — Qwen2.5-72B
  expensive quality fallback + TinyFable teacher roles (not a distilled student).

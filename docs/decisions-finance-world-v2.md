# Decision: finance_world.v1 vs finance_world.v2 (Merchant Tagging)

Date: 2026-07-18
Status: locked for next campaign; does not alter active v1 smoke

## Context

Notion labels Primary A `transaction_review`, Primary B `variance_analysis`, and
Primary C / fallback `merchant_tagging`, with `cash_reconciliation` as backup.

The sealed `finance_world.v1` path actually trains A + B + backup cash only.
Merchant Tagging was absent. Changing v1 in place would invalidate fixture hashes,
sealed corpora, and any in-flight smoke campaign.

## Decision

1. Keep `finance_world.v1` / `finance-proof.v1` immutable for the current smoke.
2. Add versioned `finance_world.v2` / `finance-proof.v2` that trains **one shared
   TinyFable generalist** across A/B/C and retains cash as diagnostic backup.
3. Never train merchant (or any task) as a separate specialist artifact.
   Campaign manifests set `specialist_routing=false` and
   `routing_policy=same_artifact_all_primary_tasks`.

## Declared v2 mixture

| Task | Weight | Role |
| --- | --- | --- |
| `transaction_review` | 0.35 | Primary A |
| `variance_analysis` | 0.35 | Primary B |
| `merchant_tagging` | 0.20 | Primary C |
| `cash_reconciliation` | 0.10 | Diagnostic backup |

Full corpus sizes: train 3840, validation 480, iid_test 960, ood_test 960
(**6240** total). Hamilton apportionment yields **1248** merchant examples
(≥1000). A/B each get 2184 vs v1’s 2340 (~6.7% lower), so merchant gets enough
weight without starving A/B. Cash stays at 10%.

Smoke v2 keeps 320/80/160 plumbing sizes with the same 35/35/20/10 mixture.

## Proof weighting

- v1 primary index: `0.45*txn + 0.45*var + 0.10*schema` (unchanged).
- v2 primary index: `0.35*txn + 0.35*var + 0.20*merchant + 0.10*schema`.
- Cash remains excluded from the primary index unless a pre-run decision log
  promotes it.

## How v1 smoke proceeds while v2 is next

- Active smoke/campaign tooling continues to pin `finance_world.v1`,
  `CORPUS_SMOKE` / `CORPUS_FULL`, `DEFAULT_FINANCE_MIXTURE`, and
  `finance-proof.v1`.
- v2 is opt-in via `CORPUS_SMOKE_V2` / `CORPUS_FULL_V2`, `FINANCE_MIXTURE_V2`,
  and `build_campaign_manifest(world="finance_world.v2", ...)`.
- Do not rebase or rewrite v1 golden fixtures (`tests/fixtures/finance_world_v1/`).
- After v1 smoke lands results, the next campaign should seal v2 corpora and
  protocol hashes before any trainable arm launches.

## Cherry-pick / integration notes

Safe to integrate (data/contracts/proof/tests/docs only):

- `src/distillery/contracts/tasks.py` (additive `merchant_tagging` + envelope v2)
- `src/distillery/contracts/budgets.py` (`PrimaryIndexWeightsV2`, `EvaluationBudgetV2`)
- `src/distillery/data/**` (v2 corpora, merchant latent/oracle/renderer/validators)
- `src/distillery/proof/protocol_v2.py`, merchant metrics, `compute_primary_index_v2`
- `src/distillery/training/batching.py` (`FINANCE_MIXTURE_V2` only; keep DEFAULT)
- `tests/data/**`, `tests/proof/**` merchant/v2 coverage, `tests/fixtures/finance_world_v2/`
- `docs/decisions-finance-world-v2.md`

Do **not** cherry-pick into active smoke paths without an explicit campaign cutover:

- `experiments/aws_smoke/**` pins / launchers
- `apps/api/**`, `apps/web/**`, `apps/inference/**`, container/image paths

When cutting over: point campaign dataset generation at `full_v2`, set proof
protocol id/sha to `finance-proof.v2`, and use `FINANCE_MIXTURE_V2` for the
shared sampler. Keep one student artifact for all primary tasks.

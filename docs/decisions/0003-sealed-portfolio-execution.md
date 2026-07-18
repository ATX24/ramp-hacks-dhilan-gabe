# Decision 0003: Sealed portfolio execution and selection

Status: accepted, plan-only
Date: 2026-07-18
Contract: `distillery.portfolio.plan.v3`
Supersedes: Decision 0002

## Decision

TinyFable Nano, Core, and Plus remain candidate metadata, not quality claims.
Each tier's shared four-task generalist is the default intent. Task specialists
start `planned`; they are not backups or routable models until sealed
validation and replication evidence clears the registry promotion gate.

The four tasks are Transaction Review, Variance Analysis, Merchant Tagging,
and Cash Reconciliation. Generalist manifests bind the complete four-task
`finance_world.v2` view. Every specialist manifest binds one sealed
task-filtered view. The shared dataset channel is a content-addressed bundle of
those views; the runtime must verify the selected view before training.

## Screening matrices

Every wave retains 16 physical slots. Slot `s` is always node `s % 2`, GPU
`s // 2`, including failed and not-started slots. Costs are allocated across
all 16 slots.

The seed-17 logit screen for each tier is:

```text
slot  role        scope                  arm                 state
0     generalist  all four               oracle_sft          planned
1     generalist  all four               sequence_kd         planned
2     generalist  all four               logit_kd            planned
3     generalist  all four               ce_ablation         planned
4-5   specialist transaction_review      logit_kd / CE       planned
6-7   specialist variance_analysis       logit_kd / CE       planned
8-9   specialist merchant_tagging        logit_kd / CE       planned
10-11 specialist cash_reconciliation     logit_kd / CE       planned
12-15 reserved    sequence screen         none                not_started
```

`ce_ablation` is the same-task, same-seed, same-dataset, same-`logit.v1`
control. The separate seed-17 sequence screen is:

```text
slot  scope                  arm                              state
0-1   transaction_review     sequence_kd / sequence CE       planned
2-3   variance_analysis      sequence_kd / sequence CE       planned
4-5   merchant_tagging       sequence_kd / sequence CE       planned
6-7   cash_reconciliation    sequence_kd / sequence CE       planned
8-15  reserved               none                             not_started
```

The sequence CE arm maps to the reviewed `oracle_sft` trainer vocabulary but
remains a distinct portfolio arm. Pair hashes exclude only the declared
treatment and bind recipe, seed, scope, and dataset view.

Nano uses `ml.g5.48xlarge` / `NVIDIA A10G`. Core and Plus use
`ml.p4de.24xlarge` / `NVIDIA A100 80GB`, resolved through the campaign hardware
profiles rather than copied A10G labels.

## Replication and selection

Finalists are locked from validation evidence before test access. The lock
contains the source wave hashes, exact treatment/control model IDs, validation
split hash, and a `None` test dataset field. `build_replication_wave(...)`
constructs real seed-23 runs. Seed 23 is part of each run ID, model ID,
artifact ID, slot protocol, and matrix hash, so it cannot collide with seed 17.

Promotion intervals are typed `ProofInterval` records. Each interval binds:

- treatment and comparator model, run, arm, and protocol;
- task or portfolio-primary scope;
- validation view and split;
- world clusters;
- world-cluster percentile bootstrap seed;
- exactly 10,000 resamples;
- a point estimate inside its interval;
- `finance-proof.v2`; and
- no test dataset.

Specialist promotion requires both seed-17 and seed-23 lower bounds of at
least +0.02 against a same-tier generalist. The contrast must clear a
preregistered tier/task omnibus gate and Holm-Bonferroni family. The generalist
remains default and the promoted specialist is explicit-switch-only.

Tier promotion has no size prior. Throughput and cost are required
measurements, not nullable claims. Candidate and incumbent measurements bind
the same benchmark hardware, accelerator, image, runtime, harness, token
count, and request count. Different training hardware is not used as
cross-tier performance evidence.

## Evidence and budgets

Readiness binds teacher and student role evidence, license evidence,
output-use evidence, runtime image evidence, and measured memory records.
Boolean readiness claims cannot open a gate. Required measured paths are
oracle SFT, sequence KD, logit KD, and CE ablation. Peak memory is at most 85%
with at least 4 GiB A10G or 8 GiB A100 headroom.

The training contract imports `TrainingBudget` and `ProofGates`: full
portfolio knobs, seeds 17/23, `finance-proof.v2`, and 10,000 bootstrap
resamples. V1 smoke values and mixed proof protocols are rejected.

Pricing binds source URI, region, instance type, current micro-USD price,
attestor, timestamps, evidence-byte hash, and byte length. The plan enforces
per-run, per-wave, experiment, and fixed $10,000 account ceilings. Throughput
and cost per 1,000 tokens remain unknown until benchmark evidence exists.

## Launch-captain procedure

These are local, deterministic steps. None submits AWS work:

1. Build role evidence, model pairs, dataset views, runtimes, current pricing,
   and `build_plan(...)`.
2. Build and verify `ReadinessEvidence` for a tier.
3. Call `materialize_slot(...)` for each active slot and persist each returned
   `SealedRunManifest`. Do not synthesize manifests for not-started slots.
4. Call `stage_portfolio_wave(...)`. It uses the reviewed
   `stage_campaign_bundle(...)` interface twice and preserves logical
   node/GPU slots. It does not call the g5-only `stage_two_job_wave(...)`.
5. Verify image inventory with `ContainerStagingEvidence`. The image must
   stage:
   `experiments.aws_smoke.campaign_index`,
   `experiments.aws_smoke.campaign_orchestrator`,
   `experiments.aws_smoke.train`, and
   `experiments.portfolio.task_filter_runtime`.
   It must attest integration that invokes the task-view resolver before the
   trainer. Current shared image work owns that integration; this change does
   not duplicate container edits.
6. Call `build_portfolio_launch_plan(...)` to produce two dry-run
   `CreateTrainingJob` request dictionaries. Review them. This repository
   exposes no submit function.
7. Record every terminal slot with `build_execution_ledger(...)`. Preserve
   failed and not-started identities and allocate actual parent costs across
   all physical slots.
8. After selection, lock finalists and construct seed-23 replication. Test
   data remains sealed until selection is complete.
9. Keep registry entries planned until checksummed artifacts and proof reports
   exist. `promote_specialist_registry(...)` and `publish_registry(...)` fail
   closed. Building a publishable object still does not upload it.

Executable regressions are in `tests/portfolio/test_plan.py`,
`tests/portfolio/test_materialize_campaign.py`, and
`tests/portfolio/test_selection_registry.py`.

## Review disposition

1. Specialists are planned until sealed promotion evidence.
2. Sequence and logit specialist treatments have same-recipe controls.
3. Seed-23 waves bind new run/model/artifact/protocol identities.
4. Typed intervals bind comparator, validation clusters, 10,000 resamples,
   and hierarchical multiplicity control.
5. Tier promotion requires measured, same-harness economics with hardware
   confounds rejected.
6. Generalists cover all tasks; specialist views bind exactly one task.
7. Portfolio protocol constants reject v1 smoke/proof mixing.
8. Slots materialize into validated `SealedRunManifest` objects.
9. g5 and p4de adapters resolve reviewed campaign hardware labels.
10. Pricing evidence and all four ceiling levels are sealed.
11. Planned registry entries have no serving URIs or live/default/backup
    status; promotion and publication are gated.
12. Readiness verifies evidence hashes instead of accepting true claims.
13. Execution ledgers preserve all 16 slot identities and costs.
14. Launch preflight is executable and explicitly gates shared image staging.

This decision does not authorize training, publish a registry, modify UI/API,
or mutate v1 artifacts.

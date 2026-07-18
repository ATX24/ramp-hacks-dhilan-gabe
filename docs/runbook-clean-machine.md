# Clean-machine runbook (local rehearsal)

Rehearse Curate → Synthesize → Train-planned → Prove on a fresh shell without
training, model downloads, or SageMaker jobs.

## Prerequisites

- Python 3.11+
- Repo checkout with `tests/fixtures/finance_world_v1/golden.jsonl`
- Optional: `uv` or `pip` to install the package in editable mode for contract imports

No AWS credentials are required for plan-only mode.

## One-command plan-only rehearsal

From the repository root:

```bash
python examples/finance_generalist.py --mode plan --json
```

Expected: JSON with four stages; `training_launched: false`;
`proof_status` of `insufficient_evidence` (or `do_not_distill` if the resolver
recommends it). Real benchmarks remain pending.

## Held-out demo cases (fixed seed)

```bash
python examples/held_out_selector.py \
  --dataset tests/fixtures/finance_world_v1/golden.jsonl \
  --seed 17 \
  --public-only
```

Only `iid_test` / `ood_test` / `test` rows are eligible. Train/validation gold
never appears in the public view.

## Artifact verification (fixture smoke)

```bash
python examples/verify_artifacts.py \
  --root tests/fixtures/finance_world_v1 \
  --fixture-manifest tests/fixtures/finance_world_v1/fixture_manifest.json
```

## Offline / precomputed fallback

```bash
python examples/offline_fallback.py \
  --artifact-root /tmp/distillery-offline-stub \
  --write-stub

python examples/offline_fallback.py \
  --artifact-root /tmp/distillery-offline-stub
```

Stub packages are labeled `stub_not_a_benchmark`. Replace with a real
checksum-verified run package when available (see
[offline-precomputed-fallback.md](./offline-precomputed-fallback.md)).

## Safety: training cannot start accidentally

```bash
# This must fail (exit 3) without the acknowledgment phrase:
python examples/finance_generalist.py --mode train

# Still fails on the local adapter even with acknowledgment (no training in W8):
python examples/finance_generalist.py --mode train \
  --i-acknowledge-training-will-launch \
  --ack-phrase I_ACKNOWLEDGE_THIS_WILL_LAUNCH_TRAINING
```

## E2E contract tests (fakes, no GPU)

```bash
python -m pytest tests/e2e -q
```

## Honest claims checklist

- Say the data is synthetic.
- Say proof status is whatever the report shows; do not upgrade it verbally.
- Say serving savings are projected unless measured.
- Say real multi-arm benchmarks are pending until artifacts land.

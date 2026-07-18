# Offline / precomputed fallback

Use when network, AWS quota, UI live status, or cold image pulls would burn the
two-minute pitch.

## Priority in the shared fallback hierarchy

1. Merchant tagging (only if promoted)
2. Local-only sealed-manifest path
3. Sequence-only (logit unavailable)
4. **Precomputed real run** ← this doc
5. Smaller smoke corpus / model pair under a new protocol hash
6. Static proof report from real artifacts

## Operator steps

1. Obtain a completed run or proof package (never hand-authored metrics).
2. Verify checksums:

```bash
python examples/verify_artifacts.py --root "$ARTIFACT_ROOT"
```

3. Present offline mode:

```bash
python examples/offline_fallback.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --report "$ARTIFACT_ROOT/report.json"
```

4. Still run the fixed-seed held-out selector live (public inputs only):

```bash
python examples/held_out_selector.py --seed 17 --public-only
```

5. Label the UI / slides: **PRECOMPUTED ARTIFACTS — checksum verified; not a live run.**

## Rehearsal stub (not for judging claims)

```bash
python examples/offline_fallback.py \
  --artifact-root /tmp/distillery-offline-stub \
  --write-stub
```

The stub report uses `proof_status=insufficient_evidence` and
`economics.label=stub_not_a_benchmark`.

## Spoken limitations (required)

- Synthetic data only for the demo corpus.
- Precomputed mode does not retrain or resubmit SageMaker jobs.
- Serving cost rows are projected unless the report marks them measured.
- If seed-23 or arms are missing, status cannot be `proved`.

# Artifact and hash verification

Immutable artifacts are content-addressed. Demo and pitch materials must only
show packages that pass checksum verification.

## Layout (planned)

```text
<run_or_proof_root>/
  integrity/SHA256SUMS
  report.json                    # proof packages
  evaluation/predictions.jsonl
  model/adapter/...
  manifest.json
  ...
```

`SHA256SUMS` uses the common `sha256sum` format:

```text
<64-hex>  relative/path
```

## Verify a real (or stub) package

```bash
python examples/verify_artifacts.py --root /path/to/artifacts
```

Exit `0` only when every listed file exists and digests match.

Strict mode (no unexpected files):

```bash
python examples/verify_artifacts.py --root /path/to/artifacts --strict-tree
```

## Verify the frozen golden fixture

```bash
python examples/verify_artifacts.py \
  --root tests/fixtures/finance_world_v1 \
  --fixture-manifest tests/fixtures/finance_world_v1/fixture_manifest.json
```

Pinned digests live in `fixture_manifest.json` (`files.*.sha256` and
`example_sha256`). Any post-freeze data fix must mint `finance_world.v2`.

## Offline demo gate

`examples/offline_fallback.py` refuses to present a package when verification
fails (unless `--allow-unverified`, which still labels `verified: false`).

## What verification does *not* prove

- Scientific `proof_status` (quality/economics gates)
- That a SageMaker job was authentic (only that bytes match the sums file)
- Production generalization on customer data

Keep verbal claims aligned: checksum-verified ≠ proved.

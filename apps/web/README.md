# Distillery web UI

Five-stage Next.js app for Distillery / TinyFable:
**Curate → Synthesize → Train → Prove → Demo**.

Fixture-backed by default. The Demo / Playground can call a typed live inference
gateway when `NEXT_PUBLIC_DISTILLERY_INFERENCE_URL` is set; otherwise live mode
surfaces an explicit unavailable state and never fabricates inference.

## Commands

```bash
pnpm install
pnpm dev
pnpm lint
pnpm typecheck
pnpm test
pnpm test:smoke
pnpm build
```

Playwright definitions are separate from Vitest and require preinstalled browser binaries:

```bash
pnpm test:e2e
```

Do not install browser binaries as part of routine fixture verification.

## Fixture modes

Append `?mode=` with one of:

`default` · `no_training_yet` · `skipped_synthesis` · `precomputed` · `proved` · `do_not_distill` · `failed_quality` · `failed_economics` · `insufficient_evidence` · `error` · `unavailable` · `loading` · `fetch_failure`

Run references persist per validated mode under `distillery.run_ref.<mode>`. A validated
explicit `run` query parameter wins over local storage.

## Demo / Playground URL state

On `/demo`, shareable query params include:

- `task` — `transaction_review` | `variance_analysis` | `cash_reconciliation`
- `models` — comma-separated registry `model_id` values
- `example` — prepopulated example id
- `runMode` — `single` | `compare`
- `infer` — `fixture_preview` | `live`

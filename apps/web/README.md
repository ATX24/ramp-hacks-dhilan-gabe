# Distillery web UI

Four-stage Next.js app for Distillery / TinyFable: **Curate → Synthesize → Train → Prove**.

Fixture-backed only. No live API calls or active jobs.

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

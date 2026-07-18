# Distillery web UI (Best-of-N candidate C)

Playground-first Distillery experience for lay judges:
**Demo (before/after) → Curate → Synthesize → Train → Prove**.

Built with shadcn/ui and official shadcn AI Elements. Root entry redirects to `/demo`.
Fixture-backed by default with explicit fixture/live labeling and an honest training
event adapter for the at-a-glance teaching card.

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

## Fixture modes

Append `?mode=` with one of:

`default` · `no_training_yet` · `skipped_synthesis` · `precomputed` · `proved` · `do_not_distill` · `failed_quality` · `failed_economics` · `insufficient_evidence` · `error` · `unavailable` · `loading` · `fetch_failure`

## Demo / Playground URL state

On `/demo`, shareable query params include:

- `task` — `transaction_review` | `variance_analysis` | `cash_reconciliation`
- `models` — comma-separated registry `model_id` values
- `example` — prepopulated example id
- `runMode` — `single` | `compare`
- `infer` — `fixture_preview` | `live`

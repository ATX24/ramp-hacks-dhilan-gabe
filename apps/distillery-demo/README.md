# Distillery demo UI

Stable saved-data Distillery demo (`98ed4a6`). Self-contained Next.js app. Root redirects to `/demo`.

Responses are deterministic saved demo data, not live inference.

## Install / build

```bash
pnpm --dir apps/distillery-demo install
pnpm --dir apps/distillery-demo typecheck
pnpm --dir apps/distillery-demo lint
pnpm --dir apps/distillery-demo build
```

## Run (Portless)

```bash
cd apps/distillery-demo
portless distillery-demo --force pnpm exec next dev
```

Open:

- `https://distillery-demo.localhost:1355/demo?mode=precomputed`
- `https://distillery-demo.localhost:1355/train`

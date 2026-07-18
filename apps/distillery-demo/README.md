# Distillery demo UI

Password-protected Distillery demo from `1351e09`. Responses use saved demo data, not live inference.

## Required environment

- `DISTILLERY_DEMO_PASSWORD`
- `DISTILLERY_AUTH_SECRET`

Set a password without recording it in shell history, then generate a signing secret:

```zsh
read -rs "DISTILLERY_DEMO_PASSWORD?Demo password (12+ characters): "
printf "\n"
export DISTILLERY_DEMO_PASSWORD
export DISTILLERY_AUTH_SECRET="$(openssl rand -hex 32)"
```

## Run

From the repository root:

```bash
pnpm --dir apps/distillery-demo install
cd apps/distillery-demo
portless distillery-demo --force pnpm exec next dev
```

Open `https://distillery-demo.localhost:1355/login`, then continue to `/demo?mode=precomputed`.

# Anthropic 2 website

Research-lab website for Anthropic 2, TinyFable, and Distillery. The site is a static public-facing Next.js application intended for Vercel at `anthropic2.dev`.

## Local development

```bash
npm install
npm run dev
```

Open `http://localhost:3000` for the lab site and `http://localhost:3000/distillery` for the Distillery page.

## Production checks

```bash
npm run typecheck
npm run lint
npm run build
```

## Vercel setup

1. Import `ATX24/ramp-hacks-dhilan-gabe` in Vercel.
2. Set the project root directory to `web`.
3. Keep Framework Preset set to Next.js.
4. Keep the default install and build commands.
5. Add `anthropic2.dev` in Project Settings, then apply the DNS records Vercel provides.

The site has no runtime secrets, database, or server-side product dependency. PDFs and the release note are served from `public/`.

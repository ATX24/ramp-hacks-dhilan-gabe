# Best-of-N UI candidate C — rationale

## Concept

Outcome first, process second. Judges land on the Demo playground, compare an
original vs taught smaller model on a familiar finance task, then scroll into
Curate → Synthesize → Train → Prove as the explanation of how that result was made.

## Why this shape

- Lay judges understand card review / budget miss / cash match faster than
  manifests, recipes, or KL.
- Before/after cards make compression and spend tangible before any ML jargon.
- Progressive disclosure keeps hashes, CIs, and telemetry available without
  forcing them into the first read.
- An honest event adapter labels fixture / prior-run / live explicitly so the
  teaching card never invents a live job.

## Stack choices

- shadcn/ui primitives for shell, cards, toggles, progress, accordion.
- Official AI Elements for suggestions, messages, and result presentation.
- Brand tokens matched to root `web/` (paper, ink, orange, Georgia serif).
- Minimal bespoke CSS beyond brand tokens and stage-nav helpers.

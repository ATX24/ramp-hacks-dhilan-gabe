# Two-minute pitch and demo sequence

Tagline: **Smaller models. Proven economics.**

## Clock

### 0:00–0:20 · Problem and Curate

"Existing traces may already be distillation data."

Open Curate (or `python examples/finance_generalist.py --mode plan`). Show the
one synthetic finance world, 45/45/10 mixture intent, content hash, and
group/OOD isolation story. Mention the latent oracle defines truth—not the
teacher.

### 0:20–0:40 · Synthesize

Show imported/oracle/teacher label counts. Teacher calls are zero when valid
responses already exist; used only for missing/rejected labels or deliberate
augmentation.

### 0:40–1:05 · Train (planned or completed)

Show `recipe="auto"` resolving transparently, pinned Qwen pair intent
(1.5B → 0.5B), tokenizer/memory/license gates, and either:

- a completed real one-job SageMaker run artifact, or
- plan-only / precomputed mode, clearly labeled.

Load the same **TinyFable** artifact story for both primary tasks. Do not imply
deployment.

### 1:05–1:35 · Generalist behavior

On fixed-seed held-out cases (`--seed 17`, public inputs only), walk
`transaction_review` (GL, balanced journal, policy action) then
`variance_analysis` (profit impact, top drivers). Side-by-side base / teacher /
sequence / logit only if those predictions exist in the verified package.
Fallback: merchant tagging (if promoted) or cash reconciliation.

### 1:35–1:55 · Prove

Show joint exact, invariants, IID/OOD, paired CIs, seeds, latency/VRAM/GPU
hours, gross cost, utilization-sensitive break-even—**only values present in
the report**. Highlight actual `proof_status`. If pending, say
`insufficient_evidence` and what evidence is missing.

### 1:55–2:00 · Close

"We are not selling cheaper training tokens. We are making the distill-or-don't
decision reproducible, easy, and portable. Smaller models. Proven economics."

## Contingency lines

| Failure | Spoken fix |
| --- | --- |
| Live AWS down | "Checksum-verified precomputed run; same proof package." |
| Logit unavailable | "Sequence path is real; logit gated on tokenizer/memory." |
| No finalist seeds | "Status is insufficient_evidence until seed 23 lands." |

## Commands for timed rehearsal

```bash
python examples/finance_generalist.py --mode plan
python examples/held_out_selector.py --seed 17 --public-only
# optional:
python examples/offline_fallback.py --artifact-root "$ARTIFACT_ROOT"
```

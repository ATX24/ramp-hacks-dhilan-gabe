import type { FinanceTaskId } from "@/lib/demo/types";
import type { StageId } from "@/lib/types";

/** Layperson-first labels. Canonical stage names stay available as technical subtitles. */
export const STAGE_PLAIN: Record<
  StageId,
  { plain: string; why: string; description: string }
> = {
  demo: {
    plain: "Try the model",
    why: "See the smaller model on a real finance task before reading how it was made.",
    description:
      "Pick a familiar finance task, compare the original model with the smaller one, and read speed, quality, and cost in plain language.",
  },
  curate: {
    plain: "Gather examples",
    why: "Frozen examples are the only evidence the smaller model is allowed to learn from.",
    description:
      "Bring in valid finance examples, freeze the held-out test set, and lock every source so nothing can quietly change later.",
  },
  synthesize: {
    plain: "Fill gaps",
    why: "Missing or rejected labels get filled on purpose — never silently invented elsewhere.",
    description:
      "Ask a larger teacher model only for labels that are missing, rejected, or deliberately expanded.",
  },
  train: {
    plain: "Teach the smaller model",
    why: "One finite teaching job with a cost ceiling — not an open-ended experiment.",
    description:
      "Choose how to teach the smaller model, then run one sealed job with a clear spend limit.",
  },
  prove: {
    plain: "Check the result",
    why: "A finished teaching job is not success. Quality and cost still have to win.",
    description:
      "Compare the smaller model with rules, its starting point, its teacher, and a cheap off-the-shelf option.",
  },
};

export const TASK_PLAIN: Record<
  FinanceTaskId,
  { title: string; blurb: string; judgePrompt: string }
> = {
  transaction_review: {
    title: "Review a company card charge",
    blurb: "Decide whether a purchase looks okay, risky, or needs a human look.",
    judgePrompt: "Should finance approve, flag, or escalate this charge?",
  },
  variance_analysis: {
    title: "Explain a budget miss",
    blurb: "Say why actual spend drifted from plan and what to check next.",
    judgePrompt: "What drove the variance, and is it actionable?",
  },
  cash_reconciliation: {
    title: "Match cash to the books",
    blurb: "Reconcile bank movements with ledger lines and call out exceptions.",
    judgePrompt: "Do the cash movements line up, or is something missing?",
  },
};

export const MODEL_ROLE_PLAIN: Record<string, { title: string; blurb: string }> = {
  student_base: {
    title: "Original smaller model",
    blurb: "Starting point before Distillery teaching.",
  },
  sequence_kd: {
    title: "Taught smaller model",
    blurb: "Smaller model after Distillery teaching (sequence method).",
  },
  logit_kd: {
    title: "Taught smaller model",
    blurb: "Smaller model after Distillery teaching (logit method).",
  },
  oracle_sft: {
    title: "Upper-bound teacher style",
    blurb: "Stronger reference trained on gold answers.",
  },
  ce_ablation: {
    title: "Control recipe",
    blurb: "Matched control used to keep the comparison honest.",
  },
  promoted_winner: {
    title: "Promoted smaller model",
    blurb: "The version Distillery would actually serve if proof passed.",
  },
};

export function plainModelLabel(armId: string, fallback: string): string {
  return MODEL_ROLE_PLAIN[armId]?.title ?? fallback;
}

export function formatUsdPlain(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "unknown";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: value >= 100 ? 0 : 2,
  }).format(value);
}

export function formatLatencyPlain(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "unknown";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function formatQualityPlain(score: number | null | undefined): string {
  if (score === null || score === undefined) return "unknown";
  const pct = Math.round(score * 100);
  if (pct >= 90) return `${pct}% · strong match`;
  if (pct >= 70) return `${pct}% · usable`;
  if (pct >= 40) return `${pct}% · mixed`;
  return `${pct}% · weak`;
}

import type { DemoModelArmId, FinanceTaskId } from "@/lib/demo/types";

function clone<T>(value: T): T {
  return structuredClone(value);
}

/**
 * Deterministic labeled fixture-preview outputs.
 * These are not live inference — callers must stamp provenance="fixture_preview".
 */
export function buildFixturePreviewOutput(
  armId: DemoModelArmId,
  task: FinanceTaskId,
  gold: Record<string, unknown> | null,
  input: Record<string, unknown>,
): Record<string, unknown> {
  if (gold) {
    return mutateFromGold(armId, task, clone(gold));
  }
  return synthesizeWithoutGold(armId, task, input);
}

function mutateFromGold(
  armId: DemoModelArmId,
  task: FinanceTaskId,
  gold: Record<string, unknown>,
): Record<string, unknown> {
  switch (armId) {
    case "oracle_sft":
    case "promoted_winner":
    case "sequence_kd":
      return {
        ...gold,
        confidence: clampConfidence(numberOr(gold.confidence, 0.85) + confidenceBoost(armId)),
      };
    case "logit_kd":
      return {
        ...gold,
        confidence: clampConfidence(numberOr(gold.confidence, 0.84)),
      };
    case "ce_ablation":
      return degradeSlightly(task, gold);
    case "student_base":
      return degradeBase(task, gold);
    default: {
      const _exhaustive: never = armId;
      return _exhaustive;
    }
  }
}

function confidenceBoost(armId: DemoModelArmId): number {
  switch (armId) {
    case "oracle_sft":
      return 0.02;
    case "promoted_winner":
      return 0.015;
    case "sequence_kd":
      return 0.01;
    default:
      return 0;
  }
}

function degradeSlightly(
  task: FinanceTaskId,
  gold: Record<string, unknown>,
): Record<string, unknown> {
  const next: Record<string, unknown> = {
    ...gold,
    confidence: clampConfidence(numberOr(gold.confidence, 0.8) - 0.05),
  };
  if (task === "transaction_review" && next.policy_action === "reject") {
    // Ablation sometimes softens to review — still schema-valid.
    return { ...next, policy_action: "review" };
  }
  return next;
}

function degradeBase(
  task: FinanceTaskId,
  gold: Record<string, unknown>,
): Record<string, unknown> {
  const next: Record<string, unknown> = {
    ...gold,
    confidence: clampConfidence(numberOr(gold.confidence, 0.7) - 0.18),
  };
  if (task === "transaction_review") {
    return {
      ...next,
      policy_action: "review",
      rule_ids: ["BASELINE-HEURISTIC"],
    };
  }
  if (task === "variance_analysis") {
    const drivers = Array.isArray(next.top_drivers) ? [...next.top_drivers] : [];
    return {
      ...next,
      top_drivers: drivers.slice(0, Math.max(1, drivers.length - 1)),
      other_impact_minor: numberOr(next.profit_impact_minor, 0),
      // Intentionally break arithmetic closure to show weaker base quality.
      // Keep task/schema fields so schema validation can still pass fields,
      // but score against gold will be 0.
      rule_ids: ["BASELINE-VAR"],
    };
  }
  if (task === "cash_reconciliation") {
    return {
      ...next,
      status: "exceptions",
      confidence: 0.55,
      exceptions: [
        {
          type: "unexplained",
          event_ids: ["baseline_gap"],
          amount_minor: 1,
        },
      ],
      difference_minor: 1,
      adjusted_book_balance_minor: numberOr(next.adjusted_book_balance_minor, 0),
      adjusted_bank_balance_minor: numberOr(next.adjusted_bank_balance_minor, 0) - 1,
    };
  }
  return next;
}

function synthesizeWithoutGold(
  armId: DemoModelArmId,
  task: FinanceTaskId,
  input: Record<string, unknown>,
): Record<string, unknown> {
  const confidence =
    armId === "student_base" ? 0.42 : armId === "ce_ablation" ? 0.61 : 0.7;
  if (task === "transaction_review") {
    const amount = numberOr(input.amount_minor, 0);
    const account =
      Array.isArray(input.gl_candidates) && typeof input.gl_candidates[0] === "string"
        ? input.gl_candidates[0]
        : "9999";
    return {
      schema_version: "transaction_review.v1",
      task: "transaction_review",
      gl_account: account,
      journal_entry: [
        { account, amount_minor: amount, side: "debit" },
        { account: "2100", amount_minor: amount, side: "credit" },
      ],
      policy_action: "review",
      rule_ids: ["FIXTURE-PREVIEW"],
      evidence: [{ source_id: "txn", field: "amount_minor", value: String(amount) }],
      confidence,
    };
  }
  if (task === "variance_analysis") {
    return {
      schema_version: "variance_analysis.v1",
      task: "variance_analysis",
      profit_impact_minor: -1000,
      direction: "unfavorable",
      top_drivers: [{ driver_id: "preview", impact_minor: -1000, rank: 1 }],
      other_impact_minor: 0,
      rule_ids: ["FIXTURE-PREVIEW"],
      evidence_ids: ["preview"],
      confidence,
    };
  }
  return {
    schema_version: "cash_reconciliation.v1",
    task: "cash_reconciliation",
    status: "exceptions",
    matched_groups: [],
    exceptions: [
      {
        type: "unexplained",
        event_ids: ["preview"],
        amount_minor: 0,
      },
    ],
    adjusted_book_balance_minor: numberOr(input.book_balance_minor, 0),
    adjusted_bank_balance_minor: numberOr(input.bank_balance_minor, 0),
    difference_minor:
      numberOr(input.book_balance_minor, 0) - numberOr(input.bank_balance_minor, 0),
    confidence,
  };
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clampConfidence(value: number): number {
  return Math.min(1, Math.max(0, Number(value.toFixed(4))));
}

/** Stable faux latency for fixture preview (ms), deterministic per model+example. */
export function fixturePreviewLatencyMs(modelId: string, exampleId: string | null): number {
  const seed = `${modelId}:${exampleId ?? "custom"}`;
  let hash = 0;
  for (let i = 0; i < seed.length; i += 1) {
    hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
  }
  return 35 + (hash % 40);
}

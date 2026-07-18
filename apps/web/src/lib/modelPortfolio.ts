import type {
  DemoModelEntry,
  DemoPortfolioEntry,
  FinanceTaskId,
} from "@/lib/demo/types";
import type { StageBundle } from "@/lib/types";

export type DistillationPriority = "quality" | "speed" | "cost";

export const SAFE_EXAMPLE_GOALS = [
  "Review finance transactions and flag policy exceptions",
  "Explain budget versus actual variance for finance teams",
  "Reconcile cash balances and surface unmatched entries",
] as const;

export const DISTILLATION_PRIORITIES: readonly {
  id: DistillationPriority;
  label: string;
  description: string;
}[] = [
  {
    id: "quality",
    label: "Quality",
    description: "Auto will favor the strongest retained accuracy.",
  },
  {
    id: "speed",
    label: "Speed",
    description: "Auto will favor the shortest bounded run.",
  },
  {
    id: "cost",
    label: "Cost",
    description: "Auto will favor the lowest safe spending limit.",
  },
] as const;

const ALL_FINANCE_TASKS: readonly FinanceTaskId[] = [
  "transaction_review",
  "variance_analysis",
  "cash_reconciliation",
];

const SPECIALISTS: readonly {
  portfolio_id: string;
  display_name: string;
  task: FinanceTaskId;
  adapter_id: string;
  purpose: string;
}[] = [
  {
    portfolio_id: "tinyfable_transaction_specialist",
    display_name: "Transaction review specialist",
    task: "transaction_review",
    adapter_id: "adapter_fixture_transaction_review_v1",
    purpose: "This backup handles transaction policy and journal entries.",
  },
  {
    portfolio_id: "tinyfable_variance_specialist",
    display_name: "Budget variance specialist",
    task: "variance_analysis",
    adapter_id: "adapter_fixture_variance_analysis_v1",
    purpose: "This backup explains the causes of budget variance.",
  },
  {
    portfolio_id: "tinyfable_cash_specialist",
    display_name: "Cash matching specialist",
    task: "cash_reconciliation",
    adapter_id: "adapter_fixture_cash_reconciliation_v1",
    purpose: "This backup handles cash matches, timing, and exceptions.",
  },
] as const;

export function buildModelPortfolio(
  bundle: StageBundle,
  models: readonly DemoModelEntry[],
): DemoPortfolioEntry[] {
  const generalistModel =
    models.find((model) => model.arm_id === "promoted_winner") ??
    models.find((model) => model.arm_id === "sequence_kd") ??
    models.find((model) => model.arm_id === "student_base") ??
    null;

  const generalist: DemoPortfolioEntry = {
    portfolio_id: "tinyfable_generalist",
    display_name: "TinyFable Generalist",
    role: "generalist",
    recommended: true,
    task_scope: ALL_FINANCE_TASKS,
    base_model_id: bundle.plan.student.id,
    adapter_id: bundle.artifact?.artifact_id ?? null,
    artifact_id: bundle.artifact?.artifact_id ?? null,
    availability: generalistModel?.serving.availability ?? "unavailable",
    selection_policy: "auto_default",
    purpose:
      "This recommended model handles transaction review, budget variance, and cash matching.",
  };

  return [
    generalist,
    ...SPECIALISTS.map(
      (specialist): DemoPortfolioEntry => ({
        portfolio_id: specialist.portfolio_id,
        display_name: specialist.display_name,
        role: "specialist",
        recommended: false,
        task_scope: [specialist.task],
        base_model_id: bundle.plan.student.id,
        adapter_id: specialist.adapter_id,
        artifact_id: null,
        availability: "fixture_preview",
        selection_policy: "explicit_only",
        purpose: specialist.purpose,
      }),
    ),
  ];
}

export interface AutoDistillationPlan {
  requestedRecipe: "auto";
  resolvedTechnique: string;
  modelPortfolioId: string;
  budgetCeilingUsd: number;
  etaMinutes: { low: number; high: number };
  proofProtocol: string;
}

export function resolveAutoDistillationPlan(
  bundle: StageBundle,
  priority: DistillationPriority,
  portfolioId = "tinyfable_generalist",
): AutoDistillationPlan {
  const estimatedHigh = bundle.plan.cost.estimate_high_usd;
  const budgetCeilingUsd =
    priority === "quality"
      ? bundle.plan.cost.max_run_usd
      : priority === "speed"
        ? Math.min(bundle.plan.cost.max_run_usd, Math.ceil(estimatedHigh * 1.25))
        : Math.min(
            bundle.plan.cost.max_run_usd,
            Math.ceil(bundle.plan.cost.estimate_low_usd * 1.25),
          );

  return {
    requestedRecipe: "auto",
    resolvedTechnique: bundle.plan.resolved_recipe ?? "pending capability checks",
    modelPortfolioId: portfolioId,
    budgetCeilingUsd,
    etaMinutes: bundle.plan.wall_time_minutes_estimate,
    proofProtocol:
      "The saved checks cover familiar and unfamiliar examples, response time, capacity, and cost.",
  };
}

import { getDemoExample } from "@/lib/demo/examples";
import type { DemoExample, FinanceTaskId } from "@/lib/demo/types";

export interface DemoExamplePreset {
  id: string;
  exampleId: string;
  task: FinanceTaskId;
  label: string;
  inferenceInput: string;
  trainingInput: string;
}

export const DEMO_EXAMPLE_CATALOG = [
  {
    id: "server-purchase",
    exampleId: "ex_txn_hard_001",
    task: "transaction_review",
    label: "Server purchase",
    inferenceInput:
      "Review a $50,000 server purchase from Dell Technologies. Decide whether finance should approve, flag, or escalate it.",
    trainingInput:
      "Teach the smaller model to review large equipment purchases and apply the approval policy.",
  },
  {
    id: "saas-renewal",
    exampleId: "ex_demo_txn_saas_001",
    task: "transaction_review",
    label: "SaaS renewal",
    inferenceInput:
      "Review a $1,899 annual SaaS renewal from Acme Cloud and decide whether it needs approval.",
    trainingInput:
      "Teach the smaller model to review software renewals against the spending threshold.",
  },
  {
    id: "budget-miss",
    exampleId: "ex_var_hard_001",
    task: "variance_analysis",
    label: "Budget miss",
    inferenceInput:
      "Explain the 2027 Q4 budget miss, rank the main causes, and say what finance should check next.",
    trainingInput:
      "Teach the smaller model to explain budget misses and rank the largest drivers.",
  },
  {
    id: "cash-mismatch",
    exampleId: "ex_cash_hard_exc_001",
    task: "cash_reconciliation",
    label: "Cash mismatch",
    inferenceInput:
      "Match the bank movements to the books and identify the bank fee and deposit in transit.",
    trainingInput:
      "Teach the smaller model to reconcile cash and identify unmatched bank or book entries.",
  },
] as const satisfies readonly DemoExamplePreset[];

export type DemoExamplePresetId = (typeof DEMO_EXAMPLE_CATALOG)[number]["id"];

export function getDemoExamplePreset(
  id: DemoExamplePresetId,
): (typeof DEMO_EXAMPLE_CATALOG)[number] {
  const preset = DEMO_EXAMPLE_CATALOG.find((item) => item.id === id);
  if (!preset) {
    throw new Error(`Unknown demo example preset: ${id}`);
  }
  return preset;
}

export function getPresetExample(preset: DemoExamplePreset): DemoExample {
  const example = getDemoExample(preset.exampleId);
  if (!example || example.task !== preset.task) {
    throw new Error(`Demo example preset ${preset.id} has an invalid example binding`);
  }
  return example;
}

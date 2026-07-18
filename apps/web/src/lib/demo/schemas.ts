import type { DemoExpectedOutputSchema, FinanceTaskId } from "@/lib/demo/types";

const SCHEMAS: Record<FinanceTaskId, DemoExpectedOutputSchema> = {
  transaction_review: {
    schema_version: "transaction_review.v1",
    task: "transaction_review",
    required_fields: [
      "task",
      "schema_version",
      "gl_account",
      "journal_entry",
      "policy_action",
      "rule_ids",
      "evidence",
      "confidence",
    ],
    description:
      "A balanced journal entry, account choice, policy action, supporting rules, and confidence.",
  },
  variance_analysis: {
    schema_version: "variance_analysis.v1",
    task: "variance_analysis",
    required_fields: [
      "task",
      "schema_version",
      "profit_impact_minor",
      "direction",
      "top_drivers",
      "other_impact_minor",
      "rule_ids",
      "evidence_ids",
      "confidence",
    ],
    description:
      "The profit impact, main causes, direction, supporting records, and confidence.",
  },
  cash_reconciliation: {
    schema_version: "cash_reconciliation.v1",
    task: "cash_reconciliation",
    required_fields: [
      "task",
      "schema_version",
      "status",
      "matched_groups",
      "exceptions",
      "adjusted_book_balance_minor",
      "adjusted_bank_balance_minor",
      "difference_minor",
      "confidence",
    ],
    description:
      "The match status, grouped entries, exceptions, adjusted balances, and book minus bank difference.",
  },
};

export function expectedOutputSchemaFor(task: FinanceTaskId): DemoExpectedOutputSchema {
  return SCHEMAS[task];
}

export function validateStructuredOutput(
  task: FinanceTaskId,
  output: Record<string, unknown>,
): { state: "valid" | "invalid"; detail: string | null } {
  const schema = SCHEMAS[task];
  const missing = schema.required_fields.filter((field) => !(field in output));
  if (missing.length > 0) {
    return {
      state: "invalid",
      detail: `The result is missing these required fields: ${missing.join(", ")}.`,
    };
  }
  if (output.task !== task) {
    return {
      state: "invalid",
      detail: `The result is for ${String(output.task)}, not the selected job ${task}.`,
    };
  }
  if (output.schema_version !== schema.schema_version) {
    return {
      state: "invalid",
      detail: `The result uses an unexpected schema version: ${String(output.schema_version)}.`,
    };
  }
  return { state: "valid", detail: null };
}

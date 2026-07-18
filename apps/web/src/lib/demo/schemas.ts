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
      "Balanced journal entry, GL account, policy action (approve|review|reject), rule IDs, evidence, confidence.",
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
      "Profit impact with arithmetic-closed top drivers, favorable/unfavorable direction, evidence IDs, confidence.",
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
      "Balanced or exceptions status, matched groups, exceptions, adjusted balances, difference_minor = book − bank.",
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
      detail: `Missing required fields: ${missing.join(", ")}`,
    };
  }
  if (output.task !== task) {
    return {
      state: "invalid",
      detail: `Output task ${String(output.task)} does not match selected task ${task}`,
    };
  }
  if (output.schema_version !== schema.schema_version) {
    return {
      state: "invalid",
      detail: `Unexpected schema_version ${String(output.schema_version)}`,
    };
  }
  return { state: "valid", detail: null };
}

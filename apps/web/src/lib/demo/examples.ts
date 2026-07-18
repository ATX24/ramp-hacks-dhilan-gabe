import { expectedOutputSchemaFor } from "@/lib/demo/schemas";
import type { DemoExample, FinanceTaskId } from "@/lib/demo/types";

/**
 * Safe, deterministic prepopulated Demo examples.
 * Held-out and demo-catalog only. Never use training gold as the default case.
 */
const EXAMPLES: readonly DemoExample[] = [
  {
    example_id: "ex_txn_hard_001",
    task: "transaction_review",
    difficulty: "hard",
    split: "iid_test",
    label: "Review a Dell server purchase",
    input: {
      amount_minor: 5000000,
      currency: "USD",
      descriptor: "DELL SERVER CLUSTER",
      gl_candidates: ["1500", "6400", "2100"],
      policy_excerpt:
        "Capex > $10k requires reject unless CAPEX-APPROVED; POL-CAPEX-009 precedes POL-IT-002",
      vendor: "Dell Technologies",
    },
    expected_output_schema: expectedOutputSchemaFor("transaction_review"),
    gold_output: {
      confidence: 0.88,
      evidence: [{ field: "amount_minor", source_id: "txn", value: "5000000" }],
      gl_account: "1500",
      journal_entry: [
        { account: "1500", amount_minor: 5000000, side: "debit" },
        { account: "2100", amount_minor: 5000000, side: "credit" },
      ],
      policy_action: "reject",
      rule_ids: ["POL-CAPEX-009"],
      schema_version: "transaction_review.v1",
      task: "transaction_review",
    },
  },
  {
    example_id: "ex_demo_txn_saas_001",
    task: "transaction_review",
    difficulty: "medium",
    split: "demo_catalog",
    label: "Review a SaaS renewal",
    input: {
      amount_minor: 189900,
      currency: "USD",
      descriptor: "ACME CLOUD ANNUAL",
      gl_candidates: ["6400", "6100", "2100"],
      policy_excerpt: "SaaS renewals over $1,000 require review POL-SAAS-014",
      vendor: "Acme Cloud",
    },
    expected_output_schema: expectedOutputSchemaFor("transaction_review"),
    gold_output: {
      confidence: 0.91,
      evidence: [{ field: "amount_minor", source_id: "txn", value: "189900" }],
      gl_account: "6400",
      journal_entry: [
        { account: "6400", amount_minor: 189900, side: "debit" },
        { account: "2100", amount_minor: 189900, side: "credit" },
      ],
      policy_action: "review",
      rule_ids: ["POL-SAAS-014"],
      schema_version: "transaction_review.v1",
      task: "transaction_review",
    },
  },
  {
    example_id: "ex_var_hard_001",
    task: "variance_analysis",
    difficulty: "hard",
    split: "ood_test",
    label: "Explain offsetting price, volume, and currency changes",
    input: {
      drivers: [
        {
          actual_margin_minor: 1200000,
          budget_margin_minor: 1000000,
          driver_id: "price",
        },
        {
          actual_margin_minor: 1650000,
          budget_margin_minor: 2000000,
          driver_id: "volume",
        },
        {
          actual_margin_minor: -50000,
          budget_margin_minor: 0,
          driver_id: "fx",
        },
      ],
      note: "offsetting favorable price and unfavorable volume plus FX",
      period: "2027-Q4",
      unattributed_actual_margin_minor: 480000,
      unattributed_budget_margin_minor: 500000,
    },
    expected_output_schema: expectedOutputSchemaFor("variance_analysis"),
    gold_output: {
      confidence: 0.86,
      direction: "unfavorable",
      evidence_ids: ["price_actual", "volume_actual", "fx_rate"],
      other_impact_minor: -20000,
      profit_impact_minor: -220000,
      rule_ids: ["VAR-OOD-MATERIAL-005", "VAR-OOD-FX-002"],
      schema_version: "variance_analysis.v1",
      task: "variance_analysis",
      top_drivers: [
        { driver_id: "volume", impact_minor: -350000, rank: 1 },
        { driver_id: "price", impact_minor: 200000, rank: 2 },
        { driver_id: "fx", impact_minor: -50000, rank: 3 },
      ],
    },
  },
  {
    example_id: "ex_var_tie_rank_001",
    task: "variance_analysis",
    difficulty: "hard",
    split: "ood_test",
    label: "Break a tie between equal drivers",
    input: {
      actual_minor: 1400000,
      budget_minor: 1200000,
      drivers: [
        {
          actual_minor: 600000,
          budget_minor: 500000,
          driver_id: "ood_alpha_cost",
        },
        {
          actual_minor: 800000,
          budget_minor: 700000,
          driver_id: "ood_beta_cost",
        },
      ],
      note: "two drivers with equal absolute impact; tie-break by driver_id ascending",
      period: "2027-Q4",
    },
    expected_output_schema: expectedOutputSchemaFor("variance_analysis"),
    gold_output: {
      confidence: 0.9,
      direction: "unfavorable",
      evidence_ids: ["ood_alpha_actual", "ood_beta_actual"],
      other_impact_minor: 0,
      profit_impact_minor: -200000,
      rule_ids: ["VAR-OOD-TIEBREAK-001"],
      schema_version: "variance_analysis.v1",
      task: "variance_analysis",
      top_drivers: [
        { driver_id: "ood_alpha_cost", impact_minor: -100000, rank: 1 },
        { driver_id: "ood_beta_cost", impact_minor: -100000, rank: 2 },
      ],
    },
  },
  {
    example_id: "ex_cash_hard_exc_001",
    task: "cash_reconciliation",
    difficulty: "hard",
    split: "iid_test",
    label: "Reconcile a bank fee and a deposit in transit",
    input: {
      bank_balance_minor: 8119900,
      bank_events: [
        { amount_minor: 250000, id: "bank_iid_match" },
        { amount_minor: -3500, id: "bank_iid_fee", type: "fee" },
      ],
      book_balance_minor: 8173400,
      book_entries: [
        { amount_minor: 250000, id: "book_iid_match" },
        { amount_minor: 50000, id: "book_iid_deposit" },
      ],
    },
    expected_output_schema: expectedOutputSchemaFor("cash_reconciliation"),
    gold_output: {
      adjusted_bank_balance_minor: 8169900,
      adjusted_book_balance_minor: 8169900,
      confidence: 0.89,
      difference_minor: 0,
      exceptions: [
        {
          amount_minor: 3500,
          event_ids: ["bank_iid_fee"],
          type: "bank_fee",
        },
        {
          amount_minor: 50000,
          event_ids: ["book_iid_deposit"],
          type: "deposit_in_transit",
        },
      ],
      matched_groups: [
        {
          bank_ids: ["bank_iid_match"],
          book_ids: ["book_iid_match"],
        },
      ],
      schema_version: "cash_reconciliation.v1",
      status: "exceptions",
      task: "cash_reconciliation",
    },
  },
  {
    example_id: "ex_demo_cash_balanced_001",
    task: "cash_reconciliation",
    difficulty: "easy",
    split: "demo_catalog",
    label: "Reconcile a clean one-to-one match",
    input: {
      bank_balance_minor: 250000,
      bank_events: [{ amount_minor: 250000, id: "bank_demo_match" }],
      book_balance_minor: 250000,
      book_entries: [{ amount_minor: 250000, id: "book_demo_match" }],
    },
    expected_output_schema: expectedOutputSchemaFor("cash_reconciliation"),
    gold_output: {
      adjusted_bank_balance_minor: 250000,
      adjusted_book_balance_minor: 250000,
      confidence: 0.96,
      difference_minor: 0,
      exceptions: [],
      matched_groups: [
        {
          bank_ids: ["bank_demo_match"],
          book_ids: ["book_demo_match"],
        },
      ],
      schema_version: "cash_reconciliation.v1",
      status: "balanced",
      task: "cash_reconciliation",
    },
  },
];

export function listDemoExamples(task?: FinanceTaskId): DemoExample[] {
  if (!task) return [...EXAMPLES];
  return EXAMPLES.filter((example) => example.task === task);
}

export function getDemoExample(exampleId: string): DemoExample | null {
  return EXAMPLES.find((example) => example.example_id === exampleId) ?? null;
}

export function defaultExampleForTask(task: FinanceTaskId): DemoExample {
  const match = listDemoExamples(task)[0];
  if (!match) {
    throw new Error(`No demo examples registered for task ${task}`);
  }
  return match;
}

/** Deterministic next example for the Random control (stable order, wraps). */
export function nextDemoExample(
  task: FinanceTaskId,
  currentExampleId: string,
): DemoExample {
  const pool = listDemoExamples(task);
  const index = pool.findIndex((example) => example.example_id === currentExampleId);
  if (index < 0) return pool[0]!;
  return pool[(index + 1) % pool.length]!;
}

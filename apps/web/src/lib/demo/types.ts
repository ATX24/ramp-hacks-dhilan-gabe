import type { ProofStatus } from "@/lib/types";

/** Finance tasks supported by the Demo/Playground. */
export type FinanceTaskId =
  | "transaction_review"
  | "variance_analysis"
  | "cash_reconciliation";

/** Trained / baseline arms the registry may advertise. */
export type DemoModelArmId =
  | "student_base"
  | "oracle_sft"
  | "sequence_kd"
  | "logit_kd"
  | "ce_ablation"
  | "promoted_winner";

/**
 * Live serving readiness for a registry model.
 * - `live`: a serving endpoint + artifact are advertised for real inference
 * - `fixture_preview`: deterministic labeled preview only (never claim live)
 * - `unavailable`: no artifact/endpoint; live calls must fail loudly
 */
export type DemoServingAvailability = "live" | "fixture_preview" | "unavailable";

export type DemoPromotionStatus = "promoted" | "not_promoted" | "unknown";

export type DemoRunMode = "single" | "compare";

export type DemoInferenceMode = "fixture_preview" | "live";

export interface DemoTeacherStudentRef {
  id: string;
  revision: string;
}

/**
 * Evidence-backed model stats. Missing evidence must stay `null`
 * and render as "unknown" in the UI. Never invent values.
 */
export interface DemoModelStats {
  advertised_parameter_count: number | null;
  adapter_parameter_count: number | null;
  compression_ratio: number | null;
  recipe: string | null;
  teacher: DemoTeacherStudentRef | null;
  student: DemoTeacherStudentRef | null;
  seed: number | null;
  data_hash: string | null;
  manifest_hash: string | null;
  artifact_hash: string | null;
  training_duration_seconds: number | null;
  training_cost_usd: number | null;
  iid_primary_index: number | null;
  iid_ci_low: number | null;
  iid_ci_high: number | null;
  ood_retention: number | null;
  ood_ci_low: number | null;
  ood_ci_high: number | null;
  proof_status: ProofStatus | null;
  promotion_status: DemoPromotionStatus;
}

export interface DemoModelServing {
  availability: DemoServingAvailability;
  endpoint_id: string | null;
  artifact_id: string | null;
  reason: string | null;
}

export interface DemoModelEntry {
  model_id: string;
  arm_id: DemoModelArmId;
  display_name: string;
  purpose: string;
  excluded: boolean;
  exclusion_reason: string | null;
  serving: DemoModelServing;
  stats: DemoModelStats;
}

export type DemoPortfolioRole = "generalist" | "specialist";

export interface DemoPortfolioEntry {
  portfolio_id: string;
  display_name: string;
  role: DemoPortfolioRole;
  recommended: boolean;
  task_scope: readonly FinanceTaskId[];
  base_model_id: string;
  adapter_id: string | null;
  artifact_id: string | null;
  availability: DemoServingAvailability;
  selection_policy: "auto_default" | "explicit_only";
  purpose: string;
}

export interface DemoModelRegistry {
  schema_version: "distillery.demo_model_registry.v1";
  run_id: string;
  dataset_id: string;
  models: DemoModelEntry[];
  portfolio: DemoPortfolioEntry[];
}

export interface DemoExpectedOutputSchema {
  schema_version: string;
  task: FinanceTaskId;
  required_fields: readonly string[];
  description: string;
}

export interface DemoExample {
  example_id: string;
  task: FinanceTaskId;
  difficulty: "easy" | "medium" | "hard";
  split: "iid_test" | "ood_test" | "test" | "demo_catalog";
  label: string;
  input: Record<string, unknown>;
  expected_output_schema: DemoExpectedOutputSchema;
  /** Gold for scoring when available; never shown as a model prediction. */
  gold_output: Record<string, unknown> | null;
}

export type DemoValidationState = "valid" | "invalid" | "unknown";

export type DemoGatewayErrorCode =
  | "SERVING_ENDPOINT_MISSING"
  | "ARTIFACT_NOT_SERVABLE"
  | "MODEL_NOT_IN_REGISTRY"
  | "LIVE_TRANSPORT_ERROR"
  | "LIVE_RESPONSE_INVALID"
  | "FIXTURE_PREVIEW_UNSUPPORTED";

export interface DemoInferenceRequest {
  model_id: string;
  task: FinanceTaskId;
  example_id: string | null;
  input: Record<string, unknown>;
  mode: DemoInferenceMode;
}

export type DemoInferenceResponse =
  | {
      status: "ok";
      provenance: "fixture_preview" | "live";
      model_id: string;
      task: FinanceTaskId;
      example_id: string | null;
      structured_output: Record<string, unknown>;
      raw_json: string;
      validation: DemoValidationState;
      validation_detail: string | null;
      latency_ms: number | null;
      prompt_tokens: number | null;
      completion_tokens: number | null;
      score: number | null;
      score_detail: string | null;
      label: string;
    }
  | {
      status: "unavailable";
      provenance: "none";
      model_id: string;
      task: FinanceTaskId;
      example_id: string | null;
      code: DemoGatewayErrorCode;
      message: string;
    }
  | {
      status: "error";
      provenance: "none";
      model_id: string;
      task: FinanceTaskId;
      example_id: string | null;
      code: DemoGatewayErrorCode;
      message: string;
      retryable: boolean;
    };

/**
 * Minimal backend contract the web Demo gateway will call for live inference.
 * Implement under apps/api later; web must not invent live outputs.
 *
 * POST {base}/v1/demo/infer
 * GET  {base}/v1/demo/models?run_id=...
 * GET  {base}/v1/demo/health
 */
export interface DemoLiveInferHttpRequest {
  model_id: string;
  artifact_id: string;
  task: FinanceTaskId;
  example_id: string | null;
  input: Record<string, unknown>;
}

export interface DemoLiveInferHttpOk {
  structured_output: Record<string, unknown>;
  latency_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
}

export interface DemoLiveHealthHttpOk {
  serving_ready: boolean;
  endpoint_id: string | null;
  available_model_ids: string[];
}

export interface DemoUrlState {
  task: FinanceTaskId;
  modelIds: string[];
  exampleId: string;
  runMode: DemoRunMode;
  inferenceMode: DemoInferenceMode;
}

export const FINANCE_TASKS: readonly {
  id: FinanceTaskId;
  label: string;
  short: string;
}[] = [
  {
    id: "transaction_review",
    label: "Transaction review",
    short: "Txn review",
  },
  {
    id: "variance_analysis",
    label: "Variance analysis",
    short: "Variance",
  },
  {
    id: "cash_reconciliation",
    label: "Cash reconciliation",
    short: "Cash recon",
  },
] as const;

export const DEMO_MODEL_ARM_ORDER: readonly DemoModelArmId[] = [
  "student_base",
  "oracle_sft",
  "sequence_kd",
  "logit_kd",
  "ce_ablation",
  "promoted_winner",
] as const;


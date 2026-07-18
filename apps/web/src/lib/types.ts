/** Contract-aligned TypeScript types for the Distillery five-stage UI. */

export type RunState =
  | "QUEUED"
  | "STARTING"
  | "SYNTHESIZING"
  | "TRAINING"
  | "EVALUATING"
  | "FINALIZING"
  | "SUCCEEDED"
  | "FAILED"
  | "CANCELLED";

export type ProofStatus =
  | "proved"
  | "do_not_distill"
  | "failed_quality"
  | "failed_economics"
  | "insufficient_evidence";

export type UiMode =
  | "default"
  | "precomputed"
  | "proved"
  | "do_not_distill"
  | "failed_quality"
  | "failed_economics"
  | "error"
  | "unavailable"
  | "insufficient_evidence"
  | "skipped_synthesis"
  | "no_training_yet"
  | "loading"
  | "fetch_failure";

export type StageId = "curate" | "synthesize" | "train" | "prove" | "demo";

export type StageLoadState =
  | { status: "ready" }
  | {
      status: "loading";
      message: string;
    }
  | {
      status: "failed";
      title: string;
      message: string;
      retryable: boolean;
    };

export type FixtureResourceKind = "dataset" | "run" | "artifact" | "report";

export type FixtureClientErrorCode =
  | "INVALID_RESOURCE_ID"
  | "RESOURCE_NOT_FOUND"
  | "RESOURCE_MISMATCH"
  | "FIXTURE_INTEGRITY_ERROR";

export interface FixtureClientErrorPayload {
  code: FixtureClientErrorCode;
  message: string;
  resource_kind: FixtureResourceKind | "fixture";
  resource_id: string | null;
  details: Record<string, unknown>;
  retryable: false;
}

export type DistilleryErrorCode =
  | "INVALID_DATASET"
  | "SCHEMA_MISMATCH"
  | "DATA_LEAKAGE_DETECTED"
  | "UNSUPPORTED_LABEL_SOURCE"
  | "MODEL_REVISION_UNPINNED"
  | "TOKENIZER_MISMATCH"
  | "CHAT_TEMPLATE_MISMATCH"
  | "LICENSE_GATE_UNRESOLVED"
  | "OUTPUT_USE_NOT_ALLOWED"
  | "RECIPE_NOT_IMPLEMENTED"
  | "RECIPE_INCOMPATIBLE"
  | "CAPABILITY_UNAVAILABLE"
  | "MEMORY_DRY_RUN_FAILED"
  | "ESTIMATED_BUDGET_EXCEEDED"
  | "AWS_QUOTA_UNAVAILABLE"
  | "AWS_SUBMISSION_FAILED"
  | "AWS_JOB_FAILED"
  | "RUN_TIMEOUT"
  | "CANCELLED"
  | "ARTIFACT_INTEGRITY_FAILED"
  | "EVALUATION_INCOMPLETE"
  | "INSUFFICIENT_EVIDENCE"
  | "INVALID_TRANSITION"
  | "AUTO_RESOLVER_FAILED";

export interface ErrorPayload {
  code: DistilleryErrorCode;
  message: string;
  details: Record<string, unknown>;
  retryable: boolean;
  run_id: string | null;
}

export interface SplitHashes {
  train: string;
  validation: string;
  test: string | null;
  iid_test: string | null;
  ood_test: string | null;
}

export interface Dataset {
  schema_version: string;
  dataset_id: string;
  content_sha256: string;
  split_sha256: SplitHashes;
  uri: string;
  provenance_summary: string;
  task_mixture: {
    transaction_review: number;
    variance_analysis: number;
    cash_reconciliation: number;
  };
  difficulty_mixture: {
    easy: number;
    medium: number;
    hard: number;
  };
  example_count: number;
  schema_errors: SchemaIssue[];
  leakage_checks: LeakageCheck[];
  world_hashes: Record<string, string>;
  label_sources: Record<string, number>;
  frozen: boolean;
  created_at: string;
}

export interface SchemaIssue {
  example_id: string;
  path: string;
  message: string;
  severity: "error" | "warning";
}

export interface LeakageCheck {
  check_id: string;
  passed: boolean;
  detail: string;
}

export interface ProvenanceExample {
  example_id: string;
  task: string;
  label_source: "imported" | "oracle" | "teacher_generated" | "relabeled" | "rejected";
  teacher_id: string | null;
  teacher_revision: string | null;
  note: string;
}

export interface SynthesisSummary {
  run_id: string;
  skipped: boolean;
  skip_reason: string | null;
  counts: {
    imported: number;
    rejected: number;
    relabeled: number;
    generated: number;
  };
  teacher: {
    id: string;
    revision: string;
    estimated_cost_usd: number;
    calls_planned: number;
  } | null;
  provenance_examples: ProvenanceExample[];
}

export type GateStatus = "pass" | "fail" | "pending" | "unavailable";

export interface PreflightGate {
  gate_id: string;
  label: string;
  status: GateStatus;
  detail: string;
}

export interface DistillationPlan {
  run_id: string;
  requested_recipe: string;
  resolved_recipe: string | null;
  resolver_reasons: string[];
  rejected_alternatives: string[];
  teacher: { id: string; revision: string };
  student: { id: string; revision: string };
  tokenizer_compatible: boolean;
  chat_template_compatible: boolean;
  gates: PreflightGate[];
  planned_job: {
    backend: "local" | "sagemaker";
    instance_type: string;
    max_runtime_seconds: number;
    finite: true;
    launched: boolean;
  };
  cost: {
    max_run_usd: number;
    estimate_low_usd: number;
    estimate_high_usd: number;
    currency: "USD";
  };
  memory_peak_gib_estimate: number;
  wall_time_minutes_estimate: { low: number; high: number };
  cancellation_supported: boolean;
  training_launched: boolean;
  note: string;
}

export interface ModelArtifactMeta {
  artifact_id: string;
  run_id: string;
  student_base_id: string;
  student_revision: string;
  adapter_uri: string;
  merged_uri: string | null;
  checksums: Record<string, string>;
  load_instructions: string;
  precomputed: boolean;
}

export interface DistillationRunView {
  run_id: string;
  dataset_id: string;
  state: RunState;
  requested_recipe: string;
  resolved_recipe: string | null;
  resolver_reasons: string[];
  skip_synthesis_reason: string | null;
  model_artifact_id: string | null;
  proof_report_id: string | null;
  failure: ErrorPayload | null;
  training_launched: boolean;
  cancel_requested: boolean;
  updated_at: string;
}

export interface TrainingEvent {
  timestamp: string;
  state: RunState;
  message: string;
}

export interface TrainingMetric {
  timestamp: string;
  step: number;
  name: string;
  value: number;
  unit: string | null;
}

export type TrainingTelemetry =
  | {
      provenance: "not_started";
      events: [];
      metrics: [];
      message: string;
    }
  | {
      provenance: "precomputed_prior_run";
      events: TrainingEvent[];
      metrics: TrainingMetric[];
      immutable: true;
      message: string;
    }
  | {
      provenance: "error";
      events: TrainingEvent[];
      metrics: [];
      message: string;
    };

export interface ArmComparison {
  arm_id: string;
  purpose: string;
  primary_index: number | null;
  ci_low: number | null;
  ci_high: number | null;
  ood_retention: number | null;
  excluded: boolean;
  exclusion_reason: string | null;
}

export interface SystemsMetrics {
  p50_latency_ms: number;
  p95_latency_ms: number;
  throughput_rps_batch1: number;
  throughput_rps_batch8: number;
  peak_vram_gib: number;
  gpu_hours: number;
  hardware: string;
  measurement_source: "fixture" | "precomputed_prior_run";
}

export interface EconomicsSummary {
  gross_experiment_cost_usd: number;
  quality_retention: number | null;
  recovered_teacher_gap: number | null;
  break_even_requests: number | "never" | null;
  serving_cost_projected: boolean;
  utilization_sensitivity: Array<{
    utilization: number;
    cost_per_request_usd: number;
  }>;
  note: string;
}

export interface ProofReportView {
  report_id: string;
  run_ids: string[];
  protocol_sha256: string;
  proof_status: ProofStatus;
  first_failed_gate: string | null;
  unevaluated_gates: string[];
  arms: ArmComparison[];
  systems: SystemsMetrics | null;
  economics: EconomicsSummary;
  limitations: string[];
  artifact_downloads: Array<{
    name: string;
    uri: string;
    sha256: string;
  }>;
  precomputed: boolean;
  created_at: string;
}

export interface StageBundle {
  mode: UiMode;
  load_state: StageLoadState;
  run: DistillationRunView;
  dataset: Dataset;
  synthesis: SynthesisSummary;
  plan: DistillationPlan;
  training_telemetry: TrainingTelemetry;
  artifact: ModelArtifactMeta | null;
  proof: ProofReportView | null;
  error: ErrorPayload | null;
}

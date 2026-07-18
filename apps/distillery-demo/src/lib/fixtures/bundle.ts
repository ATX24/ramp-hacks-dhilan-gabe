import { assertFixtureBundleIntegrity } from "@/lib/fixtureIntegrity";
import {
  DEFAULT_DATASET_ID,
  getDefaultRunId,
  getFixtureEntry,
} from "@/lib/fixtures/catalog";
import { HASH } from "@/lib/fixtures/hashes";
export { UI_MODES, parseUiMode } from "@/lib/modes";
export {
  DEFAULT_DATASET_ID,
  DEFAULT_RUN_ID,
  RESTORED_RUN_ID,
} from "@/lib/fixtures/catalog";
import type {
  Dataset,
  DistillationPlan,
  DistillationRunView,
  ErrorPayload,
  ModelArtifactMeta,
  ProofReportView,
  StageBundle,
  SynthesisSummary,
  TrainingTelemetry,
  UiMode,
} from "@/lib/types";

const BASE_DATASET: Dataset = {
  schema_version: "distillery.dataset.v1",
  dataset_id: DEFAULT_DATASET_ID,
  content_sha256: HASH.dataset,
  split_sha256: {
    train: HASH.train,
    validation: HASH.validation,
    test: null,
    iid_test: HASH.iid,
    ood_test: HASH.ood,
  },
  uri: "fixture://datasets/ds_finance_world_v1_smoke",
  provenance_summary:
    "This made-up finance sample keeps teaching and test records separate. It contains no customer data.",
  task_mixture: {
    transaction_review: 0.45,
    variance_analysis: 0.45,
    cash_reconciliation: 0.1,
  },
  difficulty_mixture: {
    easy: 0.3,
    medium: 0.4,
    hard: 0.3,
  },
  example_count: 560,
  schema_errors: [
    {
      example_id: "ex_invalid_json_001",
      path: "expected_output",
      message: "Fixture negative: malformed JSON retained for validator coverage",
      severity: "warning",
    },
  ],
  leakage_checks: [
    {
      check_id: "world_id_isolation",
      passed: true,
      detail: "No world_id overlap across train/validation/IID/OOD.",
    },
    {
      check_id: "template_family_holdout",
      passed: true,
      detail: "OOD holds out renderer/template families.",
    },
    {
      check_id: "near_duplicate_minhash",
      passed: true,
      detail: "MinHash near-duplicate scan found no cross-split collisions.",
    },
  ],
  world_hashes: {
    generator_revision: HASH.world,
    latent_oracle: HASH.world,
    split_assignment: HASH.train,
  },
  label_sources: {
    oracle: 400,
    imported: 120,
    teacher_generated: 0,
    missing: 40,
  },
  frozen: true,
  created_at: "2026-07-18T12:00:00.000Z",
};

function basePlan(runId: string): DistillationPlan {
  return {
    run_id: runId,
    requested_recipe: "auto",
    resolved_recipe: "sequence.v1",
    resolver_reasons: ["usable_responses_present", "no_teacher_calls_required"],
    rejected_alternatives: ["logit.v1", "do_not_distill"],
    teacher: {
      id: "Qwen/Qwen2.5-1.5B-Instruct",
      revision: "a1b2c3d4e5f6g7h8i9j0",
    },
    student: {
      id: "Qwen/Qwen2.5-0.5B-Instruct",
      revision: "z9y8x7w6v5u4t3s2r1q0",
    },
    tokenizer_compatible: true,
    chat_template_compatible: true,
    gates: [
      {
        gate_id: "tokenizer_fingerprint",
        label: "Tokenizer fingerprint",
        status: "pass",
        detail: "Teacher/student fingerprints recorded; sequence.v1 does not require match.",
      },
      {
        gate_id: "memory_dry_run",
        label: "Memory dry-run",
        status: "pass",
        detail: "Bounded A10G peak-memory probe within ceiling.",
      },
      {
        gate_id: "license_output_use",
        label: "License / output-use",
        status: "pass",
        detail: "Pinned model licenses and output-use disposition approved for this run.",
      },
    ],
    planned_job: {
      backend: "sagemaker",
      instance_type: "ml.g5.xlarge",
      max_runtime_seconds: 2700,
      finite: true,
      launched: false,
    },
    cost: {
      max_run_usd: 25,
      estimate_low_usd: 4.2,
      estimate_high_usd: 11.8,
      currency: "USD",
    },
    memory_peak_gib_estimate: 18.4,
    wall_time_minutes_estimate: { low: 18, high: 32 },
    cancellation_supported: true,
    training_launched: false,
    note:
      "Planning estimates below are static; plan_distillation() performs no billable work.",
  };
}

function baseSynthesis(runId: string, skipped: boolean): SynthesisSummary {
  if (skipped) {
    return {
      run_id: runId,
      skipped: true,
      skip_reason: "responses_already_present",
      counts: {
        imported: 520,
        rejected: 0,
        relabeled: 0,
        generated: 0,
      },
      teacher: null,
      provenance_examples: [
        {
          example_id: "ex_txn_0042",
          task: "transaction_review",
          label_source: "imported",
          teacher_id: null,
          teacher_revision: null,
          note: "Valid imported prompt-response pair; synthesis skipped.",
        },
        {
          example_id: "ex_var_0110",
          task: "variance_analysis",
          label_source: "oracle",
          teacher_id: null,
          teacher_revision: null,
          note: "Oracle-gold sequence retained for oracle_sft arm materialization.",
        },
      ],
    };
  }

  return {
    run_id: runId,
    skipped: false,
    skip_reason: null,
    counts: {
      imported: 420,
      rejected: 28,
      relabeled: 12,
      generated: 40,
    },
    teacher: {
      id: "Qwen/Qwen2.5-1.5B-Instruct",
      revision: "a1b2c3d4e5f6g7h8i9j0",
      estimated_cost_usd: 1.15,
      calls_planned: 40,
    },
    provenance_examples: [
      {
        example_id: "ex_txn_0007",
        task: "transaction_review",
        label_source: "imported",
        teacher_id: null,
        teacher_revision: null,
        note: "Trace already contained a schema-valid teacher response.",
      },
      {
        example_id: "ex_txn_0088",
        task: "transaction_review",
        label_source: "rejected",
        teacher_id: null,
        teacher_revision: null,
        note: "Unbalanced journal entry failed deterministic validator.",
      },
      {
        example_id: "ex_var_0033",
        task: "variance_analysis",
        label_source: "relabeled",
        teacher_id: "Qwen/Qwen2.5-1.5B-Instruct",
        teacher_revision: "a1b2c3d4e5f6g7h8i9j0",
        note: "Failed arithmetic-closure check; teacher relabel within cost ceiling.",
      },
      {
        example_id: "ex_txn_0155",
        task: "transaction_review",
        label_source: "teacher_generated",
        teacher_id: "Qwen/Qwen2.5-1.5B-Instruct",
        teacher_revision: "a1b2c3d4e5f6g7h8i9j0",
        note: "Missing response filled by teacher generator.",
      },
    ],
  };
}

function baseRun(runId: string, datasetId: string): DistillationRunView {
  return {
    run_id: runId,
    dataset_id: datasetId,
    state: "QUEUED",
    requested_recipe: "auto",
    resolved_recipe: "sequence.v1",
    resolver_reasons: ["usable_responses_present", "no_teacher_calls_required"],
    skip_synthesis_reason: null,
    model_artifact_id: null,
    proof_report_id: null,
    failure: null,
    training_launched: false,
    cancel_requested: false,
    updated_at: "2026-07-18T12:05:00.000Z",
  };
}

function baseArtifact(runId: string, artifactId: string): ModelArtifactMeta {
  return {
    artifact_id: artifactId,
    run_id: runId,
    student_base_id: "Qwen/Qwen2.5-0.5B-Instruct",
    student_revision: "z9y8x7w6v5u4t3s2r1q0",
    adapter_uri: `fixture://runs/${runId}/model/adapter`,
    merged_uri: `fixture://runs/${runId}/model/merged`,
    checksums: {
      "adapter_model.safetensors": HASH.adapter,
      "manifest.json": HASH.manifest,
    },
    load_instructions:
      "Load adapter with pinned transformers + PEFT; optional merged weights load without repository code.",
    precomputed: false,
  };
}

function baseProof(
  runId: string,
  status: ProofReportView["proof_status"],
  reportId: string,
): ProofReportView {
  const insufficient = status === "insufficient_evidence";
  const failedQuality = status === "failed_quality";
  const failedEconomics = status === "failed_economics";
  const doNotDistill = status === "do_not_distill";
  const includeLogitPair = status === "proved" || failedQuality || failedEconomics;
  const firstFailedGate = failedQuality
    ? "quality_gate"
    : failedEconomics
      ? "economics_gate"
      : insufficient
        ? "evidence_gate"
        : doNotDistill
          ? "baseline_gate"
          : null;
  const unevaluatedGates = failedQuality
    ? ["economics_gate", "evidence_gate"]
    : failedEconomics
      ? ["evidence_gate"]
      : insufficient
        ? ["economics_gate"]
        : [];
  const sequencePrimaryIndex = failedQuality ? 0.72 : insufficient ? 0.84 : 0.85;
  return {
    report_id: reportId,
    run_ids: [runId],
    protocol_sha256: HASH.protocol,
    proof_status: status,
    first_failed_gate: firstFailedGate,
    unevaluated_gates: unevaluatedGates,
    arms: [
      {
        arm_id: "rules",
        purpose: "Simple accounting and policy rules",
        primary_index: 0.71,
        ci_low: 0.68,
        ci_high: 0.74,
        ood_retention: 0.93,
        excluded: false,
        exclusion_reason: null,
      },
      {
        arm_id: "teacher",
        purpose: "Source model used as the upper reference",
        primary_index: 0.88,
        ci_low: 0.85,
        ci_high: 0.91,
        ood_retention: 0.9,
        excluded: false,
        exclusion_reason: null,
      },
      {
        arm_id: "student_base",
        purpose: "Smaller model before it learns the new job",
        primary_index: 0.62,
        ci_low: 0.58,
        ci_high: 0.66,
        ood_retention: 0.81,
        excluded: false,
        exclusion_reason: null,
      },
      {
        arm_id: "cheap_off_the_shelf",
        purpose: "Low-cost hosted model used for comparison",
        primary_index: 0.69,
        ci_low: 0.65,
        ci_high: 0.73,
        ood_retention: 0.84,
        excluded: false,
        exclusion_reason: null,
      },
      {
        arm_id: "oracle_sft",
        purpose: "Known-correct answers used as an upper comparison",
        primary_index: 0.86,
        ci_low: 0.83,
        ci_high: 0.89,
        ood_retention: 0.91,
        excluded: false,
        exclusion_reason: null,
      },
      {
        arm_id: "sequence_kd",
        purpose: "Smaller model trained from complete source answers",
        primary_index: sequencePrimaryIndex,
        ci_low: failedQuality ? 0.66 : insufficient ? 0.78 : 0.82,
        ci_high: failedQuality ? 0.78 : insufficient ? 0.9 : 0.88,
        ood_retention: failedQuality ? 0.78 : 0.9,
        excluded: false,
        exclusion_reason: null,
      },
      {
        arm_id: "logit_kd",
        purpose: "Smaller model trained from source model scores",
        primary_index: includeLogitPair ? (failedQuality ? 0.7 : 0.84) : null,
        ci_low: includeLogitPair ? (failedQuality ? 0.64 : 0.81) : null,
        ci_high: includeLogitPair ? (failedQuality ? 0.76 : 0.87) : null,
        ood_retention: includeLogitPair ? (failedQuality ? 0.76 : 0.9) : null,
        excluded: !includeLogitPair,
        exclusion_reason: includeLogitPair
          ? null
          : "CAPABILITY_UNAVAILABLE: memory dry-run failed for exact forward KL",
      },
      {
        arm_id: "ce_ablation",
        purpose: "Matched training run without the extra distillation loss",
        primary_index: includeLogitPair ? (failedQuality ? 0.68 : 0.8) : null,
        ci_low: includeLogitPair ? (failedQuality ? 0.62 : 0.77) : null,
        ci_high: includeLogitPair ? (failedQuality ? 0.74 : 0.83) : null,
        ood_retention: includeLogitPair ? (failedQuality ? 0.74 : 0.87) : null,
        excluded: !includeLogitPair,
        exclusion_reason: includeLogitPair
          ? null
          : "Matched comparator omitted because logit_kd arm is unavailable",
      },
    ],
    systems: {
      p50_latency_ms: 42,
      p95_latency_ms: 96,
      throughput_rps_batch1: 18.2,
      throughput_rps_batch8: 64.5,
      peak_vram_gib: 17.9,
      gpu_hours: 0.42,
      hardware: "ml.g5.xlarge (A10G)",
      measurement_source: "fixture",
    },
    economics: {
      gross_experiment_cost_usd: 18.4,
      quality_retention: failedQuality ? 0.81 : 0.965,
      recovered_teacher_gap: failedQuality ? 0.38 : 0.88,
      break_even_requests: insufficient ? null : failedEconomics ? "never" : 4200,
      serving_cost_projected: true,
      utilization_sensitivity: [
        { utilization: 0.05, cost_per_request_usd: 0.0048 },
        { utilization: 0.25, cost_per_request_usd: 0.0016 },
        { utilization: 0.5, cost_per_request_usd: 0.0011 },
        { utilization: 0.8, cost_per_request_usd: 0.0009 },
      ],
      note: "Running costs are projected from the saved speed test, not measured production savings.",
    },
    limitations: [
      "The test only uses made-up finance data. It does not show how customer data would perform.",
      "Running costs are projected at the listed machine use levels.",
      ...(insufficient
        ? ["A second saved check is missing, so this result needs more evidence."]
        : failedQuality
          ? ["The saved run missed its accuracy checks on unfamiliar examples."]
          : failedEconomics
            ? ["The saved run did not lower the estimated cost per request."]
            : doNotDistill
              ? ["The simpler comparison won, so the full model sweep stopped."]
              : ["A made-up benchmark cannot show how the model will behave in production."]),
    ],
    artifact_downloads: [
      {
        name: "proof_report.json",
        uri: `fixture://proof-reports/${reportId}/report.json`,
        sha256: HASH.protocol,
      },
      {
        name: "predictions.jsonl",
        uri: `fixture://proof-reports/${reportId}/predictions.jsonl`,
        sha256: HASH.prediction,
      },
    ],
    precomputed: false,
    created_at: "2026-07-18T14:00:00.000Z",
  };
}

function notStartedTelemetry(): TrainingTelemetry {
  return {
    provenance: "not_started",
    events: [],
    metrics: [],
    message: "Not started. This saved plan has no job events or measurements.",
  };
}

function priorRunTelemetry(): TrainingTelemetry {
  return {
    provenance: "precomputed_prior_run",
    immutable: true,
    message:
      "Immutable events and metrics recorded by a completed prior run. No live stream is active.",
    events: [
      {
        timestamp: "2026-07-18T12:11:00.000Z",
        state: "STARTING",
        message: "Prior finite job validated its sealed manifest.",
      },
      {
        timestamp: "2026-07-18T12:14:00.000Z",
        state: "TRAINING",
        message: "Prior run entered its bounded training phase.",
      },
      {
        timestamp: "2026-07-18T12:34:00.000Z",
        state: "FINALIZING",
        message: "Prior run wrote immutable artifacts and checksums.",
      },
      {
        timestamp: "2026-07-18T12:36:00.000Z",
        state: "SUCCEEDED",
        message: "Prior run completed; this record is read-only.",
      },
    ],
    metrics: [
      {
        timestamp: "2026-07-18T12:18:00.000Z",
        step: 10,
        name: "completion_ce",
        value: 1.84,
        unit: null,
      },
      {
        timestamp: "2026-07-18T12:25:00.000Z",
        step: 20,
        name: "completion_ce",
        value: 1.29,
        unit: null,
      },
      {
        timestamp: "2026-07-18T12:32:00.000Z",
        step: 30,
        name: "completion_ce",
        value: 1.06,
        unit: null,
      },
    ],
  };
}

function errorTelemetry(message: string): TrainingTelemetry {
  return {
    provenance: "error",
    events: [
      {
        timestamp: "2026-07-18T12:06:00.000Z",
        state: "FAILED",
        message,
      },
    ],
    metrics: [],
    message: "No training metrics were produced because preparation failed.",
  };
}

function errorPayload(code: ErrorPayload["code"], message: string, runId: string): ErrorPayload {
  return {
    code,
    message,
    details: { source: "fixture" },
    retryable: false,
    run_id: runId,
  };
}

export function buildStageBundle(
  mode: UiMode = "default",
  requestedRunId?: string,
): StageBundle {
  const entry = getFixtureEntry(mode);
  const runId = requestedRunId ?? getDefaultRunId(mode);
  const dataset = { ...BASE_DATASET };
  const plan = basePlan(runId);
  const run = baseRun(runId, dataset.dataset_id);
  let synthesis = baseSynthesis(runId, false);
  let artifact: ModelArtifactMeta | null = null;
  let proof: ProofReportView | null = null;
  let error: ErrorPayload | null = null;
  let trainingTelemetry = notStartedTelemetry();
  let loadState: StageBundle["load_state"] = { status: "ready" };

  switch (mode) {
    case "default":
    case "no_training_yet":
      // Train stage must show planned job without implying launch.
      run.state = "QUEUED";
      run.training_launched = false;
      plan.training_launched = false;
      plan.planned_job.launched = false;
      break;

    case "skipped_synthesis":
      synthesis = baseSynthesis(runId, true);
      run.skip_synthesis_reason = "responses_already_present";
      run.state = "QUEUED";
      break;

    case "precomputed":
    case "proved":
    case "do_not_distill":
    case "failed_quality":
    case "failed_economics":
    case "insufficient_evidence": {
      if (!entry.artifactId || !entry.reportId || !entry.proofStatus) {
        throw new Error(`Missing prior-run fixture resources for mode ${mode}`);
      }
      run.state = "SUCCEEDED";
      run.training_launched = false;
      run.model_artifact_id = entry.artifactId;
      run.proof_report_id = entry.reportId;
      artifact = {
        ...baseArtifact(runId, entry.artifactId),
        precomputed: true,
      };
      proof = {
        ...baseProof(runId, entry.proofStatus, entry.reportId),
        precomputed: true,
      };
      if (proof.systems) {
        proof.systems = {
          ...proof.systems,
          measurement_source: "precomputed_prior_run",
        };
      }
      proof.limitations = [
        ...proof.limitations,
        "Precomputed checksum-verified prior-run artifacts are displayed; no live job is active.",
      ];
      plan.note =
        entry.proofStatus === "insufficient_evidence"
          ? "The configuration and cost ceiling below were recorded with that artifact; the evidence gate remains incomplete."
          : "The configuration and cost ceiling below were recorded with that artifact.";
      trainingTelemetry = priorRunTelemetry();
      break;
    }

    case "error":
      error = errorPayload(
        "DATA_LEAKAGE_DETECTED",
        "Leakage check failed: injected near-duplicate crossed train and OOD.",
        runId,
      );
      dataset.leakage_checks = [
        {
          check_id: "near_duplicate_minhash",
          passed: false,
          detail: "Near-duplicate hash collision between train and OOD splits.",
        },
        ...dataset.leakage_checks.filter((c) => c.check_id !== "near_duplicate_minhash"),
      ];
      dataset.frozen = false;
      run.state = "FAILED";
      run.failure = error;
      trainingTelemetry = errorTelemetry(error.message);
      break;

    case "unavailable":
      error = errorPayload(
        "RECIPE_NOT_IMPLEMENTED",
        "Requested recipe on_policy_gkd is catalog-only and not implemented.",
        runId,
      );
      plan.requested_recipe = "on_policy_gkd";
      plan.resolved_recipe = null;
      plan.resolver_reasons = ["catalog_only"];
      plan.gates = [
        {
          gate_id: "recipe_capability",
          label: "Recipe capability",
          status: "unavailable",
          detail: "RECIPE_NOT_IMPLEMENTED — no silent downgrade.",
        },
        {
          gate_id: "tokenizer_fingerprint",
          label: "Tokenizer fingerprint",
          status: "pending",
          detail: "Not evaluated after recipe capability failure.",
        },
        {
          gate_id: "memory_dry_run",
          label: "Memory dry-run",
          status: "pending",
          detail: "Not evaluated after recipe capability failure.",
        },
        {
          gate_id: "license_output_use",
          label: "License / output-use",
          status: "pending",
          detail: "Not evaluated after recipe capability failure.",
        },
      ];
      run.requested_recipe = "on_policy_gkd";
      run.resolved_recipe = null;
      run.failure = error;
      run.state = "FAILED";
      trainingTelemetry = errorTelemetry(error.message);
      break;

    case "loading":
      loadState = {
        status: "loading",
        message: "Loading static fixture data. No network request is being made.",
      };
      break;

    case "fetch_failure":
      loadState = {
        status: "failed",
        title: "Fixture data unavailable",
        message:
          "The simulated data request failed before stage content was available. No live API was called.",
        retryable: true,
      };
      break;

    default: {
      const _exhaustive: never = mode;
      throw new Error(`Unhandled UI mode: ${_exhaustive}`);
    }
  }

  const bundle: StageBundle = {
    mode,
    load_state: loadState,
    run,
    dataset,
    synthesis,
    plan,
    training_telemetry: trainingTelemetry,
    artifact,
    proof,
    error,
  };
  assertFixtureBundleIntegrity(bundle);
  return bundle;
}

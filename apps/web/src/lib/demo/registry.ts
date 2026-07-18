import {
  DEMO_MODEL_ARM_ORDER,
  type DemoModelArmId,
  type DemoModelEntry,
  type DemoModelRegistry,
  type DemoModelStats,
  type DemoPromotionStatus,
  type DemoServingAvailability,
} from "@/lib/demo/types";
import { HASH } from "@/lib/fixtures/hashes";
import { buildModelPortfolio } from "@/lib/modelPortfolio";
import type { ArmComparison, StageBundle } from "@/lib/types";

const ARM_DISPLAY: Record<
  DemoModelArmId,
  { display_name: string; purpose: string }
> = {
  student_base: {
    display_name: "Base smaller model",
    purpose: "The smaller model before it learns this job",
  },
  oracle_sft: {
    display_name: "Known-answer training",
    purpose: "Known-correct answers used as an upper comparison",
  },
  sequence_kd: {
    display_name: "Answer distillation",
    purpose: "Training from complete source-model answers",
  },
  logit_kd: {
    display_name: "Score distillation",
    purpose: "Training from source-model token scores",
  },
  ce_ablation: {
    display_name: "Plain training comparison",
    purpose: "The same run without the extra distillation loss",
  },
  promoted_winner: {
    display_name: "Chosen model",
    purpose: "The model selected by the saved checks",
  },
};

const STUDENT_PARAM_COUNT = 494_000_000;
const TEACHER_PARAM_COUNT = 1_540_000_000;
const ADAPTER_PARAM_COUNT = 8_400_000;

function emptyStats(promotion: DemoPromotionStatus = "unknown"): DemoModelStats {
  return {
    advertised_parameter_count: null,
    adapter_parameter_count: null,
    compression_ratio: null,
    recipe: null,
    teacher: null,
    student: null,
    seed: null,
    data_hash: null,
    manifest_hash: null,
    artifact_hash: null,
    training_duration_seconds: null,
    training_cost_usd: null,
    iid_primary_index: null,
    iid_ci_low: null,
    iid_ci_high: null,
    ood_retention: null,
    ood_ci_low: null,
    ood_ci_high: null,
    proof_status: null,
    promotion_status: promotion,
  };
}

function armMetrics(arm: ArmComparison | undefined): Pick<
  DemoModelStats,
  | "iid_primary_index"
  | "iid_ci_low"
  | "iid_ci_high"
  | "ood_retention"
  | "ood_ci_low"
  | "ood_ci_high"
> {
  if (!arm || arm.excluded) {
    return {
      iid_primary_index: null,
      iid_ci_low: null,
      iid_ci_high: null,
      ood_retention: null,
      ood_ci_low: null,
      ood_ci_high: null,
    };
  }
  return {
    iid_primary_index: arm.primary_index,
    iid_ci_low: arm.ci_low,
    iid_ci_high: arm.ci_high,
    ood_retention: arm.ood_retention,
    // Fixture arms do not advertise a separate OOD CI; keep unknown.
    ood_ci_low: null,
    ood_ci_high: null,
  };
}

function recipeForArm(armId: DemoModelArmId, resolvedRecipe: string | null): string | null {
  switch (armId) {
    case "student_base":
      return null;
    case "oracle_sft":
      return "oracle_sft.v1";
    case "sequence_kd":
      return resolvedRecipe === "sequence.v1" ? "sequence.v1" : resolvedRecipe;
    case "logit_kd":
      return "logit.v1";
    case "ce_ablation":
      return "ce_ablation.v1";
    case "promoted_winner":
      return resolvedRecipe;
    default: {
      const _exhaustive: never = armId;
      return _exhaustive;
    }
  }
}

function servingForArm(
  armId: DemoModelArmId,
  bundle: StageBundle,
  excluded: boolean,
): DemoModelEntry["serving"] {
  const artifactId = bundle.artifact?.artifact_id ?? null;
  const hasPriorArtifact = artifactId !== null;

  if (armId === "student_base") {
    return {
      availability: "fixture_preview" satisfies DemoServingAvailability,
      endpoint_id: null,
      artifact_id: null,
      reason: "The base model only has a saved sample output. It has no live endpoint.",
    };
  }

  if (excluded) {
    return {
      availability: "unavailable",
      endpoint_id: null,
      artifact_id: null,
      reason: "This candidate was not part of the saved result and has no live model file.",
    };
  }

  if (!hasPriorArtifact) {
    return {
      availability: "unavailable",
      endpoint_id: null,
      artifact_id: null,
      reason: "This run does not have a trained model file.",
    };
  }

  // Live serving is never implied by fixtures. Preview is explicitly labeled.
  return {
    availability: "fixture_preview",
    endpoint_id: null,
    artifact_id: artifactId,
    reason:
      "A saved model file is available for sample output. No live endpoint is connected.",
  };
}

function statsForArm(
  armId: DemoModelArmId,
  bundle: StageBundle,
  arm: ArmComparison | undefined,
): DemoModelStats {
  const proof = bundle.proof;
  const plan = bundle.plan;
  const hasEvidence = proof !== null || bundle.artifact !== null;
  if (!hasEvidence && armId !== "student_base") {
    return emptyStats("unknown");
  }

  const metrics = armMetrics(arm);
  const promoted =
    armId === "promoted_winner" && proof?.proof_status === "proved"
      ? ("promoted" as const)
      : armId === "promoted_winner"
        ? ("not_promoted" as const)
        : proof?.proof_status === "proved" && armId === "sequence_kd"
          ? ("not_promoted" as const)
          : ("unknown" as const);

  const trained = armId !== "student_base" && bundle.artifact !== null && !arm?.excluded;

  return {
    advertised_parameter_count: STUDENT_PARAM_COUNT,
    adapter_parameter_count: trained ? ADAPTER_PARAM_COUNT : armId === "student_base" ? 0 : null,
    compression_ratio: TEACHER_PARAM_COUNT / STUDENT_PARAM_COUNT,
    recipe: recipeForArm(armId, plan.resolved_recipe),
    teacher: plan.teacher,
    student: plan.student,
    seed: proof ? 17 : null,
    data_hash: bundle.dataset.content_sha256,
    manifest_hash: trained ? HASH.manifest : null,
    artifact_hash: trained
      ? (bundle.artifact?.checksums["adapter_model.safetensors"] ?? null)
      : null,
    training_duration_seconds: trained
      ? Math.round((proof?.systems?.gpu_hours ?? 0) * 3600) || null
      : null,
    training_cost_usd: trained
      ? (proof?.economics.gross_experiment_cost_usd ?? null)
      : null,
    ...metrics,
    proof_status: proof?.proof_status ?? null,
    promotion_status: promoted,
  };
}

function shouldIncludeArm(
  armId: DemoModelArmId,
  armById: Map<string, ArmComparison>,
  bundle: StageBundle,
): boolean {
  if (armId === "student_base") return true;
  if (armId === "promoted_winner") {
    return bundle.proof?.proof_status === "proved";
  }
  // Registry-driven: only include trained arms present in the proof payload.
  return armById.has(armId);
}

/**
 * Build the Demo model registry from the stage bundle.
 * UI selectors must iterate this payload rather than hardcode candidate branches.
 */
export function buildDemoModelRegistry(bundle: StageBundle): DemoModelRegistry {
  const armById = new Map((bundle.proof?.arms ?? []).map((arm) => [arm.arm_id, arm]));
  const models: DemoModelEntry[] = [];

  for (const armId of DEMO_MODEL_ARM_ORDER) {
    if (!shouldIncludeArm(armId, armById, bundle)) continue;
    const arm = armById.get(armId);
    const excluded = arm?.excluded ?? false;
    const meta = ARM_DISPLAY[armId];
    models.push({
      model_id: `model_${armId}`,
      arm_id: armId,
      display_name: meta.display_name,
      purpose: arm?.purpose ?? meta.purpose,
      excluded,
      exclusion_reason: arm?.exclusion_reason ?? null,
      serving: servingForArm(armId, bundle, excluded),
      stats: statsForArm(armId, bundle, arm),
    });
  }

  return {
    schema_version: "distillery.demo_model_registry.v1",
    run_id: bundle.run.run_id,
    dataset_id: bundle.dataset.dataset_id,
    models,
    portfolio: buildModelPortfolio(bundle, models),
  };
}

export function findRegistryModel(
  registry: DemoModelRegistry,
  modelId: string,
): DemoModelEntry | null {
  return registry.models.find((model) => model.model_id === modelId) ?? null;
}

export function defaultWalkthroughModelIds(registry: DemoModelRegistry): string[] {
  const byArm = new Map(registry.models.map((model) => [model.arm_id, model.model_id]));
  const preferred: DemoModelArmId[] = ["student_base", "sequence_kd", "promoted_winner"];
  const selected: string[] = [];
  for (const armId of preferred) {
    const id = byArm.get(armId);
    if (id) selected.push(id);
  }
  if (selected.length === 0 && registry.models[0]) {
    selected.push(registry.models[0].model_id);
  }
  // Prefer compare when 2+ preferred arms exist; otherwise single.
  if (selected.length >= 2) return selected.slice(0, 2);
  return selected.slice(0, 1);
}

import type {
  DistillationPlan,
  DistillationRunView,
  ModelArtifactMeta,
  RunState,
} from "@/lib/types";

const ACTIVE_RUN_STATES: ReadonlySet<RunState> = new Set([
  "STARTING",
  "SYNTHESIZING",
  "TRAINING",
  "EVALUATING",
  "FINALIZING",
]);

export type TrainPresentation =
  | {
      kind: "preparation";
      title: "Preparation only";
      body: string;
      badge: "No active run";
      configurationHeading: "Planned finite job";
      costHeading: "Cost ceiling";
    }
  | {
      kind: "prior_completion";
      title: "Prior run artifact";
      body: string;
      badge: "Precomputed prior completion";
      configurationHeading: "Recorded finite-job configuration";
      costHeading: "Recorded cost ceiling";
    }
  | {
      kind: "failed";
      title: "Preparation failed";
      body: string;
      badge: "No active run";
      configurationHeading: "Attempted configuration";
      costHeading: "Configured cost ceiling";
    }
  | {
      kind: "active";
      title: "Active run";
      body: string;
      badge: "Active";
      configurationHeading: "Finite-job configuration";
      costHeading: "Cost ceiling";
    };

export function hasPrecomputedPriorCompletion(
  run: DistillationRunView,
  artifact: ModelArtifactMeta | null,
): boolean {
  return run.state === "SUCCEEDED" && artifact?.precomputed === true;
}

export function isRunCancellable(
  run: DistillationRunView,
  plan: DistillationPlan,
): boolean {
  return (
    plan.cancellation_supported &&
    plan.planned_job.launched &&
    run.training_launched &&
    ACTIVE_RUN_STATES.has(run.state) &&
    !run.cancel_requested
  );
}

export function deriveTrainPresentation(
  run: DistillationRunView,
  artifact: ModelArtifactMeta | null,
): TrainPresentation {
  if (hasPrecomputedPriorCompletion(run, artifact)) {
    return {
      kind: "prior_completion",
      title: "Prior run artifact",
      body:
        "This checksum-verified artifact records a completed prior run. Nothing is active in this UI; proof status is reported separately on Prove.",
      badge: "Precomputed prior completion",
      configurationHeading: "Recorded finite-job configuration",
      costHeading: "Recorded cost ceiling",
    };
  }

  if (run.failure || run.state === "FAILED" || run.state === "CANCELLED") {
    return {
      kind: "failed",
      title: "Preparation failed",
      body:
        "The fixture records a terminal preparation failure. No active run or stoppable job is represented.",
      badge: "No active run",
      configurationHeading: "Attempted configuration",
      costHeading: "Configured cost ceiling",
    };
  }

  if (run.training_launched && ACTIVE_RUN_STATES.has(run.state)) {
    return {
      kind: "active",
      title: "Active run",
      body: "The run record reports active work for a previously started finite job.",
      badge: "Active",
      configurationHeading: "Finite-job configuration",
      costHeading: "Cost ceiling",
    };
  }

  return {
    kind: "preparation",
    title: "Preparation only",
    body:
      "This fixture contains preflight and finite-job planning metadata. No active or previously completed run is represented.",
    badge: "No active run",
    configurationHeading: "Planned finite job",
    costHeading: "Cost ceiling",
  };
}

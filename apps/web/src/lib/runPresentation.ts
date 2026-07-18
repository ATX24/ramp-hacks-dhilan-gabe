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
      title: "The plan is ready";
      body: string;
      badge: "Not started";
      configurationHeading: "Planned job";
      costHeading: "Spending limit";
    }
  | {
      kind: "prior_completion";
      title: "This run finished earlier";
      body: string;
      badge: "Saved run";
      configurationHeading: "Saved job setup";
      costHeading: "Saved spending limit";
    }
  | {
      kind: "failed";
      title: "The plan could not start";
      body: string;
      badge: "Not running";
      configurationHeading: "Attempted setup";
      costHeading: "Spending limit";
    }
  | {
      kind: "active";
      title: "The job is running";
      body: string;
      badge: "Running";
      configurationHeading: "Job setup";
      costHeading: "Spending limit";
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
      title: "This run finished earlier",
      body:
        "The saved model files came from an earlier run. Nothing is running now. Open Check result to see whether it passed.",
      badge: "Saved run",
      configurationHeading: "Saved job setup",
      costHeading: "Saved spending limit",
    };
  }

  if (run.failure || run.state === "FAILED" || run.state === "CANCELLED") {
    return {
      kind: "failed",
      title: "The plan could not start",
      body:
        "The saved sample stops before training. There is no running job to stop.",
      badge: "Not running",
      configurationHeading: "Attempted setup",
      costHeading: "Spending limit",
    };
  }

  if (run.training_launched && ACTIVE_RUN_STATES.has(run.state)) {
    return {
      kind: "active",
      title: "The job is running",
      body: "This record says a bounded training job is still running.",
      badge: "Running",
      configurationHeading: "Job setup",
      costHeading: "Spending limit",
    };
  }

  return {
    kind: "preparation",
    title: "The plan is ready",
    body:
      "This saved sample contains a plan only. It has not started or finished a training job.",
    badge: "Not started",
    configurationHeading: "Planned job",
    costHeading: "Spending limit",
  };
}

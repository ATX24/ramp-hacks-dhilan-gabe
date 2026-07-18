import { deriveTrainPresentation } from "@/lib/runPresentation";
import type {
  DistillationPlan,
  DistillationRunView,
  ModelArtifactMeta,
  TrainingEvent,
  TrainingTelemetry,
} from "@/lib/types";

/** Explicit data origin — never blur fixture rehearsal with live jobs. */
export type TrainingDataOrigin = "fixture" | "precomputed_prior_run" | "live";

export type PlainTrainingStatus =
  | "not_started"
  | "preparing"
  | "teaching"
  | "checking"
  | "finished"
  | "failed"
  | "cancelled"
  | "unavailable";

export interface PlainTrainingEvent {
  timestamp: string | null;
  /** Machine state when available; null when only a derived summary exists. */
  state: string | null;
  /** Plain-language sentence suitable for a lay judge. */
  summary: string;
  origin: TrainingDataOrigin;
  /** Raw technical message kept for progressive disclosure. */
  technical: string | null;
}

export interface LiveTrainingGlance {
  origin: TrainingDataOrigin;
  originLabel: string;
  status: PlainTrainingStatus;
  statusLabel: string;
  progressPercent: number | null;
  progressLabel: string;
  etaLabel: string;
  spendLabel: string;
  experimentLabel: string;
  recentEvent: PlainTrainingEvent;
  events: PlainTrainingEvent[];
  isLive: boolean;
}

function originFromTelemetry(
  telemetry: TrainingTelemetry,
  plan: DistillationPlan,
): TrainingDataOrigin {
  if (telemetry.provenance === "precomputed_prior_run") {
    return "precomputed_prior_run";
  }
  // Fixture client never launches; only report live when the plan truly launched.
  if (plan.training_launched && plan.planned_job.launched) {
    return "live";
  }
  return "fixture";
}

function originLabel(origin: TrainingDataOrigin): string {
  switch (origin) {
    case "live":
      return "Live training feed";
    case "precomputed_prior_run":
      return "Prior-run record · not live";
    case "fixture":
      return "Fixture rehearsal · not live";
    default: {
      const _exhaustive: never = origin;
      return _exhaustive;
    }
  }
}

function adaptEvent(
  event: TrainingEvent,
  origin: TrainingDataOrigin,
): PlainTrainingEvent {
  return {
    timestamp: event.timestamp,
    state: event.state,
    summary: plainStateSentence(event.state, event.message),
    origin,
    technical: event.message,
  };
}

function plainStateSentence(state: string, technical: string): string {
  switch (state) {
    case "QUEUED":
      return "Waiting in line to start teaching.";
    case "STARTING":
      return "Starting the teaching job.";
    case "SYNTHESIZING":
      return "Filling missing labels before teaching.";
    case "TRAINING":
      return "Teaching the smaller model right now.";
    case "EVALUATING":
      return "Checking whether the smaller model is good enough.";
    case "FINALIZING":
      return "Packaging the result and sealing evidence.";
    case "SUCCEEDED":
      return "Teaching finished. Proof still decides if it was worth it.";
    case "FAILED":
      return "Teaching stopped with a recorded failure.";
    case "CANCELLED":
      return "Teaching was cancelled before finishing.";
    default:
      return technical || `Recorded state: ${state}`;
  }
}

function statusFromRun(
  run: DistillationRunView,
  presentationKind: ReturnType<typeof deriveTrainPresentation>["kind"],
): { status: PlainTrainingStatus; label: string } {
  if (presentationKind === "failed" || run.state === "FAILED") {
    return { status: "failed", label: "Failed" };
  }
  if (run.state === "CANCELLED") {
    return { status: "cancelled", label: "Cancelled" };
  }
  if (presentationKind === "prior_completion" || run.state === "SUCCEEDED") {
    return { status: "finished", label: "Finished · proof separate" };
  }
  if (presentationKind === "active") {
    if (run.state === "EVALUATING" || run.state === "FINALIZING") {
      return { status: "checking", label: "Checking results" };
    }
    if (run.state === "TRAINING" || run.state === "SYNTHESIZING") {
      return { status: "teaching", label: "Teaching in progress" };
    }
    return { status: "preparing", label: "Preparing" };
  }
  if (presentationKind === "preparation") {
    return { status: "not_started", label: "Not started" };
  }
  return { status: "unavailable", label: "Unavailable" };
}

function progressFor(
  status: PlainTrainingStatus,
  telemetry: TrainingTelemetry,
): { percent: number | null; label: string } {
  if (status === "not_started") {
    return { percent: 0, label: "0% · waiting to start" };
  }
  if (status === "finished") {
    return { percent: 100, label: "100% · teaching complete" };
  }
  if (status === "failed" || status === "cancelled") {
    return { percent: null, label: "Stopped before completion" };
  }
  if (telemetry.provenance === "precomputed_prior_run" && telemetry.metrics.length > 0) {
    const maxStep = Math.max(...telemetry.metrics.map((metric) => metric.step));
    const assumedTotal = Math.max(maxStep, 30);
    const percent = Math.min(100, Math.round((maxStep / assumedTotal) * 100));
    return {
      percent,
      label: `${percent}% · last recorded step ${maxStep}`,
    };
  }
  if (status === "teaching") {
    return { percent: 55, label: "In progress · exact % unknown" };
  }
  if (status === "checking") {
    return { percent: 85, label: "Nearly done · checking quality" };
  }
  if (status === "preparing") {
    return { percent: 15, label: "Preparing the sealed job" };
  }
  return { percent: null, label: "Progress unknown" };
}

function etaFor(
  status: PlainTrainingStatus,
  plan: DistillationPlan,
  origin: TrainingDataOrigin,
): string {
  if (status === "finished") return "No remaining time · already finished";
  if (status === "not_started") {
    return `Estimated ${plan.wall_time_minutes_estimate.low}–${plan.wall_time_minutes_estimate.high} min once started`;
  }
  if (status === "failed" || status === "cancelled") return "No ETA · job stopped";
  if (origin === "fixture") {
    return `Fixture estimate ${plan.wall_time_minutes_estimate.low}–${plan.wall_time_minutes_estimate.high} min · not a live clock`;
  }
  return `About ${plan.wall_time_minutes_estimate.low}–${plan.wall_time_minutes_estimate.high} min remaining (estimate)`;
}

function spendFor(plan: DistillationPlan, origin: TrainingDataOrigin): string {
  const ceiling = plan.cost.max_run_usd;
  const low = plan.cost.estimate_low_usd;
  const high = plan.cost.estimate_high_usd;
  const base = `Spend ceiling $${ceiling} · estimate $${low}–$${high}`;
  if (origin === "fixture") return `${base} · fixture planning only`;
  if (origin === "precomputed_prior_run") return `${base} · prior-run record`;
  return base;
}

/**
 * Honest adapter from fixture/live training records to a layperson glance card.
 * Never invents live progress when provenance is fixture or prior-run only.
 */
export function adaptLiveTrainingGlance(input: {
  run: DistillationRunView;
  plan: DistillationPlan;
  telemetry: TrainingTelemetry;
  artifact: ModelArtifactMeta | null;
}): LiveTrainingGlance {
  const { run, plan, telemetry, artifact } = input;
  const presentation = deriveTrainPresentation(run, artifact);
  const origin = originFromTelemetry(telemetry, plan);
  const { status, label: statusLabel } = statusFromRun(run, presentation.kind);
  const progress = progressFor(status, telemetry);

  const adaptedEvents =
    telemetry.events.length > 0
      ? telemetry.events.map((event) => adaptEvent(event, origin))
      : [
          {
            timestamp: run.updated_at,
            state: run.state,
            summary:
              telemetry.message ||
              (status === "not_started"
                ? "No teaching job has started in this view."
                : plainStateSentence(run.state, presentation.body)),
            origin,
            technical: telemetry.message || presentation.body,
          } satisfies PlainTrainingEvent,
        ];

  const recentEvent = adaptedEvents[adaptedEvents.length - 1]!;

  const recipe = plan.resolved_recipe ?? plan.requested_recipe;
  const experimentLabel = `${plan.student.id} ← ${plan.teacher.id} · ${recipe}`;

  return {
    origin,
    originLabel: originLabel(origin),
    status,
    statusLabel,
    progressPercent: progress.percent,
    progressLabel: progress.label,
    etaLabel: etaFor(status, plan, origin),
    spendLabel: spendFor(plan, origin),
    experimentLabel,
    recentEvent,
    events: adaptedEvents,
    isLive: origin === "live",
  };
}

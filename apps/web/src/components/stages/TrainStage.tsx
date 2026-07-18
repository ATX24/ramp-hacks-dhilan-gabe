"use client";

import Link from "next/link";
import { useState } from "react";
import { ErrorBanner } from "@/components/ErrorBanner";
import { GateList } from "@/components/GateList";
import { StatusBadge } from "@/components/StatusBadge";
import { TrainingTelemetryPanel } from "@/components/TrainingTelemetryPanel";
import { createApiClient } from "@/lib/api";
import { buildProjectHref } from "@/lib/navigation";
import {
  deriveTrainPresentation,
  isRunCancellable,
} from "@/lib/runPresentation";
import type {
  DistillationPlan,
  DistillationRunView,
  ErrorPayload,
  ModelArtifactMeta,
  TrainingTelemetry,
  UiMode,
} from "@/lib/types";

export function TrainStage({
  mode,
  plan,
  run,
  artifact,
  telemetry,
  error,
}: {
  mode: UiMode;
  plan: DistillationPlan;
  run: DistillationRunView;
  artifact: ModelArtifactMeta | null;
  telemetry: TrainingTelemetry;
  error: ErrorPayload | null;
}) {
  const [cancelRequested, setCancelRequested] = useState(run.cancel_requested);
  const [cancelNote, setCancelNote] = useState<string | null>(null);

  const presentation = deriveTrainPresentation(run, artifact);
  const cancellable = isRunCancellable(run, plan);
  const bannerClass =
    presentation.kind === "failed"
      ? "banner-error"
      : presentation.kind === "preparation"
        ? "banner-warn"
        : "banner-info";

  return (
    <section aria-labelledby="train-heading">
      <div className="panel">
        <p className="text-kicker text-[var(--orange)]">Train</p>
        <h1 id="train-heading">Make the smaller model</h1>
        <p>
          This page shows how the model would learn, which machine it would use, and
          the most the job may spend. Exact run settings stay under Advanced.
        </p>
        <ErrorBanner error={error} />
        <div
          className={`banner ${bannerClass}`}
          data-testid="run-presentation"
          data-presentation={presentation.kind}
          role="status"
        >
          <strong>{presentation.title}</strong>
          <p style={{ margin: 0 }}>
            {presentation.body} {plan.note}
          </p>
        </div>
        <div className="meta-row">
          <span>
            {presentation.kind === "prior_completion"
              ? "Saved demo record"
              : presentation.kind === "active"
                ? "Live job"
                : "Saved job plan"}
          </span>
          <StatusBadge
            tone={
              presentation.kind === "prior_completion"
                ? "precomputed"
                : presentation.kind === "active"
                  ? "pass"
                  : presentation.kind === "failed"
                    ? "fail"
                    : "pending"
            }
          >
            {presentation.badge}
          </StatusBadge>
        </div>
      </div>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
          Advanced job history
        </summary>
        <p>
          This record shows the exact saved events and measurements. It is not needed
          for the usual path.
        </p>
        <TrainingTelemetryPanel telemetry={telemetry} />
      </details>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
          Advanced training method
        </summary>
        <p>
          The training method (recipe) changes how the smaller model learns. Auto uses
          the supported method that fits the data and machine.
        </p>
        <div className="meta-row">
          <span>
            You asked for <code>{plan.requested_recipe}</code>
          </span>
          <span>
            Auto chose{" "}
            <code>{plan.resolved_recipe ?? "null"}</code>
          </span>
        </div>
        <p>Reasons:</p>
        <ul className="list-plain">
          {plan.resolver_reasons.map((reason) => (
            <li key={reason}>
              <code>{reason}</code>
            </li>
          ))}
        </ul>
        <p>Other methods it did not choose:</p>
        <ul className="list-plain">
          {plan.rejected_alternatives.map((alt) => (
            <li key={alt}>
              <code>{alt}</code>
            </li>
          ))}
        </ul>
        <p>
          Run ID: <code>{run.run_id}</code>
          <br />
          Internal state: <code>{run.state}</code>
        </p>
      </details>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
          Advanced model versions
        </summary>
        <p>
          The source model provides answers. The smaller model learns from them.
          Exact versions stay fixed so another run can use the same pair.
        </p>
        <div className="grid-2">
          <div className="stat">
            <span className="label">Source model (teacher)</span>
            <span className="value" style={{ fontSize: "1.05rem" }}>
              {plan.teacher.id}
            </span>
            <span className="mono">{plan.teacher.revision}</span>
          </div>
          <div className="stat">
            <span className="label">Smaller model (student)</span>
            <span className="value" style={{ fontSize: "1.05rem" }}>
              {plan.student.id}
            </span>
            <span className="mono">{plan.student.revision}</span>
          </div>
        </div>
      </details>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
          Advanced safety checks
        </summary>
        <p>
          These checks make sure the text format matches, the job fits in memory, and
          the model licenses allow this use. The run stops if a required check fails.
        </p>
        <GateList gates={plan.gates} />
      </details>

      <div className="grid-2">
        <details className="panel">
          <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
            Advanced machine setup
          </summary>
          <p>{presentation.configurationHeading}</p>
          <ul className="list-plain">
            <li>
              Job service (backend): <code>{plan.planned_job.backend}</code>
            </li>
            <li>
              Machine type (instance): <code>{plan.planned_job.instance_type}</code>
            </li>
            <li>
              Time limit: {plan.planned_job.max_runtime_seconds}s
            </li>
            <li>
              The job has a hard stop: {plan.planned_job.finite ? "yes" : "no"}
            </li>
            <li>
              Current work:{" "}
              <strong data-testid="job-activity">
                {presentation.kind === "active" ? "active" : "none"}
              </strong>
            </li>
            <li>
              Source:{" "}
              {presentation.kind === "prior_completion"
                ? "saved earlier run"
                : "saved sample plan"}
            </li>
            <li>
              Estimated peak memory: {plan.memory_peak_gib_estimate} GiB
            </li>
            <li>
              Time estimate: {plan.wall_time_minutes_estimate.low} to{" "}
              {plan.wall_time_minutes_estimate.high} min
            </li>
          </ul>
        </details>

        <div className="panel">
          <h3>{presentation.costHeading}</h3>
          <div className="grid-3">
            <div className="stat">
              <span className="label">Maximum</span>
              <span className="value">${plan.cost.max_run_usd}</span>
            </div>
            <div className="stat">
              <span className="label">Low estimate</span>
              <span className="value">${plan.cost.estimate_low_usd}</span>
            </div>
            <div className="stat">
              <span className="label">High estimate</span>
              <span className="value">${plan.cost.estimate_high_usd}</span>
            </div>
          </div>
          {cancellable ? (
            <div className="controls" style={{ marginTop: "1rem" }}>
              <button
                type="button"
                className="btn"
                disabled={cancelRequested}
                data-testid="cancel-button"
                onClick={async () => {
                  const client = createApiClient({ mode, runId: run.run_id });
                  const updated = await client.cancelRun(run.run_id);
                  setCancelRequested(updated.cancel_requested);
                  setCancelNote(
                    "The saved sample now shows a stop request. No live service was called.",
                  );
                }}
              >
                {cancelRequested ? "Stop requested" : "Stop this job"}
              </button>
            </div>
          ) : (
            <p data-testid="cancellation-unavailable">
              There is no running job to stop. Leaving this alone changes nothing.
            </p>
          )}
          {cancelNote ? (
            <p role="status" data-testid="cancel-note">
              {cancelNote}
            </p>
          ) : null}
        </div>
      </div>

      {artifact ? (
        <details className="panel">
          <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
            Advanced saved model files
          </summary>
          <p>
            These file names and fingerprints identify the exact model output. They are
            useful when you download, repeat, or audit a run.
          </p>
          <>
            <div className="meta-row">
              <span>
                File set <code>{artifact.artifact_id}</code>
              </span>
              {artifact.precomputed ? (
                <StatusBadge tone="precomputed">Saved run</StatusBadge>
              ) : null}
            </div>
            <ul className="list-plain">
              <li>
                Adapter file: <code>{artifact.adapter_uri}</code>
              </li>
              <li>
                Merged file: <code>{artifact.merged_uri ?? "Not available"}</code>
              </li>
              {Object.entries(artifact.checksums).map(([name, sha]) => (
                <li key={name}>
                  {name}: <span className="hash">{sha}</span>
                </li>
              ))}
            </ul>
            <p>{artifact.load_instructions}</p>
          </>
        </details>
      ) : (
        <div className="panel">
          <div className="grid gap-3" data-testid="no-artifacts-yet">
            <p>
              There are no model files yet. This saved sample only contains the checks
              that happen before a run.
            </p>
            <Link
              href={buildProjectHref(mode, run.run_id)}
              className="btn btn-primary w-fit"
            >
              Return to the project setup
            </Link>
          </div>
        </div>
      )}
    </section>
  );
}

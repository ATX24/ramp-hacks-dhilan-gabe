"use client";

import { useState } from "react";
import { ErrorBanner } from "@/components/ErrorBanner";
import { GateList } from "@/components/GateList";
import { LiveTrainingCard } from "@/components/LiveTrainingCard";
import { StatusBadge } from "@/components/StatusBadge";
import { TrainingTelemetryPanel } from "@/components/TrainingTelemetryPanel";
import { createApiClient } from "@/lib/api";
import { STAGE_PLAIN } from "@/lib/plainLanguage";
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
    <section aria-labelledby="train-heading" className="grid gap-4">
      <div className="panel">
        <p className="text-kicker">{STAGE_PLAIN.train.plain}</p>
        <h2 id="train-heading">Train</h2>
        <p>{STAGE_PLAIN.train.description}</p>
        <p className="text-sm text-muted-foreground">
          Why this matters: {STAGE_PLAIN.train.why} Technical preflight still
          covers recipe, tokenizer, memory, and license gates under Advanced
          details below.
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
            Run <code>{run.run_id}</code>
          </span>
          <span>
            {presentation.kind === "prior_completion" ? "Recorded state" : "Run state"}{" "}
            <code>{run.state}</code>
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

      <LiveTrainingCard
        run={run}
        plan={plan}
        telemetry={telemetry}
        artifact={artifact}
      />

      <TrainingTelemetryPanel telemetry={telemetry} />

      <details className="panel">
        <summary className="cursor-pointer font-serif text-lg">
          Advanced · recipe, gates, and job plan
        </summary>
      <div className="panel" style={{ marginTop: "1rem", boxShadow: "none" }}>
        <h3>Recipe resolution</h3>
        <div className="meta-row">
          <span>
            Requested <code>{plan.requested_recipe}</code>
          </span>
          <span>
            Resolved{" "}
            <code>{plan.resolved_recipe ?? "null"}</code>
          </span>
        </div>
        <p>Resolver reasons:</p>
        <ul className="list-plain">
          {plan.resolver_reasons.map((reason) => (
            <li key={reason}>
              <code>{reason}</code>
            </li>
          ))}
        </ul>
        <p>Rejected alternatives:</p>
        <ul className="list-plain">
          {plan.rejected_alternatives.map((alt) => (
            <li key={alt}>
              <code>{alt}</code>
            </li>
          ))}
        </ul>
      </div>

      <div className="panel">
        <h3>Pinned models</h3>
        <div className="grid-2">
          <div className="stat">
            <span className="label">Teacher</span>
            <span className="value" style={{ fontSize: "1.05rem" }}>
              {plan.teacher.id}
            </span>
            <span className="mono">{plan.teacher.revision}</span>
          </div>
          <div className="stat">
            <span className="label">Student (TinyFable base)</span>
            <span className="value" style={{ fontSize: "1.05rem" }}>
              {plan.student.id}
            </span>
            <span className="mono">{plan.student.revision}</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>Tokenizer · memory · license gates</h3>
        <GateList gates={plan.gates} />
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>{presentation.configurationHeading}</h3>
          <ul className="list-plain">
            <li>
              Backend: <code>{plan.planned_job.backend}</code>
            </li>
            <li>
              Instance: <code>{plan.planned_job.instance_type}</code>
            </li>
            <li>
              Max runtime: {plan.planned_job.max_runtime_seconds}s
            </li>
            <li>
              Finite job: {plan.planned_job.finite ? "yes" : "no"}
            </li>
            <li>
              Current activity:{" "}
              <strong data-testid="job-activity">
                {presentation.kind === "active" ? "active" : "none"}
              </strong>
            </li>
            <li>
              Record source:{" "}
              {presentation.kind === "prior_completion"
                ? "precomputed prior run"
                : "fixture preparation"}
            </li>
            <li>
              Peak memory estimate: {plan.memory_peak_gib_estimate} GiB
            </li>
            <li>
              Wall time estimate: {plan.wall_time_minutes_estimate.low}–
              {plan.wall_time_minutes_estimate.high} min
            </li>
          </ul>
        </div>

        <div className="panel">
          <h3>{presentation.costHeading}</h3>
          <div className="grid-3">
            <div className="stat">
              <span className="label">Ceiling</span>
              <span className="value">${plan.cost.max_run_usd}</span>
            </div>
            <div className="stat">
              <span className="label">Est. low</span>
              <span className="value">${plan.cost.estimate_low_usd}</span>
            </div>
            <div className="stat">
              <span className="label">Est. high</span>
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
                    "Cancellation was recorded by the fixture client. No live service call was made.",
                  );
                }}
              >
                {cancelRequested ? "Cancellation requested" : "Request cancellation"}
              </button>
            </div>
          ) : (
            <p data-testid="cancellation-unavailable">
              Cancellation unavailable: this view does not represent an active, started
              run.
            </p>
          )}
          {cancelNote ? (
            <p role="status" data-testid="cancel-note">
              {cancelNote}
            </p>
          ) : null}
        </div>
      </div>

      <div className="panel">
        <h3>Artifact integrity</h3>
        {artifact ? (
          <>
            <div className="meta-row">
              <span>
                Artifact <code>{artifact.artifact_id}</code>
              </span>
              {artifact.precomputed ? (
                <StatusBadge tone="precomputed">Precomputed</StatusBadge>
              ) : null}
            </div>
            <ul className="list-plain">
              <li>
                Adapter URI: <code>{artifact.adapter_uri}</code>
              </li>
              <li>
                Merged URI: <code>{artifact.merged_uri ?? "—"}</code>
              </li>
              {Object.entries(artifact.checksums).map(([name, sha]) => (
                <li key={name}>
                  {name}: <span className="hash">{sha}</span>
                </li>
              ))}
            </ul>
            <p>{artifact.load_instructions}</p>
          </>
        ) : (
          <p data-testid="no-artifacts-yet">
            No artifacts yet. This fixture contains preflight data only.
          </p>
        )}
      </div>
      </details>
    </section>
  );
}

"use client";

import Link from "next/link";
import { useState } from "react";
import { DemoExamplePicker } from "@/components/DemoExamplePicker";
import { ErrorBanner } from "@/components/ErrorBanner";
import { GateList } from "@/components/GateList";
import { LiveTrainingCard } from "@/components/LiveTrainingCard";
import { StatusBadge } from "@/components/StatusBadge";
import { TrainingTelemetryPanel } from "@/components/TrainingTelemetryPanel";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { createApiClient } from "@/lib/api";
import {
  DEMO_EXAMPLE_CATALOG,
  getDemoExamplePreset,
  type DemoExamplePresetId,
} from "@/lib/demo/exampleCatalog";
import { buildStageHref } from "@/lib/navigation";
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
  const [selectedPresetId, setSelectedPresetId] =
    useState<DemoExamplePresetId>("server-purchase");
  const [trainingInput, setTrainingInput] = useState<string>(
    DEMO_EXAMPLE_CATALOG[0].trainingInput,
  );
  const [previewedInput, setPreviewedInput] = useState<string | null>(null);

  const presentation = deriveTrainPresentation(run, artifact);
  const cancellable = isRunCancellable(run, plan);
  const bannerClass =
    presentation.kind === "failed"
      ? "banner-error"
      : presentation.kind === "preparation"
        ? "banner-warn"
        : "banner-info";

  function selectExample(id: DemoExamplePresetId) {
    const preset = getDemoExamplePreset(id);
    setSelectedPresetId(id);
    setTrainingInput(preset.trainingInput);
    setPreviewedInput(null);
  }

  return (
    <section aria-labelledby="train-heading" className="grid gap-4">
      <Card
        className="rounded-[20px] border-0 bg-card shadow-none ring-1 ring-black/10"
        data-testid="train-demo"
      >
        <CardContent className="grid gap-3 pt-4">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge tone="precomputed">Saved demo plan</StatusBadge>
            <span className="text-sm text-muted-foreground">
              No training job starts from this preview
            </span>
          </div>

          <div>
            <h1
              id="train-heading"
              className="font-serif text-3xl tracking-tight sm:text-4xl"
            >
              {previewedInput
                ? "Your teaching plan is ready"
                : "What do you want your smaller model to do?"}
            </h1>
            <p className="mt-1 text-sm text-muted-foreground sm:text-base">
              Describe one useful finance job. The saved example is ready to use.
            </p>
          </div>

          <div className="grid gap-2">
            <Label htmlFor="train-plain-input">Goal</Label>
            <Textarea
              id="train-plain-input"
              data-testid="train-plain-input"
              className="min-h-20 bg-background/60 text-sm sm:text-base"
              value={trainingInput}
              onChange={(event) => setTrainingInput(event.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Use a job description, not private data. Nothing is sent from this preview.
            </p>
          </div>

          <Button
            type="button"
            size="lg"
            className="w-full sm:w-fit sm:min-w-44"
            data-testid="train-demo-run"
            disabled={trainingInput.trim().length === 0}
            onClick={() => setPreviewedInput(trainingInput.trim())}
          >
            Preview my model plan
          </Button>

          <div
            className="grid gap-2 sm:grid-cols-2"
            data-testid="train-model-candidates"
          >
            <div className="rounded-xl border border-border bg-background/60 p-3">
              <p className="font-serif text-base">Current base model</p>
              <p className="text-xs text-muted-foreground">
                The smaller model before it learns this finance job.
              </p>
            </div>
            <div className="rounded-xl border border-border bg-background/60 p-3">
              <p className="font-serif text-base">Taught smaller model</p>
              <p className="text-xs text-muted-foreground">
                The same model after the saved teaching plan.
              </p>
            </div>
          </div>

          <details className="rounded-xl border border-black/15 px-3">
            <summary className="min-h-11 cursor-pointer py-3 text-sm font-medium">
              Try another finance example
            </summary>
            <div className="pb-3">
              <DemoExamplePicker
                selectedId={selectedPresetId}
                onSelect={selectExample}
              />
            </div>
          </details>

          {previewedInput ? (
            <div
              className="rounded-xl border border-border bg-background/60 p-3"
              data-testid="train-demo-result"
              role="status"
            >
              <h2 className="font-serif text-lg">Teaching preview</h2>
              <p className="mt-1 text-sm">{previewedInput}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                Saved demo data only. No training job was launched.
              </p>
              <Link
                href={buildStageHref("/demo", mode, run.run_id)}
                className="btn btn-primary mt-3 w-fit"
              >
                Compare both models
              </Link>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-lg">
          Saved run status
        </summary>
        <div className="mt-4 grid gap-4">
      <div className="panel">
        <p className="text-kicker">{STAGE_PLAIN.train.plain}</p>
        <h2>Run status</h2>
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
            {presentation.kind === "prior_completion" ? "Saved state" : "Current state"}{" "}
            <code>
              {presentation.kind === "preparation" && run.state === "QUEUED"
                ? "Planned"
                : run.state}
            </code>
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
        </div>
      </details>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-lg">
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
                ? "saved earlier run"
                : "saved sample plan"}
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
                    "The saved preview now shows a stop request. No live service was called.",
                  );
                }}
              >
                {cancelRequested ? "Cancellation requested" : "Request cancellation"}
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
            No model files exist yet. This saved preview only contains the checks that
            happen before a run.
          </p>
        )}
      </div>
      </details>
    </section>
  );
}

"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { DemoExamplePicker } from "@/components/DemoExamplePicker";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  DEMO_EXAMPLE_CATALOG,
  getDemoExamplePreset,
  getPresetExample,
  type DemoExamplePresetId,
} from "@/lib/demo/exampleCatalog";
import { createDemoGateway } from "@/lib/demo/gateway";
import { buildDemoModelRegistry, findRegistryModel } from "@/lib/demo/registry";
import type { DemoInferenceResponse, DemoModelEntry } from "@/lib/demo/types";
import {
  formatLatencyPlain,
  formatQualityPlain,
  plainModelLabel,
} from "@/lib/plainLanguage";
import { buildStageHref } from "@/lib/navigation";
import type { StageBundle } from "@/lib/types";

type DemoOkResult = Extract<DemoInferenceResponse, { status: "ok" }>;

function comparePair(models: DemoModelEntry[]): DemoModelEntry[] {
  const original = models.find((model) => model.arm_id === "student_base");
  const taught = models.find((model) => model.arm_id === "sequence_kd");
  return [original, taught].filter(
    (model): model is DemoModelEntry => model !== undefined,
  );
}

export function DemoStage({ bundle }: { bundle: StageBundle }) {
  const registry = useMemo(() => buildDemoModelRegistry(bundle), [bundle]);
  const models = useMemo(() => comparePair(registry.models), [registry.models]);
  const gateway = useMemo(() => createDemoGateway(), []);
  const [selectedPresetId, setSelectedPresetId] =
    useState<DemoExamplePresetId>("server-purchase");
  const selectedPreset = getDemoExamplePreset(selectedPresetId);
  const example = getPresetExample(selectedPreset);
  const [plainInput, setPlainInput] = useState<string>(
    DEMO_EXAMPLE_CATALOG[0].inferenceInput,
  );
  const [rawInput, setRawInput] = useState(() =>
    JSON.stringify(example.input, null, 2),
  );
  const [results, setResults] = useState<DemoInferenceResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  function selectExample(id: DemoExamplePresetId) {
    const nextPreset = getDemoExamplePreset(id);
    const nextExample = getPresetExample(nextPreset);
    setSelectedPresetId(id);
    setPlainInput(nextPreset.inferenceInput);
    setRawInput(JSON.stringify(nextExample.input, null, 2));
    setResults([]);
    setError(null);
  }

  async function runComparison() {
    let input: Record<string, unknown>;
    try {
      const parsed = JSON.parse(rawInput) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("The raw example must be a JSON object.");
      }
      input = {
        ...(parsed as Record<string, unknown>),
        plain_language_request: plainInput,
      };
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "The raw example is not valid JSON.",
      );
      return;
    }

    if (models.length !== 2) {
      setError("The saved demo needs both the original and taught model arms.");
      return;
    }

    setError(null);
    setRunning(true);
    try {
      setResults(
        await Promise.all(
          models.map((model) =>
            gateway.infer(registry, {
              model_id: model.model_id,
              task: selectedPreset.task,
              example_id: example.example_id,
              input,
              mode: "fixture_preview",
            }),
          ),
        ),
      );
    } finally {
      setRunning(false);
    }
  }

  const heading = running
    ? "Running both models"
    : results.length > 0
      ? "Compare the answers"
      : "Compare models on saved demo data";
  const okResults = results.filter(
    (result): result is DemoOkResult => result.status === "ok",
  );
  const rankedResults = [...okResults].sort(
    (left, right) => (right.score ?? -1) - (left.score ?? -1),
  );
  const recommended = rankedResults[0] ?? null;
  const comparison = rankedResults[1] ?? null;
  const recommendedModel = recommended
    ? findRegistryModel(registry, recommended.model_id)
    : null;
  const qualityDifference =
    recommended?.score !== null &&
    recommended?.score !== undefined &&
    comparison?.score !== null &&
    comparison?.score !== undefined
      ? Math.round((recommended.score - comparison.score) * 100)
      : null;
  const speedDifference =
    recommended?.latency_ms !== null &&
    recommended?.latency_ms !== undefined &&
    comparison?.latency_ms !== null &&
    comparison?.latency_ms !== undefined
      ? Math.abs(recommended.latency_ms - comparison.latency_ms)
      : null;

  return (
    <section aria-labelledby="demo-heading" className="grid gap-4">
      <Card
        className="rounded-[20px] border-0 bg-card shadow-none ring-1 ring-black/10"
        data-testid="demo-hero"
      >
        <CardContent className="grid gap-3 pt-4">
          <div className="flex items-center gap-2">
            <StatusBadge tone="precomputed">Saved demo data</StatusBadge>
            <span className="text-sm text-muted-foreground">Not live inference</span>
          </div>

          <div>
            <h1 id="demo-heading" className="font-serif text-3xl tracking-tight sm:text-4xl">
              {heading}
            </h1>
            <p className="mt-1 text-sm text-muted-foreground sm:text-base">
              Bring examples, teach a smaller model, and verify it works.
            </p>
          </div>

          <DemoExamplePicker
            selectedId={selectedPresetId}
            onSelect={selectExample}
          />

          <div className="grid gap-2">
            <Label htmlFor="demo-plain-input">Review or edit the input</Label>
            <Textarea
              id="demo-plain-input"
              data-testid="demo-plain-input"
              className="min-h-20 bg-background/60 text-sm sm:text-base"
              value={plainInput}
              onChange={(event) => setPlainInput(event.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Your edits stay here until you choose another example.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-sm">
            <strong>Compare:</strong>
            {models.map((model) => (
              <span
                key={model.model_id}
                className="rounded-full border border-border bg-background px-2.5 py-1"
              >
                {plainModelLabel(model.arm_id, model.display_name)}
              </span>
            ))}
          </div>

          <Button
            type="button"
            size="lg"
            className="w-full sm:w-fit sm:min-w-40"
            data-testid="demo-run"
            disabled={running}
            onClick={runComparison}
          >
            {running ? "Running..." : "Run comparison"}
          </Button>

          <details className="rounded-xl border border-black/15 px-3 py-2">
            <summary className="min-h-11 cursor-pointer py-3 text-sm font-medium">
              Edit raw example
            </summary>
            <div className="mt-3 grid gap-2">
              <Label htmlFor="demo-input">Raw example JSON</Label>
              <Textarea
                id="demo-input"
                className="min-h-40 font-mono text-xs"
                value={rawInput}
                spellCheck={false}
                onChange={(event) => setRawInput(event.target.value)}
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="w-fit"
                onClick={() => setRawInput(JSON.stringify(example.input, null, 2))}
              >
                Reset example
              </Button>
            </div>
          </details>
        </CardContent>
      </Card>

      {error ? (
        <p
          role="alert"
          className="rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive"
        >
          {error}
        </p>
      ) : null}

      <Card className="rounded-[20px] border-0 bg-card shadow-none ring-1 ring-black/10">
        <CardHeader>
          <CardTitle className="font-serif text-xl">Results</CardTitle>
        </CardHeader>
        <CardContent>
          {results.length === 0 && !running ? (
            <p className="text-sm text-muted-foreground">
              Run the example to compare both saved model answers.
            </p>
          ) : null}
          {recommended ? (
            <div
              className="mb-4 grid gap-3 rounded-xl border border-border bg-secondary/35 p-4"
              data-testid="demo-decision"
            >
              <div>
                <p className="text-kicker text-[var(--orange)]">Decision</p>
                <h2 className="mt-1 font-serif text-xl">
                  {recommendedModel?.arm_id === "student_base"
                    ? "Keep the current base model for this example"
                    : "Prefer the taught smaller model for the next check"}
                </h2>
                <p className="mt-1 text-sm">
                  {qualityDifference === null
                    ? "The saved outputs are ready for a human review."
                    : qualityDifference === 0
                      ? "Both models matched the known answer equally well."
                      : `The leading model scored ${qualityDifference} percentage points higher on this saved example.`}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Confidence comes from saved demo data. This is not live inference.
                </p>
              </div>
              <dl className="grid gap-2 sm:grid-cols-3">
                <ComparisonFact
                  term="Quality"
                  value={
                    qualityDifference === null
                      ? "Not scored"
                      : `${qualityDifference} point difference`
                  }
                />
                <ComparisonFact
                  term="Speed"
                  value={
                    speedDifference === null
                      ? "Not measured"
                      : `${speedDifference} ms difference`
                  }
                />
                <ComparisonFact term="Cost" value="Not measured in a live run" />
              </dl>
              <Link
                href={buildStageHref("/prove", bundle.mode, bundle.run.run_id)}
                className="btn btn-primary w-fit"
              >
                Review the saved proof
              </Link>
            </div>
          ) : null}
          <div className="grid gap-3 md:grid-cols-2" data-testid="demo-results">
            {results.map((result) => (
              <ResultCard
                key={result.model_id}
                result={result}
                model={findRegistryModel(registry, result.model_id)}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}

function ResultCard({
  result,
  model,
}: {
  result: DemoInferenceResponse;
  model: DemoModelEntry | null;
}) {
  const name = model
    ? plainModelLabel(model.arm_id, model.display_name)
    : result.model_id;

  if (result.status !== "ok") {
    return (
      <article
        className="rounded-xl border border-border p-4"
        data-testid={`demo-result-${result.model_id}`}
        data-status={result.status}
      >
        <h2 className="font-serif text-lg">{name}</h2>
        <p role="alert" className="mt-2 text-sm">{result.message}</p>
        <details className="mt-2 text-xs text-muted-foreground">
          <summary className="min-h-11 cursor-pointer py-3 font-medium">
            Advanced error details
          </summary>
          Error code: <code>{result.code}</code>
        </details>
      </article>
    );
  }

  const confidence =
    typeof result.structured_output.confidence === "number"
      ? `${Math.round(result.structured_output.confidence * 100)}% in this output`
      : "Not provided";

  return (
    <article
      className="rounded-xl border border-border bg-background/60 p-4"
      data-testid={`demo-result-${result.model_id}`}
      data-status="ok"
      data-provenance={result.provenance}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="font-serif text-lg">{name}</h2>
        <StatusBadge tone="precomputed">Saved demo data</StatusBadge>
      </div>
      <dl className="mt-3 grid gap-2 rounded-xl bg-secondary/35 p-3 text-sm">
        <ComparisonFact
          term="Decision"
          value={plainOutcome(result.structured_output)}
        />
        <ComparisonFact term="Why" value={plainReason(result.structured_output)} />
        <ComparisonFact term="Confidence" value={confidence} />
        <ComparisonFact term="Quality" value={formatQualityPlain(result.score)} />
        <ComparisonFact term="Speed" value={formatLatencyPlain(result.latency_ms)} />
        <ComparisonFact term="Cost" value="Not measured in a live run" />
      </dl>
      <p className="mt-3 text-xs text-muted-foreground">
        Saved demo data. Not live inference.
      </p>
      <details className="mt-3">
        <summary className="min-h-11 cursor-pointer py-3 text-sm font-medium">
          Advanced output details
        </summary>
        <p className="mb-2 text-xs text-muted-foreground">
          Raw result (JSON)
        </p>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-soft/60 p-3 font-mono text-xs">
          {JSON.stringify(result.structured_output, null, 2)}
        </pre>
      </details>
    </article>
  );
}

function ComparisonFact({ term, value }: { term: string; value: string }) {
  return (
    <div className="grid grid-cols-[6rem_1fr] gap-2">
      <dt className="text-muted-foreground">{term}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function plainOutcome(output: Record<string, unknown>): string {
  const action = output.policy_action;
  if (typeof action === "string") {
    if (action === "approve") return "Approve the transaction.";
    if (action === "reject") return "Reject the transaction.";
    return "Send the transaction for review.";
  }
  const direction = output.direction;
  if (typeof direction === "string") {
    return direction === "favorable"
      ? "The variance is favorable."
      : "The variance is unfavorable.";
  }
  const status = output.status;
  if (typeof status === "string") {
    return status === "balanced"
      ? "The cash balances match."
      : "Review the cash exceptions.";
  }
  return "Review the structured result.";
}

function plainReason(output: Record<string, unknown>): string {
  const evidence = output.evidence;
  if (Array.isArray(evidence)) {
    if (typeof evidence[0] === "string") return evidence[0];
    const first = evidence[0];
    if (
      first &&
      typeof first === "object" &&
      "field" in first &&
      "value" in first
    ) {
      return `The output cites ${String(first.field)} with value ${String(first.value)}.`;
    }
  }
  const topDrivers = output.top_drivers;
  if (Array.isArray(topDrivers) && topDrivers.length > 0) {
    return "The result names the largest saved driver first.";
  }
  const exceptions = output.exceptions;
  if (Array.isArray(exceptions)) {
    return exceptions.length > 0
      ? "The result found entries that need review."
      : "The result found no unmatched entries.";
  }
  return "The saved result does not include a plain-language reason.";
}

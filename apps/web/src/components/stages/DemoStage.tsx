"use client";

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
import type { StageBundle } from "@/lib/types";

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

  return (
    <section aria-labelledby="demo-heading" className="grid gap-4">
      <Card className="border-border/80 bg-card/90 shadow-none" data-testid="demo-hero">
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

          <details className="rounded-lg border border-border px-3 py-2">
            <summary className="cursor-pointer text-sm font-medium">
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

      <Card className="border-border/80 bg-card/90 shadow-none">
        <CardHeader>
          <CardTitle className="font-serif text-xl">Results</CardTitle>
        </CardHeader>
        <CardContent>
          {results.length === 0 && !running ? (
            <p className="text-sm text-muted-foreground">
              Run the example to compare both saved model answers.
            </p>
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
      <article className="rounded-xl border border-border p-4" data-status={result.status}>
        <h2 className="font-serif text-lg">{name}</h2>
        <p role="alert" className="mt-2 text-sm">
          {result.code}: {result.message}
        </p>
      </article>
    );
  }

  return (
    <article
      className="rounded-xl border border-border bg-background/60 p-4"
      data-status="ok"
      data-provenance={result.provenance}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="font-serif text-lg">{name}</h2>
        <StatusBadge tone="precomputed">Saved demo data</StatusBadge>
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-3">
        <div>
          <dt className="text-xs text-muted-foreground">Score</dt>
          <dd className="font-serif text-lg">{formatQualityPlain(result.score)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted-foreground">Latency</dt>
          <dd className="font-serif text-lg">{formatLatencyPlain(result.latency_ms)}</dd>
        </div>
      </dl>
      <p className="mt-3 text-xs text-muted-foreground">
        Saved demo data. Not live inference.
      </p>
      <details className="mt-3">
        <summary className="cursor-pointer text-sm font-medium">
          View structured answer
        </summary>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-soft/60 p-3 font-mono text-xs">
          {JSON.stringify(result.structured_output, null, 2)}
        </pre>
      </details>
    </article>
  );
}

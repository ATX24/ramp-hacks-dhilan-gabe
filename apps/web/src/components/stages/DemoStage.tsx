"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { LiveTrainingCard } from "@/components/LiveTrainingCard";
import { StatusBadge } from "@/components/StatusBadge";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  defaultExampleForTask,
  getDemoExample,
  listDemoExamples,
  nextDemoExample,
} from "@/lib/demo/examples";
import {
  createDemoGateway,
  resolveLiveInferenceBaseUrl,
} from "@/lib/demo/gateway";
import {
  formatCi,
  formatCount,
  formatDurationSeconds,
  formatIndex,
  formatRatio,
  formatUnknown,
  formatUsd,
} from "@/lib/demo/format";
import {
  buildDemoModelRegistry,
  defaultWalkthroughModelIds,
  findRegistryModel,
} from "@/lib/demo/registry";
import { buildDemoHref, parseDemoUrlState } from "@/lib/demo/urlState";
import {
  FINANCE_TASKS,
  type DemoInferenceMode,
  type DemoInferenceResponse,
  type DemoModelEntry,
  type DemoRunMode,
  type FinanceTaskId,
} from "@/lib/demo/types";
import { buildStageHref, STAGES } from "@/lib/navigation";
import {
  formatLatencyPlain,
  formatQualityPlain,
  formatUsdPlain,
  plainModelLabel,
  STAGE_PLAIN,
  TASK_PLAIN,
} from "@/lib/plainLanguage";
import type { StageBundle } from "@/lib/types";

function unknownOr(value: string): string {
  return value;
}

function pickBeforeAfter(models: DemoModelEntry[]): {
  before: DemoModelEntry | null;
  after: DemoModelEntry | null;
} {
  const byArm = new Map(models.map((model) => [model.arm_id, model]));
  const before = byArm.get("student_base") ?? models[0] ?? null;
  const after =
    byArm.get("promoted_winner") ??
    byArm.get("sequence_kd") ??
    byArm.get("logit_kd") ??
    models.find((model) => model.model_id !== before?.model_id) ??
    null;
  return { before, after };
}

export function DemoStage({ bundle }: { bundle: StageBundle }) {
  const registry = useMemo(() => buildDemoModelRegistry(bundle), [bundle]);
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const fallbackModels = useMemo(
    () => defaultWalkthroughModelIds(registry),
    [registry],
  );

  const initial = useMemo(
    () => parseDemoUrlState(searchParams, fallbackModels),
    [searchParams, fallbackModels],
  );

  const [task, setTask] = useState<FinanceTaskId>(initial.task);
  const [modelIds, setModelIds] = useState<string[]>(initial.modelIds);
  const [exampleId, setExampleId] = useState(initial.exampleId);
  const [runMode, setRunMode] = useState<DemoRunMode>(initial.runMode);
  const [inferenceMode, setInferenceMode] = useState<DemoInferenceMode>(
    initial.inferenceMode,
  );
  const [inputText, setInputText] = useState(() =>
    JSON.stringify(defaultExampleForTask(initial.task).input, null, 2),
  );
  const [inputError, setInputError] = useState<string | null>(null);
  const [results, setResults] = useState<DemoInferenceResponse[]>([]);
  const [runError, setRunError] = useState<string | null>(null);
  const [isPending, setIsPending] = useState(false);
  const [selectedStatsModelId, setSelectedStatsModelId] = useState(
    initial.modelIds[0] ?? registry.models[0]?.model_id ?? "",
  );

  const example = getDemoExample(exampleId) ?? defaultExampleForTask(task);
  const gateway = useMemo(() => createDemoGateway(), []);
  const liveBaseUrl = resolveLiveInferenceBaseUrl();
  const taskPlain = TASK_PLAIN[task];
  const { before, after } = useMemo(
    () => pickBeforeAfter(registry.models),
    [registry.models],
  );

  useEffect(() => {
    const next = getDemoExample(exampleId) ?? defaultExampleForTask(task);
    setInputText(JSON.stringify(next.input, null, 2));
    setInputError(null);
  }, [exampleId, task]);

  useEffect(() => {
    if (pathname !== "/demo") return;
    const desired = {
      task,
      modelIds,
      exampleId,
      runMode,
      inferenceMode,
    };
    const current = parseDemoUrlState(searchParams, modelIds);
    const same =
      current.task === desired.task &&
      current.exampleId === desired.exampleId &&
      current.runMode === desired.runMode &&
      current.inferenceMode === desired.inferenceMode &&
      current.modelIds.join(",") === desired.modelIds.join(",");
    if (same) return;
    router.replace(
      buildDemoHref(desired, {
        mode: bundle.mode,
        runId: bundle.run.run_id,
      }),
      { scroll: false },
    );
  }, [
    bundle.mode,
    bundle.run.run_id,
    exampleId,
    inferenceMode,
    modelIds,
    pathname,
    router,
    runMode,
    searchParams,
    task,
  ]);

  const selectedModels = modelIds
    .map((id) => findRegistryModel(registry, id))
    .filter((model): model is DemoModelEntry => model !== null);

  const statsModel =
    findRegistryModel(registry, selectedStatsModelId) ??
    selectedModels[0] ??
    registry.models[0] ??
    null;

  function applyTask(nextTask: FinanceTaskId) {
    const nextExample = defaultExampleForTask(nextTask);
    setTask(nextTask);
    setExampleId(nextExample.example_id);
    setResults([]);
    setRunError(null);
  }

  function toggleModel(modelId: string) {
    setModelIds((current) => {
      if (runMode === "single") return [modelId];
      if (current.includes(modelId)) {
        const next = current.filter((id) => id !== modelId);
        return next.length > 0 ? next : current;
      }
      return [...current, modelId].slice(0, 4);
    });
    setSelectedStatsModelId(modelId);
  }

  function resetExample() {
    const fresh = getDemoExample(exampleId) ?? defaultExampleForTask(task);
    setInputText(JSON.stringify(fresh.input, null, 2));
    setInputError(null);
  }

  function randomExample() {
    const next = nextDemoExample(task, exampleId);
    setExampleId(next.example_id);
    setResults([]);
  }

  function parseInput(): Record<string, unknown> | null {
    try {
      const parsed = JSON.parse(inputText) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setInputError("Input must be a JSON object.");
        return null;
      }
      setInputError(null);
      return parsed as Record<string, unknown>;
    } catch {
      setInputError("Input JSON is invalid.");
      return null;
    }
  }

  function runInference() {
    const input = parseInput();
    if (!input) return;
    const targets =
      runMode === "single" ? selectedModels.slice(0, 1) : selectedModels;
    if (targets.length === 0) {
      setRunError("Select at least one model from the registry.");
      return;
    }
    if (runMode === "compare" && targets.length < 2) {
      setRunError("Compare mode requires two or more registry models.");
      return;
    }

    setRunError(null);
    setIsPending(true);
    void (async () => {
      try {
        const nextResults: DemoInferenceResponse[] = [];
        for (const model of targets) {
          const response = await gateway.infer(registry, {
            model_id: model.model_id,
            task,
            example_id: exampleId,
            input,
            mode: inferenceMode,
          });
          nextResults.push(response);
        }
        setResults(nextResults);
      } finally {
        setIsPending(false);
      }
    })();
  }

  function useBeforeAfterPair() {
    if (!before || !after) return;
    setRunMode("compare");
    setModelIds([before.model_id, after.model_id]);
    setSelectedStatsModelId(after.model_id);
    setResults([]);
  }

  return (
    <section aria-labelledby="demo-heading" className="grid gap-4">
      <Card className="overflow-hidden border-border/80 bg-card/90 shadow-none">
        <CardHeader className="gap-4 border-b border-border/70 pb-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-2xl">
              <p className="text-kicker text-muted-foreground">Playground</p>
              <h2
                id="demo-heading"
                className="mt-2 font-serif text-3xl font-normal tracking-tight md:text-4xl"
              >
                Demo
              </h2>
              <p className="mt-3 text-base leading-relaxed text-muted-foreground">
                Start with a tangible before/after. Choose a familiar finance
                task, compare the original and smaller model, then scroll to see
                how Curate → Synthesize → Train → Prove made that result.
              </p>
            </div>
            <aside
              className="w-full max-w-sm rounded-xl border border-border bg-soft/50 p-4"
              data-testid="demo-walkthrough"
            >
              <h3 className="font-serif text-lg font-normal">Judge walkthrough</h3>
              <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-muted-foreground">
                <li>Pick a finance task you recognize.</li>
                <li>Compare original vs taught smaller model.</li>
                <li>Read speed, quality, and cost in plain language.</li>
                <li>Open advanced details only if you want hashes or metrics.</li>
              </ol>
            </aside>
          </div>
        </CardHeader>
        <CardContent className="grid gap-4 pt-5">
          <div className="flex flex-wrap gap-2">
            <StatusBadge tone={liveBaseUrl ? "pass" : "unavailable"}>
              {liveBaseUrl ? "Live gateway configured" : "Live gateway unavailable"}
            </StatusBadge>
            <StatusBadge tone="precomputed">Fixture preview available</StatusBadge>
            <span className="text-sm text-muted-foreground">
              Registry run <code>{registry.run_id}</code> · {registry.models.length}{" "}
              models
            </span>
          </div>

          <div>
            <p className="mb-2 text-sm font-medium">Familiar finance tasks</p>
            <Suggestions className="pb-1">
              {FINANCE_TASKS.map((item) => (
                <Suggestion
                  key={item.id}
                  suggestion={item.id}
                  data-testid={`demo-task-suggestion-${item.id}`}
                  variant={task === item.id ? "default" : "outline"}
                  onClick={() => applyTask(item.id)}
                >
                  {TASK_PLAIN[item.id].title}
                </Suggestion>
              ))}
            </Suggestions>
          </div>

          <div className="rounded-xl border border-border bg-background/50 p-4">
            <p className="text-kicker text-muted-foreground">Current task</p>
            <h3 className="mt-1 font-serif text-2xl font-normal">{taskPlain.title}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{taskPlain.blurb}</p>
            <p className="mt-2 text-sm">
              <strong className="font-medium">Judge question:</strong>{" "}
              {taskPlain.judgePrompt}
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <PlainModelCard
              eyebrow="Before"
              model={before}
              empty="Original model unavailable in this fixture."
            />
            <PlainModelCard
              eyebrow="After"
              model={after}
              empty="Taught smaller model unavailable until a proved/trained fixture is loaded."
            />
          </div>

          {before && after ? (
            <Button
              type="button"
              variant="secondary"
              className="w-fit"
              onClick={useBeforeAfterPair}
            >
              Compare original vs smaller model
            </Button>
          ) : null}
        </CardContent>
      </Card>

      <LiveTrainingCard
        run={bundle.run}
        plan={bundle.plan}
        telemetry={bundle.training_telemetry}
        artifact={bundle.artifact}
      />

      <Card className="border-border/80 bg-card/90 shadow-none">
        <CardHeader>
          <CardTitle className="font-serif text-xl font-normal">
            Ask both models
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Controls keep the same shareable URL state. Fixture previews are
            labeled. Live calls only run through the typed serving gateway.
          </p>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div
            className="grid gap-3 md:grid-cols-2 xl:grid-cols-4"
            role="group"
            aria-label="Demo controls"
          >
            <div className="grid gap-2">
              <Label htmlFor="demo-task">Finance task</Label>
              <select
                id="demo-task"
                data-testid="demo-task-select"
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm"
                value={task}
                onChange={(event) => applyTask(event.target.value as FinanceTaskId)}
              >
                {FINANCE_TASKS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {TASK_PLAIN[item.id].title}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="demo-example">Example</Label>
              <select
                id="demo-example"
                data-testid="demo-example-select"
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm"
                value={exampleId}
                onChange={(event) => {
                  setExampleId(event.target.value);
                  setResults([]);
                }}
              >
                {listDemoExamples(task).map((item) => (
                  <option key={item.example_id} value={item.example_id}>
                    {item.label} ({item.example_id})
                  </option>
                ))}
              </select>
            </div>

            <div className="grid gap-2">
              <span id="demo-run-mode-label" className="text-sm font-medium">
                Run mode
              </span>
              <ToggleGroup
                type="single"
                value={runMode}
                onValueChange={(value) => {
                  if (value !== "single" && value !== "compare") return;
                  setRunMode(value);
                  if (value === "single") {
                    setModelIds((current) => current.slice(0, 1));
                  }
                }}
                aria-labelledby="demo-run-mode-label"
                variant="outline"
              >
                <ToggleGroupItem
                  value="single"
                  data-testid="demo-run-mode-single"
                  aria-pressed={runMode === "single"}
                >
                  One model
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="compare"
                  data-testid="demo-run-mode-compare"
                  aria-pressed={runMode === "compare"}
                >
                  Compare 2+
                </ToggleGroupItem>
              </ToggleGroup>
            </div>

            <div className="grid gap-2">
              <span id="demo-infer-mode-label" className="text-sm font-medium">
                Inference
              </span>
              <ToggleGroup
                type="single"
                value={inferenceMode}
                onValueChange={(value) => {
                  if (value !== "fixture_preview" && value !== "live") return;
                  setInferenceMode(value);
                }}
                aria-labelledby="demo-infer-mode-label"
                variant="outline"
              >
                <ToggleGroupItem
                  value="fixture_preview"
                  data-testid="demo-infer-fixture"
                  aria-pressed={inferenceMode === "fixture_preview"}
                >
                  Fixture preview
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="live"
                  data-testid="demo-infer-live"
                  aria-pressed={inferenceMode === "live"}
                >
                  Live
                </ToggleGroupItem>
              </ToggleGroup>
            </div>
          </div>

          <fieldset className="rounded-xl border border-border p-3">
            <legend className="px-1 text-sm font-medium">Models from registry</legend>
            {registry.models.length === 0 ? (
              <p data-testid="demo-models-empty">No models are registered for this run.</p>
            ) : (
              <ul className="grid gap-2" data-testid="demo-model-list">
                {registry.models.map((model) => {
                  const checked = modelIds.includes(model.model_id);
                  return (
                    <li key={model.model_id}>
                      <label className="flex cursor-pointer items-start gap-3 rounded-lg border border-transparent px-2 py-2 hover:border-border hover:bg-soft/40">
                        <input
                          type={runMode === "single" ? "radio" : "checkbox"}
                          name="demo-model"
                          value={model.model_id}
                          checked={checked}
                          data-testid={`demo-model-${model.arm_id}`}
                          onChange={() => toggleModel(model.model_id)}
                          className="mt-1"
                        />
                        <span>
                          <strong>
                            {plainModelLabel(model.arm_id, model.display_name)}
                          </strong>
                          <span className="mt-0.5 block text-xs text-muted-foreground">
                            <code>{model.model_id}</code> · {model.serving.availability}
                            {model.excluded ? " · excluded" : ""}
                            {" · "}
                            {model.display_name}
                          </span>
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            )}
          </fieldset>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="border-border/80 bg-card/90 shadow-none">
          <CardHeader>
            <CardTitle className="font-serif text-xl font-normal">Prompt</CardTitle>
            <p className="text-sm text-muted-foreground">
              Example <code>{example.example_id}</code> · split{" "}
              <code>{example.split}</code> · {example.difficulty}
            </p>
          </CardHeader>
          <CardContent className="grid gap-3">
            <Label className="sr-only" htmlFor="demo-input">
              Example input JSON
            </Label>
            <Textarea
              id="demo-input"
              data-testid="demo-input"
              spellCheck={false}
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              className="min-h-48 font-mono text-xs"
            />
            {inputError ? (
              <p
                className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
                role="alert"
                data-testid="demo-input-error"
              >
                {inputError}
              </p>
            ) : null}
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="outline"
                data-testid="demo-reset-example"
                onClick={resetExample}
              >
                Reset example
              </Button>
              <Button
                type="button"
                variant="outline"
                data-testid="demo-random-example"
                onClick={randomExample}
              >
                Random example
              </Button>
              <Button
                type="button"
                data-testid="demo-run"
                disabled={isPending}
                onClick={runInference}
              >
                {isPending
                  ? "Running…"
                  : runMode === "compare"
                    ? "Compare models"
                    : "Run model"}
              </Button>
            </div>
            <p data-testid="demo-expected-schema" className="text-sm text-muted-foreground">
              <code>{example.expected_output_schema.schema_version}</code> —{" "}
              {example.expected_output_schema.description}
            </p>
          </CardContent>
        </Card>

        <Card
          className="border-border/80 bg-card/90 shadow-none"
          data-testid="demo-stats-panel"
        >
          <CardHeader>
            <CardTitle className="font-serif text-xl font-normal">
              Speed, quality, cost
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3">
            {statsModel ? (
              <>
                <Label htmlFor="demo-stats-model">Inspect model</Label>
                <select
                  id="demo-stats-model"
                  data-testid="demo-stats-model"
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 text-sm"
                  value={statsModel.model_id}
                  onChange={(event) => setSelectedStatsModelId(event.target.value)}
                >
                  {registry.models.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {plainModelLabel(model.arm_id, model.display_name)}
                    </option>
                  ))}
                </select>
                <PlainEconomics model={statsModel} />
                <details className="rounded-xl border border-border p-3">
                  <summary className="cursor-pointer font-medium">
                    Advanced metrics & provenance
                  </summary>
                  <DemoStatsTable model={statsModel} />
                </details>
              </>
            ) : (
              <p data-testid="demo-stats-empty">No model available for statistics.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/80 bg-card/90 shadow-none">
        <CardHeader>
          <CardTitle className="font-serif text-xl font-normal">Results</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          {runError ? (
            <p
              className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm"
              role="alert"
              data-testid="demo-run-error"
            >
              {runError}
            </p>
          ) : null}
          {isPending ? (
            <p data-testid="demo-results-loading" role="status">
              Running selected model(s)…
            </p>
          ) : null}
          {!isPending && results.length === 0 ? (
            <p data-testid="demo-results-empty">
              No results yet. Choose models and run a fixture preview or live call.
            </p>
          ) : null}
          <div
            className={
              results.length > 1
                ? "grid gap-3 md:grid-cols-2"
                : "grid gap-3"
            }
            data-testid="demo-results"
          >
            {results.map((result) => (
              <DemoResultCard
                key={`${result.model_id}-${result.status}-${result.example_id}`}
                result={result}
                model={findRegistryModel(registry, result.model_id)}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      <Card
        className="border-border/80 bg-card/90 shadow-none"
        data-testid="demo-journey"
      >
        <CardHeader>
          <CardTitle className="font-serif text-2xl font-normal">
            How this result was made
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            The playground shows the outcome. The Distillery journey explains the
            evidence path behind it.
          </p>
        </CardHeader>
        <CardContent>
          <ol className="grid gap-3 md:grid-cols-2">
            {STAGES.filter((stage) => stage.id !== "demo").map((stage) => (
              <li key={stage.id}>
                <Link
                  href={buildStageHref(stage.href, bundle.mode, bundle.run.run_id)}
                  className="block rounded-xl border border-border bg-background/60 p-4 transition hover:border-orange/50"
                >
                  <p className="text-kicker text-muted-foreground">
                    {stage.index} · {stage.name}
                  </p>
                  <p className="mt-1 font-serif text-xl">{stage.plain}</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {STAGE_PLAIN[stage.id].description}
                  </p>
                </Link>
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>
    </section>
  );
}

function PlainModelCard({
  eyebrow,
  model,
  empty,
}: {
  eyebrow: string;
  model: DemoModelEntry | null;
  empty: string;
}) {
  if (!model) {
    return (
      <div className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
        <p className="text-kicker">{eyebrow}</p>
        <p className="mt-2">{empty}</p>
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-border bg-background/60 p-4">
      <p className="text-kicker text-muted-foreground">{eyebrow}</p>
      <p className="mt-1 font-serif text-xl">
        {plainModelLabel(model.arm_id, model.display_name)}
      </p>
      <p className="mt-1 text-sm text-muted-foreground">
        {MODEL_BLURB(model)}
      </p>
      <ul className="mt-3 space-y-1 text-sm">
        <li>
          Size:{" "}
          {model.stats.advertised_parameter_count
            ? formatCount(model.stats.advertised_parameter_count)
            : "unknown"}
        </li>
        <li>Training cost: {formatUsdPlain(model.stats.training_cost_usd)}</li>
        <li>
          Quality index: {formatQualityPlain(model.stats.iid_primary_index)}
        </li>
      </ul>
    </div>
  );
}

function MODEL_BLURB(model: DemoModelEntry): string {
  if (model.arm_id === "student_base") {
    return "The starting smaller model before Distillery teaching.";
  }
  if (model.arm_id === "promoted_winner" || model.arm_id === "sequence_kd") {
    return "The smaller model after Distillery teaching.";
  }
  return model.purpose;
}

function PlainEconomics({ model }: { model: DemoModelEntry }) {
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      <div className="rounded-xl border border-border p-3">
        <p className="text-kicker text-muted-foreground">Speed</p>
        <p className="mt-1 font-serif text-lg">
          {model.stats.training_duration_seconds
            ? formatDurationSeconds(model.stats.training_duration_seconds)
            : "unknown at serve-time"}
        </p>
        <p className="text-xs text-muted-foreground">Training duration evidence</p>
      </div>
      <div className="rounded-xl border border-border p-3">
        <p className="text-kicker text-muted-foreground">Quality</p>
        <p className="mt-1 font-serif text-lg">
          {formatQualityPlain(model.stats.iid_primary_index)}
        </p>
        <p className="text-xs text-muted-foreground">Held-out finance score</p>
      </div>
      <div className="rounded-xl border border-border p-3">
        <p className="text-kicker text-muted-foreground">Cost</p>
        <p className="mt-1 font-serif text-lg">
          {formatUsdPlain(model.stats.training_cost_usd)}
        </p>
        <p className="text-xs text-muted-foreground">Teaching spend evidence</p>
      </div>
    </div>
  );
}

function DemoStatsTable({ model }: { model: DemoModelEntry }) {
  const rows: Array<[string, string]> = [
    ["Advertised parameters", formatCount(model.stats.advertised_parameter_count)],
    ["Adapter parameters", formatCount(model.stats.adapter_parameter_count)],
    ["Compression", formatRatio(model.stats.compression_ratio)],
    ["Recipe", formatUnknown(model.stats.recipe)],
    [
      "Teacher",
      model.stats.teacher
        ? `${model.stats.teacher.id}@${model.stats.teacher.revision.slice(0, 12)}`
        : "unknown",
    ],
    [
      "Student",
      model.stats.student
        ? `${model.stats.student.id}@${model.stats.student.revision.slice(0, 12)}`
        : "unknown",
    ],
    ["Seed", formatUnknown(model.stats.seed)],
    ["Data hash", formatUnknown(model.stats.data_hash)],
    ["Manifest hash", formatUnknown(model.stats.manifest_hash)],
    ["Artifact hash", formatUnknown(model.stats.artifact_hash)],
    ["Training duration", formatDurationSeconds(model.stats.training_duration_seconds)],
    ["Training cost", formatUsd(model.stats.training_cost_usd)],
    ["IID primary index", formatIndex(model.stats.iid_primary_index)],
    ["IID 95% CI", formatCi(model.stats.iid_ci_low, model.stats.iid_ci_high)],
    ["OOD retention", formatIndex(model.stats.ood_retention)],
    ["OOD 95% CI", formatCi(model.stats.ood_ci_low, model.stats.ood_ci_high)],
    ["Proof status", formatUnknown(model.stats.proof_status)],
    ["Promotion status", unknownOr(model.stats.promotion_status)],
  ];

  return (
    <div className="table-wrap mt-2">
      <table className="data">
        <thead>
          <tr>
            <th scope="col">Field</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td>{label}</td>
              <td>
                <span
                  className={value === "unknown" ? "demo-unknown" : "mono"}
                  data-unknown={value === "unknown" ? "true" : "false"}
                >
                  {value}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DemoResultCard({
  result,
  model,
}: {
  result: DemoInferenceResponse;
  model: DemoModelEntry | null;
}) {
  if (result.status === "unavailable" || result.status === "error") {
    return (
      <article
        className="rounded-xl border border-border p-4"
        data-testid={`demo-result-${result.model_id}`}
        data-status={result.status}
      >
        <header className="mb-2 flex flex-wrap items-center gap-2">
          <strong>
            {model
              ? plainModelLabel(model.arm_id, model.display_name)
              : result.model_id}
          </strong>
          <StatusBadge tone="unavailable">{result.status}</StatusBadge>
          <code>{result.code}</code>
        </header>
        <p role="alert">{result.message}</p>
      </article>
    );
  }

  return (
    <Message
      from="assistant"
      className="max-w-none rounded-xl border border-border p-4"
      data-testid={`demo-result-${result.model_id}`}
      data-status="ok"
      data-provenance={result.provenance}
    >
      <header className="mb-2 flex flex-wrap items-center gap-2">
        <strong>
          {model
            ? plainModelLabel(model.arm_id, model.display_name)
            : result.model_id}
        </strong>
        <StatusBadge
          tone={result.provenance === "fixture_preview" ? "precomputed" : "pass"}
        >
          {result.label}
        </StatusBadge>
        <StatusBadge
          tone={
            result.validation === "valid"
              ? "pass"
              : result.validation === "invalid"
                ? "fail"
                : "pending"
          }
        >
          {result.validation}
        </StatusBadge>
      </header>
      <MessageContent className="max-w-none">
        <ul className="mb-3 space-y-1 text-sm">
          <li>Latency: {formatLatencyPlain(result.latency_ms)}</li>
          <li>
            Tokens: prompt{" "}
            {result.prompt_tokens === null ? "unknown" : result.prompt_tokens},
            completion{" "}
            {result.completion_tokens === null ? "unknown" : result.completion_tokens}
          </li>
          <li>
            Score: {formatQualityPlain(result.score)}
            {result.score_detail ? ` — ${result.score_detail}` : ""}
          </li>
          {result.validation_detail ? <li>{result.validation_detail}</li> : null}
        </ul>
        <h4 className="font-serif text-base">Structured output</h4>
        <pre
          className="mt-1 overflow-x-auto rounded-lg bg-soft/60 p-3 font-mono text-xs"
          data-testid={`demo-structured-${result.model_id}`}
        >
          {JSON.stringify(result.structured_output, null, 2)}
        </pre>
        <h4 className="mt-3 font-serif text-base">Raw JSON</h4>
        <pre
          className="mt-1 overflow-x-auto rounded-lg bg-soft/60 p-3 font-mono text-xs"
          data-testid={`demo-raw-${result.model_id}`}
        >
          {result.raw_json}
        </pre>
      </MessageContent>
    </Message>
  );
}

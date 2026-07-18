"use client";

import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { LayoutListIcon, SearchIcon, ShuffleIcon } from "lucide-react";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
} from "@/components/ai-elements/conversation";
import { Loader } from "@/components/ai-elements/loader";
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import { DemoResultCard } from "@/components/demo/DemoResultCard";
import { DemoStatsTable } from "@/components/demo/DemoStatsTable";
import {
  StatusBadge,
  armBadgeLabel,
  armTone,
} from "@/components/StatusBadge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Panel,
  PanelBody,
  PanelDescription,
  PanelHeader,
  PanelTitle,
} from "@/components/ui/panel";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
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
  formatIndex,
  formatRatio,
  formatUnknown,
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
import type { StageBundle } from "@/lib/types";
import { cn } from "@/lib/utils";

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
  const [exampleQuery, setExampleQuery] = useState("");
  const [selectedStatsModelId, setSelectedStatsModelId] = useState(
    initial.modelIds[0] ?? registry.models[0]?.model_id ?? "",
  );

  const example = getDemoExample(exampleId) ?? defaultExampleForTask(task);
  const gateway = useMemo(() => createDemoGateway(), []);
  const liveBaseUrl = resolveLiveInferenceBaseUrl();
  const liveAvailable =
    Boolean(liveBaseUrl) &&
    registry.models.some((model) => model.serving.availability === "live");

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
  const examples = listDemoExamples(task);
  const filteredExamples = examples.filter((item) => {
    const query = exampleQuery.trim().toLowerCase();
    if (!query) return true;
    return (
      item.label.toLowerCase().includes(query) ||
      item.example_id.toLowerCase().includes(query) ||
      item.difficulty.toLowerCase().includes(query)
    );
  });
  const canRun = inferenceMode === "fixture_preview" || liveAvailable;

  function applyTask(nextTask: FinanceTaskId) {
    const nextExample = defaultExampleForTask(nextTask);
    setTask(nextTask);
    setExampleId(nextExample.example_id);
    setExampleQuery("");
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
        setInputError("Use one JSON object for the example input.");
        return null;
      }
      setInputError(null);
      return parsed as Record<string, unknown>;
    } catch {
      setInputError("The example input is not valid JSON.");
      return null;
    }
  }

  function runInference() {
    if (!canRun) {
      setRunError("A live endpoint is not connected. Use the saved demo output.");
      return;
    }
    const input = parseInput();
    if (!input) return;
    const targets =
      runMode === "single" ? selectedModels.slice(0, 1) : selectedModels;
    if (targets.length === 0) {
      setRunError("Choose at least one model.");
      return;
    }
    if (runMode === "compare" && targets.length < 2) {
      setRunError("Choose at least two models to compare them.");
      return;
    }

    setRunError(null);
    setIsPending(true);
    void (async () => {
      try {
        const nextResults: DemoInferenceResponse[] = [];
        for (const model of targets) {
          nextResults.push(
            await gateway.infer(registry, {
              model_id: model.model_id,
              task,
              example_id: exampleId,
              input,
              mode: inferenceMode,
            }),
          );
        }
        setResults(nextResults);
      } finally {
        setIsPending(false);
      }
    })();
  }

  return (
    <section aria-labelledby="demo-heading" className="grid gap-4">
      <Panel>
        <PanelHeader className="grid gap-4 md:grid-cols-[minmax(0,1.4fr)_minmax(16rem,0.9fr)]">
          <div>
            <p className="text-kicker text-[var(--orange)]">Demo</p>
            <h1
              id="demo-heading"
              className="font-serif text-3xl tracking-[-0.04em]"
            >
              Try the result
            </h1>
            <PanelDescription className="mt-2">
              Pick a finance job and a saved example. You can compare saved outputs.
              A live call only happens when a working endpoint is connected.
            </PanelDescription>
          </div>
          <aside
            className="rounded-[14px] border border-border bg-secondary/35 p-4"
            data-testid="demo-walkthrough"
          >
            <h2 className="mb-2 font-serif text-lg">A short judge path</h2>
            <ol className="m-0 list-decimal space-y-1.5 pl-4 text-sm text-muted-foreground">
              <li>Start with transaction review and compare two models.</li>
              <li>Read the decision, why, confidence, speed, and cost.</li>
              <li>Try budget variance, then cash matching.</li>
              <li>Open Advanced only when you need the technical record.</li>
            </ol>
          </aside>
        </PanelHeader>
        <PanelBody className="grid gap-5">
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <StatusBadge tone={liveAvailable ? "pass" : "unavailable"}>
              {liveAvailable ? "Live endpoint connected" : "Live endpoint unavailable"}
            </StatusBadge>
            <StatusBadge tone="precomputed">Saved demo outputs ready</StatusBadge>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <DemoSetting
              label="Job"
              helper="This changes the kind of finance work in the example. Transaction review is the default. Changing it also picks a matching sample."
            >
              <Select
                value={task}
                onValueChange={(value) => applyTask(value as FinanceTaskId)}
              >
                <SelectTrigger id="demo-task" data-testid="demo-task-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {FINANCE_TASKS.map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </DemoSetting>

            <DemoSetting
              label="Saved example"
              helper="This changes the input without changing the job. Leave the first example selected for the short judge path."
            >
              <div className="relative">
                <SearchIcon className="absolute top-2.5 left-3 size-4 text-muted-foreground" />
                <Input
                  type="search"
                  value={exampleQuery}
                  onChange={(event) => setExampleQuery(event.target.value)}
                  placeholder="Search saved examples"
                  className="pl-9"
                  aria-label="Search saved examples"
                  data-testid="demo-example-search"
                />
              </div>
              <Select
                value={exampleId}
                onValueChange={(value) => {
                  setExampleId(value);
                  setResults([]);
                }}
              >
                <SelectTrigger id="demo-example" data-testid="demo-example-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {filteredExamples.length > 0 ? (
                    filteredExamples.map((item) => (
                      <SelectItem key={item.example_id} value={item.example_id}>
                        {item.label} ({item.example_id})
                      </SelectItem>
                    ))
                  ) : (
                    <p className="px-3 py-2 text-sm text-muted-foreground">
                      No saved examples match.
                    </p>
                  )}
                </SelectContent>
              </Select>
            </DemoSetting>

            <DemoSetting
              label="View"
              helper="This changes how many models run. One model is faster to read. Compare uses every model you check below."
            >
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
                className="justify-start"
                aria-label="Choose one model or compare models"
              >
                <ToggleGroupItem
                  value="single"
                  className="min-h-11 rounded-full px-4"
                  data-testid="demo-run-mode-single"
                >
                  One model
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="compare"
                  className="min-h-11 rounded-full px-4"
                  data-testid="demo-run-mode-compare"
                >
                  Compare models
                </ToggleGroupItem>
              </ToggleGroup>
            </DemoSetting>

            <DemoSetting
              label="Output source"
              helper={
                liveAvailable
                  ? "This changes where the output comes from. Saved demo makes no call. Live uses the connected endpoint."
                  : "Saved demo makes no call. Live is disabled because this project has no connected endpoint."
              }
            >
              <ToggleGroup
                type="single"
                value={inferenceMode}
                onValueChange={(value) => {
                  if (value === "fixture_preview") {
                    setInferenceMode(value);
                  }
                  if (value === "live" && liveAvailable) {
                    setInferenceMode(value);
                  }
                }}
                className="justify-start"
                aria-label="Choose saved demo or live output"
              >
                <ToggleGroupItem
                  value="fixture_preview"
                  className="min-h-11 rounded-full px-4"
                  data-testid="demo-infer-fixture"
                >
                  Saved demo
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="live"
                  className="min-h-11 rounded-full px-4"
                  data-testid="demo-infer-live"
                  disabled={!liveAvailable}
                  aria-describedby={!liveAvailable ? "live-unavailable-reason" : undefined}
                >
                  Live endpoint
                </ToggleGroupItem>
              </ToggleGroup>
              {!liveAvailable ? (
                <p
                  id="live-unavailable-reason"
                  className="text-sm text-muted-foreground"
                >
                  Live output stays off until an endpoint and a loadable model file are
                  connected.
                </p>
              ) : null}
            </DemoSetting>
          </div>

          <fieldset className="grid gap-3">
            <legend className="font-serif text-lg">Models</legend>
            <p className="text-sm text-muted-foreground">
              This changes which output you see. The default compares the base model
              with TinyFable Generalist. In one-model view, the first checked model
              runs.
            </p>
            {registry.models.length === 0 ? (
              <p data-testid="demo-models-empty">
                This run does not list any models.
              </p>
            ) : (
              <ul
                className="m-0 grid list-none gap-2 p-0 sm:grid-cols-2"
                data-testid="demo-model-list"
              >
                {registry.models.map((model) => {
                  const checked = modelIds.includes(model.model_id);
                  return (
                    <li key={model.model_id}>
                      <label
                        className={cn(
                          "grid min-h-14 cursor-pointer grid-cols-[auto_1fr_auto] items-start gap-3 rounded-[14px] border border-border bg-card p-3",
                          checked && "border-[var(--orange)] bg-secondary/35",
                        )}
                      >
                        <input
                          type={runMode === "single" ? "radio" : "checkbox"}
                          name="demo-model"
                          value={model.model_id}
                          checked={checked}
                          data-testid={`demo-model-${model.arm_id}`}
                          onChange={() => toggleModel(model.model_id)}
                          className="mt-1"
                        />
                        <span className="min-w-0">
                          <span className="font-serif text-base">
                            {model.arm_id === "student_base"
                              ? "Current base model"
                              : model.arm_id === "promoted_winner"
                                ? "TinyFable Generalist"
                                : model.display_name}
                          </span>
                          <span className="mt-0.5 block text-xs text-muted-foreground">
                            {model.purpose}
                          </span>
                        </span>
                        <StatusBadge tone={armTone(model.arm_id)}>
                          {armBadgeLabel(model.arm_id)}
                        </StatusBadge>
                      </label>
                    </li>
                  );
                })}
              </ul>
            )}
          </fieldset>

          {statsModel ? (
            <Sheet>
              <SheetTrigger asChild>
                <Button
                  type="button"
                  variant="outline"
                  className="h-11 w-fit rounded-full"
                  data-testid="demo-metrics-drawer-trigger"
                >
                  <LayoutListIcon className="size-4" />
                  Open the full model record
                </Button>
              </SheetTrigger>
              <SheetContent
                side="right"
                className="w-full overflow-y-auto sm:max-w-2xl"
                data-testid="demo-metrics-drawer"
              >
                <SheetHeader>
                  <SheetTitle className="font-serif">
                    Full model record
                  </SheetTitle>
                  <SheetDescription>
                    This record contains model IDs, training details, fingerprints,
                    scores, and saved check status. Unknown values stay unknown.
                  </SheetDescription>
                </SheetHeader>
                <div className="grid gap-3 px-4 pb-6">
                  <Label htmlFor="demo-stats-model">Model to inspect</Label>
                  <p className="text-sm text-muted-foreground">
                    This changes the record shown below. It does not run the model.
                    The first selected model is the default.
                  </p>
                  <Select
                    value={statsModel.model_id}
                    onValueChange={setSelectedStatsModelId}
                  >
                    <SelectTrigger
                      id="demo-stats-model"
                      data-testid="demo-stats-model"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {registry.models.map((model) => (
                        <SelectItem key={model.model_id} value={model.model_id}>
                          {model.display_name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <DemoStatsTable model={statsModel} />
                </div>
              </SheetContent>
            </Sheet>
          ) : null}
        </PanelBody>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel>
          <PanelHeader>
            <PanelTitle className="text-xl">Example input</PanelTitle>
            <PanelDescription>
              Saved example <code>{example.example_id}</code>. Difficulty:{" "}
              {example.difficulty}.
            </PanelDescription>
          </PanelHeader>
          <PanelBody className="grid gap-3">
            <p className="text-sm text-muted-foreground">
              Editing this changes the example sent to the selected output source.
              Reset restores the saved input. The default is safe made-up data.
            </p>
            <Suggestions className="pb-1">
              {examples.map((item) => (
                <Suggestion
                  key={item.example_id}
                  suggestion={item.example_id}
                  onClick={() => {
                    setExampleId(item.example_id);
                    setResults([]);
                  }}
                  variant={item.example_id === exampleId ? "default" : "outline"}
                >
                  {item.label}
                </Suggestion>
              ))}
            </Suggestions>
            <PromptInput
              className="rounded-[14px] border-border bg-card"
              onSubmit={(_message, event) => {
                event.preventDefault();
                runInference();
              }}
            >
              <PromptInputBody>
                <PromptInputTextarea
                  id="demo-input"
                  aria-label="Example input as JSON"
                  data-testid="demo-input"
                  spellCheck={false}
                  value={inputText}
                  onChange={(event) => setInputText(event.target.value)}
                  className="min-h-56 font-mono text-[13px] leading-relaxed"
                />
              </PromptInputBody>
              <PromptInputFooter className="flex-wrap gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className="min-h-11 rounded-full"
                  data-testid="demo-reset-example"
                  onClick={resetExample}
                >
                  Reset input
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  className="min-h-11 rounded-full"
                  data-testid="demo-random-example"
                  onClick={randomExample}
                >
                  <ShuffleIcon className="size-4" />
                  Pick another
                </Button>
                <PromptInputSubmit
                  data-testid="demo-run"
                  disabled={isPending || !canRun}
                  status={isPending ? "submitted" : undefined}
                  className="ml-auto min-h-11 rounded-full px-4"
                  onClick={(event) => {
                    event.preventDefault();
                    runInference();
                  }}
                >
                  {isPending
                    ? "Running..."
                    : runMode === "compare"
                      ? "Compare models"
                      : "Run model"}
                </PromptInputSubmit>
              </PromptInputFooter>
            </PromptInput>
            {inputError ? (
              <Alert
                role="alert"
                data-testid="demo-input-error"
                className="border-[color-mix(in_oklab,var(--fail)_30%,transparent)] bg-[color-mix(in_oklab,var(--fail)_8%,transparent)]"
              >
                <AlertDescription>{inputError}</AlertDescription>
              </Alert>
            ) : null}
            <details>
              <summary className="min-h-11 cursor-pointer py-3 font-medium">
                Advanced input format
              </summary>
              <div>
                <h2 className="font-serif text-base">
                  Expected result shape (schema)
                </h2>
                <p className="mt-1 text-sm" data-testid="demo-expected-schema">
                  <code>{example.expected_output_schema.schema_version}</code>.{" "}
                  {example.expected_output_schema.description}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Required fields:{" "}
                  {example.expected_output_schema.required_fields.join(", ")}
                </p>
              </div>
            </details>
          </PanelBody>
        </Panel>

        <Panel data-testid="demo-stats-panel">
          <PanelHeader>
            <PanelTitle className="text-xl">Model facts</PanelTitle>
            <PanelDescription>
              A short summary for the selected model. Open the full record for IDs,
              fingerprints, and saved evidence.
            </PanelDescription>
          </PanelHeader>
          <PanelBody>
            {statsModel ? (
              <div className="grid gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge tone={armTone(statsModel.arm_id)}>
                    {armBadgeLabel(statsModel.arm_id)}
                  </StatusBadge>
                  <StatusBadge
                    tone={
                      statsModel.serving.availability === "live"
                        ? "pass"
                        : statsModel.serving.availability === "fixture_preview"
                          ? "precomputed"
                          : "unavailable"
                    }
                  >
                    {statsModel.serving.availability === "live"
                      ? "Live"
                      : statsModel.serving.availability === "fixture_preview"
                        ? "Saved demo"
                        : "Not available"}
                  </StatusBadge>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Fact
                    label="Compression"
                    value={formatRatio(statsModel.stats.compression_ratio)}
                  />
                  <Fact
                    label="Main score"
                    value={formatIndex(statsModel.stats.iid_primary_index)}
                  />
                  <Fact
                    label="Unfamiliar examples"
                    value={formatIndex(statsModel.stats.ood_retention)}
                  />
                  <Fact
                    label="Result"
                    value={formatUnknown(statsModel.stats.proof_status)}
                  />
                </div>
              </div>
            ) : (
              <p data-testid="demo-stats-empty">
                There is no model record for this run.
              </p>
            )}
          </PanelBody>
        </Panel>
      </div>

      <Panel>
        <PanelHeader>
          <PanelTitle className="text-xl">Results</PanelTitle>
          <PanelDescription>
            The decision comes first. Technical output stays under Advanced.
          </PanelDescription>
        </PanelHeader>
        <PanelBody>
          {runError ? (
            <Alert
              role="alert"
              data-testid="demo-run-error"
              className="mb-3 border-[color-mix(in_oklab,var(--fail)_30%,transparent)] bg-[color-mix(in_oklab,var(--fail)_8%,transparent)]"
            >
              <AlertDescription>{runError}</AlertDescription>
            </Alert>
          ) : null}
          {results.length > 0 ? (
            <ComparisonSummary results={results} models={registry.models} />
          ) : null}
          <Conversation className="relative min-h-48 rounded-[14px] border border-border bg-card/50">
            <ConversationContent>
              {isPending ? (
                <div
                  className="flex items-center gap-2 p-4 text-sm"
                  data-testid="demo-results-loading"
                  role="status"
                >
                  <Loader size={18} />
                  Getting the selected outputs...
                </div>
              ) : null}
              {!isPending && results.length === 0 ? (
                <ConversationEmptyState
                  data-testid="demo-results-empty"
                  title="No result yet"
                  description="Choose models, then run the saved demo."
                />
              ) : null}
              <div
                className={cn(
                  "grid gap-4 p-3",
                  results.length > 1 ? "lg:grid-cols-2" : "grid-cols-1",
                )}
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
            </ConversationContent>
          </Conversation>
        </PanelBody>
      </Panel>
    </section>
  );
}

function DemoSetting({
  label,
  helper,
  children,
}: {
  label: string;
  helper: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-2">
      <Label className="font-serif text-lg font-normal">{label}</Label>
      <p className="text-sm text-muted-foreground">{helper}</p>
      {children}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[14px] border border-border bg-card px-3 py-2">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 font-serif text-xl">{value}</p>
    </div>
  );
}

function ComparisonSummary({
  results,
  models,
}: {
  results: readonly DemoInferenceResponse[];
  models: readonly DemoModelEntry[];
}) {
  const completed = results.filter(
    (result): result is Extract<DemoInferenceResponse, { status: "ok" }> =>
      result.status === "ok",
  );
  if (completed.length === 0) return null;

  const ranked = [...completed].sort((left, right) => {
    const scoreDifference = (right.score ?? -1) - (left.score ?? -1);
    if (scoreDifference !== 0) return scoreDifference;
    return (left.latency_ms ?? Number.MAX_SAFE_INTEGER) -
      (right.latency_ms ?? Number.MAX_SAFE_INTEGER);
  });
  const best = ranked[0]!;
  const model = models.find((item) => item.model_id === best.model_id);
  const name =
    model?.arm_id === "student_base"
      ? "the current base model"
      : model?.arm_id === "promoted_winner"
        ? "TinyFable Generalist"
        : model?.display_name ?? "the leading model";
  const equalQuality = completed.every(
    (result) => result.score === completed[0]?.score,
  );
  const equalScore = completed[0]?.score;

  return (
    <div
      className="mb-4 rounded-[14px] border border-[color-mix(in_oklab,var(--pass)_35%,var(--border))] bg-[color-mix(in_oklab,var(--pass)_6%,var(--card))] p-4"
      data-testid="demo-decision"
    >
      <p className="text-kicker text-[var(--pass)]">Decision</p>
      <h2 className="mt-1 font-serif text-2xl">
        {equalQuality
          ? equalScore === 1
            ? "Take both models to the final checks"
            : "Keep both models out of rollout for now"
          : `Take ${name} to the final checks`}
      </h2>
      <p className="mt-2 text-sm">
        {equalQuality
          ? equalScore === 1
            ? "The models matched the known answer equally well in this example. Compare more examples before choosing."
            : "Neither model matched the known answer in this example. Fix the result before choosing."
          : `${name} matched the known answer more closely in this example.`}
      </p>
      <p className="mt-1 text-sm text-muted-foreground">
        Confidence is low. This is one saved example, not a live run. Cost is not
        measured here.
      </p>
    </div>
  );
}

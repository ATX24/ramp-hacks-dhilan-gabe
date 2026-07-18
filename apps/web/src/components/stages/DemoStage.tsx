"use client";

import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { StatusBadge } from "@/components/StatusBadge";
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
import type { StageBundle } from "@/lib/types";

function unknownOr(value: string): string {
  return value;
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

  // Sync editable input when the example changes from URL/controls.
  useEffect(() => {
    const next = getDemoExample(exampleId) ?? defaultExampleForTask(task);
    setInputText(JSON.stringify(next.input, null, 2));
    setInputError(null);
  }, [exampleId, task]);

  // Keep shareable URL state in sync without replace loops.
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

  return (
    <section aria-labelledby="demo-heading" className="demo-stage">
      <div className="panel demo-hero">
        <div className="demo-hero-copy">
          <h2 id="demo-heading">Demo</h2>
          <p className="demo-kicker">Playground</p>
          <p>
            Interactive playground for TinyFable post-trained arms on finance
            tasks with prepopulated held-out examples. Fixture previews are
            labeled. Live inference only runs through the typed serving gateway.
          </p>
        </div>
        <aside className="demo-walkthrough" data-testid="demo-walkthrough">
          <h3>Judge walkthrough</h3>
          <ol>
            <li>Start on transaction review with base vs sequence (or winner).</li>
            <li>Inspect stats: compression, hashes, IID/OOD, proof/promotion.</li>
            <li>Switch task → variance, then cash reconciliation.</li>
            <li>Toggle compare mode; never treat fixture preview as live.</li>
          </ol>
        </aside>
      </div>

      <div className="panel">
        <div className="meta-row">
          <span>
            Registry run <code>{registry.run_id}</code>
          </span>
          <span>
            Models <code>{registry.models.length}</code>
          </span>
          <StatusBadge tone={liveBaseUrl ? "pass" : "unavailable"}>
            {liveBaseUrl ? "Live gateway configured" : "Live gateway unavailable"}
          </StatusBadge>
          <StatusBadge tone="precomputed">Fixture preview available</StatusBadge>
        </div>

        <div className="demo-controls" role="group" aria-label="Demo controls">
          <div className="demo-control-block">
            <label htmlFor="demo-task">Finance task</label>
            <select
              id="demo-task"
              data-testid="demo-task-select"
              value={task}
              onChange={(event) => applyTask(event.target.value as FinanceTaskId)}
            >
              {FINANCE_TASKS.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          <div className="demo-control-block">
            <label htmlFor="demo-example">Example</label>
            <select
              id="demo-example"
              data-testid="demo-example-select"
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

          <div className="demo-control-block">
            <span id="demo-run-mode-label">Run mode</span>
            <div
              className="controls"
              role="radiogroup"
              aria-labelledby="demo-run-mode-label"
            >
              <button
                type="button"
                className={`btn ${runMode === "single" ? "btn-primary" : ""}`}
                aria-pressed={runMode === "single"}
                data-testid="demo-run-mode-single"
                onClick={() => {
                  setRunMode("single");
                  setModelIds((current) => current.slice(0, 1));
                }}
              >
                One model
              </button>
              <button
                type="button"
                className={`btn ${runMode === "compare" ? "btn-primary" : ""}`}
                aria-pressed={runMode === "compare"}
                data-testid="demo-run-mode-compare"
                onClick={() => setRunMode("compare")}
              >
                Compare 2+
              </button>
            </div>
          </div>

          <div className="demo-control-block">
            <span id="demo-infer-mode-label">Inference</span>
            <div
              className="controls"
              role="radiogroup"
              aria-labelledby="demo-infer-mode-label"
            >
              <button
                type="button"
                className={`btn ${inferenceMode === "fixture_preview" ? "btn-primary" : ""}`}
                aria-pressed={inferenceMode === "fixture_preview"}
                data-testid="demo-infer-fixture"
                onClick={() => setInferenceMode("fixture_preview")}
              >
                Fixture preview
              </button>
              <button
                type="button"
                className={`btn ${inferenceMode === "live" ? "btn-primary" : ""}`}
                aria-pressed={inferenceMode === "live"}
                data-testid="demo-infer-live"
                onClick={() => setInferenceMode("live")}
              >
                Live
              </button>
            </div>
          </div>
        </div>

        <fieldset className="demo-model-fieldset">
          <legend>Models from registry</legend>
          {registry.models.length === 0 ? (
            <p data-testid="demo-models-empty">No models are registered for this run.</p>
          ) : (
            <ul className="demo-model-list" data-testid="demo-model-list">
              {registry.models.map((model) => {
                const checked = modelIds.includes(model.model_id);
                return (
                  <li key={model.model_id}>
                    <label className="demo-model-option">
                      <input
                        type={runMode === "single" ? "radio" : "checkbox"}
                        name="demo-model"
                        value={model.model_id}
                        checked={checked}
                        data-testid={`demo-model-${model.arm_id}`}
                        onChange={() => toggleModel(model.model_id)}
                      />
                      <span>
                        <strong>{model.display_name}</strong>
                        <span className="demo-model-meta">
                          <code>{model.model_id}</code> · {model.serving.availability}
                          {model.excluded ? " · excluded" : ""}
                        </span>
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </fieldset>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>Input</h3>
          <p>
            Example <code>{example.example_id}</code> · split{" "}
            <code>{example.split}</code> · {example.difficulty}
          </p>
          <label className="sr-only" htmlFor="demo-input">
            Example input JSON
          </label>
          <textarea
            id="demo-input"
            className="demo-textarea"
            data-testid="demo-input"
            spellCheck={false}
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
          />
          {inputError ? (
            <p className="banner banner-error" role="alert" data-testid="demo-input-error">
              {inputError}
            </p>
          ) : null}
          <div className="controls" style={{ marginTop: "0.85rem" }}>
            <button
              type="button"
              className="btn"
              data-testid="demo-reset-example"
              onClick={resetExample}
            >
              Reset example
            </button>
            <button
              type="button"
              className="btn"
              data-testid="demo-random-example"
              onClick={randomExample}
            >
              Random example
            </button>
            <button
              type="button"
              className="btn btn-primary"
              data-testid="demo-run"
              disabled={isPending}
              onClick={runInference}
            >
              {isPending ? "Running…" : runMode === "compare" ? "Compare models" : "Run model"}
            </button>
          </div>
          <h4>Expected output schema</h4>
          <p data-testid="demo-expected-schema">
            <code>{example.expected_output_schema.schema_version}</code> —{" "}
            {example.expected_output_schema.description}
          </p>
          <p className="demo-schema-fields">
            Required: {example.expected_output_schema.required_fields.join(", ")}
          </p>
        </div>

        <div className="panel" data-testid="demo-stats-panel">
          <h3>Model statistics</h3>
          {statsModel ? (
            <>
              <label htmlFor="demo-stats-model">Inspect model</label>
              <select
                id="demo-stats-model"
                data-testid="demo-stats-model"
                value={statsModel.model_id}
                onChange={(event) => setSelectedStatsModelId(event.target.value)}
              >
                {registry.models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.display_name}
                  </option>
                ))}
              </select>
              <DemoStatsTable model={statsModel} />
            </>
          ) : (
            <p data-testid="demo-stats-empty">No model available for statistics.</p>
          )}
        </div>
      </div>

      <div className="panel">
        <h3>Results</h3>
        {runError ? (
          <p className="banner banner-error" role="alert" data-testid="demo-run-error">
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
          className={results.length > 1 ? "demo-results-compare" : "demo-results-single"}
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
      </div>
    </section>
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
    <div className="table-wrap" style={{ marginTop: "0.75rem" }}>
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
        className="demo-result-card"
        data-testid={`demo-result-${result.model_id}`}
        data-status={result.status}
      >
        <header className="meta-row">
          <strong>{model?.display_name ?? result.model_id}</strong>
          <StatusBadge tone="unavailable">{result.status}</StatusBadge>
          <code>{result.code}</code>
        </header>
        <p role="alert">{result.message}</p>
      </article>
    );
  }

  return (
    <article
      className="demo-result-card"
      data-testid={`demo-result-${result.model_id}`}
      data-status="ok"
      data-provenance={result.provenance}
    >
      <header className="meta-row">
        <strong>{model?.display_name ?? result.model_id}</strong>
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
      <ul className="list-plain">
        <li>
          Latency:{" "}
          {result.latency_ms === null ? "unknown" : `${result.latency_ms} ms`}
        </li>
        <li>
          Tokens: prompt{" "}
          {result.prompt_tokens === null ? "unknown" : result.prompt_tokens}, completion{" "}
          {result.completion_tokens === null ? "unknown" : result.completion_tokens}
        </li>
        <li>
          Score:{" "}
          {result.score === null ? "unknown" : result.score.toFixed(3)}
          {result.score_detail ? ` — ${result.score_detail}` : ""}
        </li>
        {result.validation_detail ? <li>{result.validation_detail}</li> : null}
      </ul>
      <h4>Structured output</h4>
      <pre className="demo-json" data-testid={`demo-structured-${result.model_id}`}>
        {JSON.stringify(result.structured_output, null, 2)}
      </pre>
      <h4>Raw JSON</h4>
      <pre className="demo-json" data-testid={`demo-raw-${result.model_id}`}>
        {result.raw_json}
      </pre>
    </article>
  );
}

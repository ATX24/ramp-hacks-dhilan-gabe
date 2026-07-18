import { defaultExampleForTask, getDemoExample } from "@/lib/demo/examples";
import type {
  DemoInferenceMode,
  DemoRunMode,
  DemoUrlState,
  FinanceTaskId,
} from "@/lib/demo/types";
import { FINANCE_TASKS } from "@/lib/demo/types";

const TASK_IDS = new Set<string>(FINANCE_TASKS.map((task) => task.id));

function singleParam(
  params: URLSearchParams | Record<string, string | string[] | undefined>,
  key: string,
): string | undefined {
  if (params instanceof URLSearchParams) {
    const value = params.get(key);
    return value === null ? undefined : value;
  }
  const raw = params[key];
  return typeof raw === "string" ? raw : undefined;
}

function parseTask(value: string | undefined): FinanceTaskId {
  if (value && TASK_IDS.has(value)) return value as FinanceTaskId;
  return "transaction_review";
}

function parseModelIds(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function parseRunMode(value: string | undefined, modelCount: number): DemoRunMode {
  if (value === "compare") return "compare";
  if (value === "single") return "single";
  return modelCount >= 2 ? "compare" : "single";
}

function parseInferenceMode(value: string | undefined): DemoInferenceMode {
  return value === "live" ? "live" : "fixture_preview";
}

export function parseDemoUrlState(
  params: URLSearchParams | Record<string, string | string[] | undefined>,
  fallbackModelIds: string[],
): DemoUrlState {
  const task = parseTask(singleParam(params, "task"));
  const requestedModels = parseModelIds(singleParam(params, "models"));
  const modelIds = requestedModels.length > 0 ? requestedModels : fallbackModelIds;
  const runMode = parseRunMode(singleParam(params, "runMode"), modelIds.length);
  const inferenceMode = parseInferenceMode(singleParam(params, "infer"));
  const exampleParam = singleParam(params, "example");
  const example =
    exampleParam && getDemoExample(exampleParam)?.task === task
      ? exampleParam
      : defaultExampleForTask(task).example_id;

  const normalizedModels =
    runMode === "single" ? modelIds.slice(0, 1) : modelIds.slice(0, 4);

  return {
    task,
    modelIds: normalizedModels.length > 0 ? normalizedModels : fallbackModelIds.slice(0, 1),
    exampleId: example,
    runMode: runMode === "compare" && normalizedModels.length < 2 ? "single" : runMode,
    inferenceMode,
  };
}

export function serializeDemoUrlState(
  state: DemoUrlState,
  base?: {
    mode?: string;
    runId?: string;
  },
): URLSearchParams {
  const params = new URLSearchParams();
  if (base?.mode) params.set("mode", base.mode);
  if (base?.runId) params.set("run", base.runId);
  params.set("task", state.task);
  params.set("models", state.modelIds.join(","));
  params.set("example", state.exampleId);
  params.set("runMode", state.runMode);
  params.set("infer", state.inferenceMode);
  return params;
}

export function buildDemoHref(
  state: DemoUrlState,
  base?: { mode?: string; runId?: string },
): string {
  return `/demo?${serializeDemoUrlState(state, base).toString()}`;
}

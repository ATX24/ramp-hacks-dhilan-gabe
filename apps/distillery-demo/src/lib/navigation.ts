import { isResourceId } from "@/lib/ids";
import { isUiMode, parseUiMode } from "@/lib/modes";
import { STAGE_PLAIN } from "@/lib/plainLanguage";
import type { StageId, UiMode } from "@/lib/types";

export type SearchParams = Record<string, string | string[] | undefined>;

export type RunSelection =
  | { kind: "absent" }
  | { kind: "valid"; runId: string }
  | { kind: "invalid"; rawValue: string | string[] };

export const STAGES = [
  {
    id: "curate",
    href: "/curate",
    index: "01",
    name: "Curate",
    plain: STAGE_PLAIN.curate.plain,
  },
  {
    id: "synthesize",
    href: "/synthesize",
    index: "02",
    name: "Synthesize",
    plain: STAGE_PLAIN.synthesize.plain,
  },
  {
    id: "train",
    href: "/train",
    index: "03",
    name: "Train",
    plain: STAGE_PLAIN.train.plain,
  },
  {
    id: "prove",
    href: "/prove",
    index: "04",
    name: "Prove",
    plain: STAGE_PLAIN.prove.plain,
  },
  {
    id: "demo",
    href: "/demo",
    index: "05",
    name: "Demo",
    plain: STAGE_PLAIN.demo.plain,
  },
] as const satisfies ReadonlyArray<{
  id: StageId;
  href: `/${StageId}`;
  index: string;
  name: string;
  plain: string;
}>;

export type StageRoute = (typeof STAGES)[number]["href"];

export const STAGE_ROUTES: readonly StageRoute[] = STAGES.map((stage) => stage.href);

function singleModeParam(searchParams: SearchParams | undefined): string | undefined {
  const raw = searchParams?.mode;
  return typeof raw === "string" ? raw : undefined;
}

export function parseRunSelection(searchParams: SearchParams | undefined): RunSelection {
  const raw = searchParams?.run;
  if (raw === undefined) return { kind: "absent" };
  if (typeof raw === "string" && isResourceId("run", raw)) {
    return { kind: "valid", runId: raw };
  }
  return { kind: "invalid", rawValue: raw };
}

export function resolveModeFromSearch(searchParams: SearchParams | undefined): UiMode {
  return parseUiMode(singleModeParam(searchParams));
}

export function buildStageHref(
  route: StageRoute,
  mode: UiMode,
  runId?: string,
): string {
  const params = new URLSearchParams({ mode });
  if (runId !== undefined) {
    if (!isResourceId("run", runId)) {
      throw new Error("Stage navigation received an invalid run ID");
    }
    params.set("run", runId);
  }
  return `${route}?${params.toString()}`;
}

export function buildCentralHref(
  stage: "train" | "demo",
  searchParams: SearchParams | undefined,
): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(searchParams ?? {})) {
    if (typeof value === "string") {
      params.set(key, value);
    } else if (Array.isArray(value)) {
      for (const item of value) params.append(key, item);
    }
  }
  params.set("stage", stage);
  return `/?${params.toString()}`;
}

/** Playground-first entry: land judges on Demo, then reveal the journey. */
export function buildRootRedirect(searchParams: SearchParams | undefined): string {
  const rawMode = singleModeParam(searchParams);
  const runSelection = parseRunSelection(searchParams);
  if (rawMode !== undefined && !isUiMode(rawMode)) return "/demo";
  const mode = isUiMode(rawMode) ? rawMode : "default";
  if (runSelection.kind === "invalid") {
    return rawMode === undefined ? "/demo" : buildStageHref("/demo", mode);
  }
  if (runSelection.kind === "valid") {
    return buildStageHref("/demo", mode, runSelection.runId);
  }
  return rawMode === undefined ? "/demo" : buildStageHref("/demo", mode);
}

export function isStageRoute(pathname: string): pathname is StageRoute {
  return (STAGE_ROUTES as readonly string[]).includes(pathname);
}

export function buildModeHref(pathname: string, mode: UiMode): string {
  return buildStageHref(isStageRoute(pathname) ? pathname : "/demo", mode);
}

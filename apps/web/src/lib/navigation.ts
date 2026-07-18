import { isResourceId } from "@/lib/ids";
import { isUiMode, parseUiMode } from "@/lib/modes";
import type { StageId, UiMode } from "@/lib/types";

export type SearchParams = Record<string, string | string[] | undefined>;

export type RunSelection =
  | { kind: "absent" }
  | { kind: "valid"; runId: string }
  | { kind: "invalid"; rawValue: string | string[] };

export const STAGES = [
  { id: "curate", href: "/curate", index: "01", name: "Curate" },
  { id: "synthesize", href: "/synthesize", index: "02", name: "Synthesize" },
  { id: "train", href: "/train", index: "03", name: "Train" },
  { id: "prove", href: "/prove", index: "04", name: "Prove" },
  { id: "demo", href: "/demo", index: "05", name: "Demo" },
] as const satisfies ReadonlyArray<{
  id: StageId;
  href: `/${StageId}`;
  index: string;
  name: string;
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

export function buildRootRedirect(searchParams: SearchParams | undefined): string {
  const rawMode = singleModeParam(searchParams);
  const runSelection = parseRunSelection(searchParams);
  if (rawMode !== undefined && !isUiMode(rawMode)) return "/curate";
  const mode = isUiMode(rawMode) ? rawMode : "default";
  if (runSelection.kind === "invalid") {
    return rawMode === undefined ? "/curate" : buildStageHref("/curate", mode);
  }
  if (runSelection.kind === "valid") {
    return buildStageHref("/curate", mode, runSelection.runId);
  }
  return rawMode === undefined ? "/curate" : buildStageHref("/curate", mode);
}

export function isStageRoute(pathname: string): pathname is StageRoute {
  return (STAGE_ROUTES as readonly string[]).includes(pathname);
}

export function buildModeHref(pathname: string, mode: UiMode): string {
  return buildStageHref(isStageRoute(pathname) ? pathname : "/curate", mode);
}

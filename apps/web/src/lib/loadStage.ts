import { createApiClient } from "@/lib/api";
import { isFixtureClientError, fixtureClientError } from "@/lib/fixtureErrors";
import { buildStageBundle } from "@/lib/fixtures/bundle";
import { getDefaultRunId } from "@/lib/fixtures/catalog";
import {
  parseRunSelection,
  resolveModeFromSearch,
  type RunSelection,
  type SearchParams,
} from "@/lib/navigation";
import type { StageBundle, UiMode } from "@/lib/types";

export { resolveModeFromSearch } from "@/lib/navigation";

export interface StageLoadRequest {
  bundle: StageBundle;
  runSelection: RunSelection;
}

export function withStageLoadFailure(
  mode: UiMode,
  runId: string,
  error: unknown,
): StageBundle {
  const bundle = buildStageBundle(mode, runId);
  const title = isFixtureClientError(error)
    ? error.payload.code
    : "FIXTURE_LOAD_FAILED";
  const message =
    error instanceof Error ? error.message : "The saved sample could not be opened.";
  return {
    ...bundle,
    load_state: {
      status: "failed",
      title,
      message,
      retryable: false,
    },
  };
}

export async function loadStageRequest(
  searchParams: SearchParams | undefined,
): Promise<StageLoadRequest> {
  const mode = resolveModeFromSearch(searchParams);
  const runSelection = parseRunSelection(searchParams);
  const fallbackRunId =
    runSelection.kind === "valid" ? runSelection.runId : getDefaultRunId(mode);

  if (runSelection.kind === "invalid") {
    const error = fixtureClientError(
      "INVALID_RESOURCE_ID",
      "The run ID in this link is not valid.",
      "run",
      Array.isArray(runSelection.rawValue)
        ? runSelection.rawValue.join(",")
        : runSelection.rawValue,
    );
    return {
      bundle: withStageLoadFailure(mode, fallbackRunId, error),
      runSelection,
    };
  }

  try {
    const client = createApiClient({
      mode,
      runId: runSelection.kind === "valid" ? runSelection.runId : undefined,
    });
    return {
      bundle: await client.loadStage(),
      runSelection,
    };
  } catch (error) {
    return {
      bundle: withStageLoadFailure(mode, fallbackRunId, error),
      runSelection,
    };
  }
}

export async function loadStageBundle(
  searchParams: SearchParams | undefined,
): Promise<StageBundle> {
  return (await loadStageRequest(searchParams)).bundle;
}

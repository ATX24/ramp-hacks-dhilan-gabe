/**
 * Fixture-backed Distillery API client.
 * Never calls live network endpoints.
 */

import {
  buildStageBundle,
} from "@/lib/fixtures/bundle";
import {
  DEFAULT_DATASET_ID,
  getDefaultRunId,
  modeForArtifactId,
  modeForReportId,
  modesForRunId,
} from "@/lib/fixtures/catalog";
import { fixtureClientError } from "@/lib/fixtureErrors";
import { assertResourceId } from "@/lib/ids";
import { parseUiMode } from "@/lib/modes";
import { isRunCancellable } from "@/lib/runPresentation";
import type {
  Dataset,
  DistillationPlan,
  DistillationRunView,
  ModelArtifactMeta,
  ProofReportView,
  StageBundle,
  SynthesisSummary,
  UiMode,
  FixtureResourceKind,
} from "@/lib/types";

export type ApiClientOptions = {
  mode?: UiMode;
  runId?: string;
};

function resolveMode(explicit?: UiMode): UiMode {
  if (explicit) return explicit;
  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = parseUiMode(params.get("mode"));
    if (params.has("mode")) return fromQuery;
  }
  return "default";
}

function resolveRunId(mode: UiMode, explicit?: string): string {
  if (explicit !== undefined) return explicit;
  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = params.get("run");
    if (fromQuery !== null) return fromQuery;
  }
  return getDefaultRunId(mode);
}

function notFound(kind: FixtureResourceKind, resourceId: string): never {
  throw fixtureClientError(
    "RESOURCE_NOT_FOUND",
    `${kind} fixture was not found.`,
    kind,
    resourceId,
  );
}

function mismatch(
  kind: FixtureResourceKind,
  resourceId: string,
  mode: UiMode,
  ownerModes: readonly UiMode[],
): never {
  throw fixtureClientError(
    "RESOURCE_MISMATCH",
    `${kind} fixture does not belong to mode ${mode}.`,
    kind,
    resourceId,
    { requested_mode: mode, fixture_modes: ownerModes },
  );
}

export class DistilleryApiClient {
  private readonly mode: UiMode;
  private readonly runId: string;
  private cancelRequested = false;

  constructor(options: ApiClientOptions = {}) {
    this.mode = resolveMode(options.mode);
    this.runId = resolveRunId(this.mode, options.runId);
  }

  getMode(): UiMode {
    return this.mode;
  }

  private bundleForRun(runId: string): StageBundle {
    assertResourceId("run", runId);
    const ownerModes = modesForRunId(runId);
    if (ownerModes.length === 0) notFound("run", runId);
    if (!ownerModes.includes(this.mode)) {
      mismatch("run", runId, this.mode, ownerModes);
    }
    return buildStageBundle(this.mode, runId);
  }

  getBundle(): StageBundle {
    const bundle = this.bundleForRun(this.runId);
    if (this.cancelRequested) {
      return {
        ...bundle,
        run: {
          ...bundle.run,
          cancel_requested: true,
          state: bundle.run.training_launched ? "CANCELLED" : bundle.run.state,
        },
      };
    }
    return bundle;
  }

  async getDataset(datasetId: string = DEFAULT_DATASET_ID): Promise<Dataset> {
    assertResourceId("dataset", datasetId);
    if (datasetId !== DEFAULT_DATASET_ID) notFound("dataset", datasetId);
    return this.getBundle().dataset;
  }

  async getSynthesis(runId: string = this.runId): Promise<SynthesisSummary> {
    return this.bundleForRun(runId).synthesis;
  }

  async planDistillation(runId: string = this.runId): Promise<DistillationPlan> {
    // Pure planning: never mutates launch state.
    return this.bundleForRun(runId).plan;
  }

  async getRun(runId: string = this.runId): Promise<DistillationRunView> {
    return this.bundleForRun(runId).run;
  }

  async cancelRun(runId: string = this.runId): Promise<DistillationRunView> {
    const bundle = this.bundleForRun(runId);
    if (!isRunCancellable(bundle.run, bundle.plan)) {
      return bundle.run;
    }
    this.cancelRequested = true;
    const run = this.bundleForRun(runId).run;
    return {
      ...run,
      cancel_requested: true,
    };
  }

  async getArtifact(artifactId: string): Promise<ModelArtifactMeta> {
    assertResourceId("artifact", artifactId);
    const ownerMode = modeForArtifactId(artifactId);
    if (!ownerMode) notFound("artifact", artifactId);
    if (ownerMode !== this.mode) {
      mismatch("artifact", artifactId, this.mode, [ownerMode]);
    }
    const artifact = this.getBundle().artifact;
    if (!artifact || artifact.artifact_id !== artifactId) {
      mismatch("artifact", artifactId, this.mode, [ownerMode]);
    }
    return artifact;
  }

  async getProofReport(reportId: string): Promise<ProofReportView> {
    assertResourceId("report", reportId);
    const ownerMode = modeForReportId(reportId);
    if (!ownerMode) notFound("report", reportId);
    if (ownerMode !== this.mode) {
      mismatch("report", reportId, this.mode, [ownerMode]);
    }
    const proof = this.getBundle().proof;
    if (!proof || proof.report_id !== reportId) {
      mismatch("report", reportId, this.mode, [ownerMode]);
    }
    return proof;
  }

  /** Convenience loader for stage pages. */
  async loadStage(): Promise<StageBundle> {
    return this.getBundle();
  }
}

export function createApiClient(options?: ApiClientOptions): DistilleryApiClient {
  return new DistilleryApiClient(options);
}

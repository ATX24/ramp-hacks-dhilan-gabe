import { describe, expect, it } from "vitest";
import { createApiClient } from "@/lib/api";
import { buildStageBundle, parseUiMode } from "@/lib/fixtures/bundle";
import {
  DEFAULT_DATASET_ID,
  FIXTURE_MODE_CATALOG,
  RESTORED_RUN_ID,
} from "@/lib/fixtures/catalog";
import { loadStageRequest } from "@/lib/loadStage";

describe("fixture API client", () => {
  it("never marks training as launched in default mode", async () => {
    const client = createApiClient({ mode: "default" });
    const plan = await client.planDistillation();
    const run = await client.getRun();
    expect(plan.training_launched).toBe(false);
    expect(plan.planned_job.launched).toBe(false);
    expect(run.training_launched).toBe(false);
  });

  it("returns skipped synthesis with explicit reason", async () => {
    const client = createApiClient({ mode: "skipped_synthesis" });
    const synthesis = await client.getSynthesis();
    expect(synthesis.skipped).toBe(true);
    expect(synthesis.skip_reason).toBe("responses_already_present");
  });

  it("surfaces RECIPE_NOT_IMPLEMENTED in unavailable mode", async () => {
    const client = createApiClient({ mode: "unavailable" });
    const bundle = await client.loadStage();
    expect(bundle.error?.code).toBe("RECIPE_NOT_IMPLEMENTED");
    expect(bundle.plan.resolved_recipe).toBeNull();
  });

  it("labels precomputed proof economics as projected", async () => {
    const client = createApiClient({ mode: "precomputed" });
    const reportId = FIXTURE_MODE_CATALOG.precomputed.reportId;
    expect(reportId).not.toBeNull();
    if (!reportId) return;
    const proof = await client.getProofReport(reportId);
    expect(proof.precomputed).toBe(true);
    expect(proof.economics.serving_cost_projected).toBe(true);
  });

  it("refuses cancellation when no active started run exists", async () => {
    const client = createApiClient({ mode: "no_training_yet" });
    const updated = await client.cancelRun();
    expect(updated.cancel_requested).toBe(false);
    expect(updated.training_launched).toBe(false);
  });

  it("honors an explicit compatible run ID", async () => {
    const client = createApiClient({ mode: "default", runId: RESTORED_RUN_ID });
    expect((await client.getRun()).run_id).toBe(RESTORED_RUN_ID);
    expect((await client.planDistillation(RESTORED_RUN_ID)).run_id).toBe(
      RESTORED_RUN_ID,
    );
  });

  it("returns typed invalid, not-found, and mismatch errors", async () => {
    const client = createApiClient({ mode: "default" });
    await expect(client.getRun("bad")).rejects.toMatchObject({
      payload: { code: "INVALID_RESOURCE_ID" },
    });
    await expect(client.getRun("run_fixture_missing_999")).rejects.toMatchObject({
      payload: { code: "RESOURCE_NOT_FOUND" },
    });
    await expect(
      client.getRun(FIXTURE_MODE_CATALOG.proved.defaultRunId),
    ).rejects.toMatchObject({
      payload: { code: "RESOURCE_MISMATCH" },
    });
    await expect(client.getDataset("ds_missing_fixture_999")).rejects.toMatchObject({
      payload: { code: "RESOURCE_NOT_FOUND" },
    });
  });

  it("honors artifact and report IDs without substitution", async () => {
    const entry = FIXTURE_MODE_CATALOG.proved;
    expect(entry.artifactId).not.toBeNull();
    expect(entry.reportId).not.toBeNull();
    if (!entry.artifactId || !entry.reportId) return;

    const client = createApiClient({ mode: "proved" });
    expect((await client.getArtifact(entry.artifactId)).artifact_id).toBe(
      entry.artifactId,
    );
    expect((await client.getProofReport(entry.reportId)).report_id).toBe(
      entry.reportId,
    );
    const otherReportId = FIXTURE_MODE_CATALOG.failed_quality.reportId;
    expect(otherReportId).not.toBeNull();
    if (!otherReportId) return;
    await expect(client.getProofReport(otherReportId)).rejects.toMatchObject({
      payload: { code: "RESOURCE_MISMATCH" },
    });
    await expect(
      client.getArtifact("art_tinyfable_missing_999"),
    ).rejects.toMatchObject({
      payload: { code: "RESOURCE_NOT_FOUND" },
    });
  });

  it("returns the one known dataset only", async () => {
    const client = createApiClient({ mode: "default" });
    expect((await client.getDataset(DEFAULT_DATASET_ID)).dataset_id).toBe(
      DEFAULT_DATASET_ID,
    );
  });

  it("honors explicit run navigation and renders typed request failures", async () => {
    const explicit = await loadStageRequest({
      mode: "default",
      run: RESTORED_RUN_ID,
    });
    expect(explicit.bundle.run.run_id).toBe(RESTORED_RUN_ID);
    expect(explicit.runSelection.kind).toBe("valid");

    const mismatch = await loadStageRequest({
      mode: "default",
      run: FIXTURE_MODE_CATALOG.proved.defaultRunId,
    });
    expect(mismatch.bundle.load_state).toMatchObject({
      status: "failed",
      title: "RESOURCE_MISMATCH",
    });

    const malformed = await loadStageRequest({
      mode: "default",
      run: "../bad",
    });
    expect(malformed.bundle.load_state).toMatchObject({
      status: "failed",
      title: "INVALID_RESOURCE_ID",
    });
  });
});

describe("bundle modes", () => {
  it("parses known modes and defaults unknown values", () => {
    expect(parseUiMode("insufficient_evidence")).toBe("insufficient_evidence");
    expect(parseUiMode("not-a-mode")).toBe("default");
  });

  it("builds insufficient_evidence proof status", () => {
    const bundle = buildStageBundle("insufficient_evidence");
    expect(bundle.proof?.proof_status).toBe("insufficient_evidence");
    expect(bundle.proof?.first_failed_gate).toBe("evidence_gate");
    expect(bundle.artifact?.precomputed).toBe(true);
    expect(bundle.run.state).toBe("SUCCEEDED");
    expect(bundle.proof?.systems?.measurement_source).toBe(
      "precomputed_prior_run",
    );
  });

  it("fails leakage check in error mode", () => {
    const bundle = buildStageBundle("error");
    expect(bundle.error?.code).toBe("DATA_LEAKAGE_DETECTED");
    expect(bundle.dataset.leakage_checks.some((c) => !c.passed)).toBe(true);
  });

  it("provides explicit loading and request-failure states", () => {
    expect(buildStageBundle("loading").load_state.status).toBe("loading");
    expect(buildStageBundle("fetch_failure").load_state.status).toBe("failed");
  });
});

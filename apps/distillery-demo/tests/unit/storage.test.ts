import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clearRunReference,
  loadRunReference,
  persistRunReference,
} from "@/lib/storage";

describe("run reference persistence", () => {
  beforeEach(() => {
    clearRunReference();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("persists and reloads run reference across refresh simulation", () => {
    expect(
      persistRunReference("default", {
        mode: "default",
        runId: "run_fixture_tinyfable_001",
        datasetId: "ds_finance_world_v1_smoke",
        updatedAt: "2026-07-18T12:00:00.000Z",
      }),
    ).toBe(true);
    const loaded = loadRunReference("default");
    expect(loaded?.runId).toBe("run_fixture_tinyfable_001");
    expect(loaded?.datasetId).toBe("ds_finance_world_v1_smoke");
  });

  it("returns null for corrupt JSON and invalid shapes", () => {
    window.localStorage.setItem("distillery.run_ref.default", "{not-json");
    expect(loadRunReference("default")).toBeNull();

    window.localStorage.setItem(
      "distillery.run_ref.default",
      JSON.stringify({
        mode: "default",
        runId: "bad",
        datasetId: "ds_finance_world_v1_smoke",
        updatedAt: "not-a-date",
      }),
    );
    expect(loadRunReference("default")).toBeNull();
  });

  it("guards server-side rendering without window", () => {
    vi.stubGlobal("window", undefined);
    expect(loadRunReference("default")).toBeNull();
    expect(
      persistRunReference("default", {
        mode: "default",
        runId: "run_fixture_tinyfable_001",
        datasetId: "ds_finance_world_v1_smoke",
        updatedAt: "2026-07-18T12:00:00.000Z",
      }),
    ).toBe(false);
    expect(() => clearRunReference()).not.toThrow();
  });

  it("fails closed when storage access throws", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("denied", "SecurityError");
    });
    expect(loadRunReference("default")).toBeNull();
  });

  it("rejects invalid references before persistence", () => {
    expect(
      persistRunReference("default", {
        mode: "default",
        runId: "not-a-run",
        datasetId: "ds_finance_world_v1_smoke",
        updatedAt: "2026-07-18T12:00:00.000Z",
      }),
    ).toBe(false);
    expect(loadRunReference("default")).toBeNull();
  });

  it("isolates stored references by mode", () => {
    expect(
      persistRunReference("default", {
        mode: "default",
        runId: "run_fixture_tinyfable_restored_002",
        datasetId: "ds_finance_world_v1_smoke",
        updatedAt: "2026-07-18T12:00:00.000Z",
      }),
    ).toBe(true);
    expect(loadRunReference("default")?.runId).toBe(
      "run_fixture_tinyfable_restored_002",
    );
    expect(loadRunReference("proved")).toBeNull();
  });

  it("rejects a reference stored under the wrong mode", () => {
    expect(
      persistRunReference("proved", {
        mode: "default",
        runId: "run_fixture_tinyfable_001",
        datasetId: "ds_finance_world_v1_smoke",
        updatedAt: "2026-07-18T12:00:00.000Z",
      }),
    ).toBe(false);
  });
});

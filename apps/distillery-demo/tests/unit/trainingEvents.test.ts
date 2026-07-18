import { describe, expect, it } from "vitest";
import { buildStageBundle } from "@/lib/fixtures/bundle";
import { adaptLiveTrainingGlance } from "@/lib/trainingEvents";

describe("honest training event adapter", () => {
  it("labels default fixtures as fixture rehearsal, never live", () => {
    const bundle = buildStageBundle("default");
    const glance = adaptLiveTrainingGlance({
      run: bundle.run,
      plan: bundle.plan,
      telemetry: bundle.training_telemetry,
      artifact: bundle.artifact,
    });
    expect(glance.isLive).toBe(false);
    expect(glance.origin).toBe("fixture");
    expect(glance.originLabel).toMatch(/Fixture rehearsal/i);
    expect(glance.status).toBe("not_started");
  });

  it("labels precomputed prior runs without claiming live progress", () => {
    const bundle = buildStageBundle("proved");
    const glance = adaptLiveTrainingGlance({
      run: bundle.run,
      plan: bundle.plan,
      telemetry: bundle.training_telemetry,
      artifact: bundle.artifact,
    });
    expect(glance.isLive).toBe(false);
    expect(glance.origin).toBe("precomputed_prior_run");
    expect(glance.originLabel).toMatch(/Prior-run record/i);
    expect(glance.status).toBe("finished");
    expect(glance.recentEvent.summary.length).toBeGreaterThan(0);
    expect(glance.spendLabel).toMatch(/Spend ceiling/);
  });
});

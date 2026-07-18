import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  assertProofProvenance,
  assertFixtureBundleIntegrity,
} from "@/lib/fixtureIntegrity";
import { buildStageBundle } from "@/lib/fixtures/bundle";
import { assertSha256, HASH, isSha256 } from "@/lib/fixtures/hashes";

describe("fixture SHA-256 integrity", () => {
  it.each(Object.entries(HASH))(
    "stores the real digest for deterministic %s input",
    (name, digest) => {
      const expected = createHash("sha256")
        .update(`distillery-fixture-v1:${name}`)
        .digest("hex");
      expect(digest).toBe(expected);
      expect(isSha256(digest)).toBe(true);
    },
  );

  it("rejects malformed, uppercase, and truncated claims", () => {
    expect(() => assertSha256("a".repeat(63))).toThrow(/64-character/);
    expect(() => assertSha256("A".repeat(64))).toThrow(/lowercase/);
    expect(() => assertSha256("g".repeat(64))).toThrow(/hexadecimal/);
  });

  it("validates dataset, artifact, and proof protocol hashes", () => {
    const bundle = buildStageBundle("proved");
    expect(() => assertFixtureBundleIntegrity(bundle)).not.toThrow();
    expect(isSha256(bundle.dataset.content_sha256)).toBe(true);
    expect(
      Object.values(bundle.artifact?.checksums ?? {}).every(isSha256),
    ).toBe(true);
    expect(
      bundle.proof?.artifact_downloads.every((artifact) =>
        isSha256(artifact.sha256),
      ),
    ).toBe(true);
    expect(isSha256(bundle.proof?.protocol_sha256)).toBe(true);
  });
});

describe("fixture provenance integrity", () => {
  it("rejects economics labels that contradict the contract field", () => {
    const proof = structuredClone(buildStageBundle("proved").proof);
    expect(proof).not.toBeNull();
    if (!proof) return;
    proof.economics.serving_cost_projected = false;
    expect(() => assertProofProvenance(proof)).toThrow(
      /serving_cost_projected/,
    );
  });

  it("rejects claimed proof without prior-run provenance", () => {
    const proof = structuredClone(buildStageBundle("proved").proof);
    expect(proof).not.toBeNull();
    if (!proof) return;
    proof.precomputed = false;
    expect(() => assertProofProvenance(proof)).toThrow(/precomputed prior-run/);
  });

  it("rejects projected limitations on measured economics", () => {
    const proof = structuredClone(buildStageBundle("proved").proof);
    expect(proof).not.toBeNull();
    if (!proof) return;
    proof.economics.serving_cost_projected = false;
    proof.economics.note = "Serving costs are measured prior-run values.";
    expect(() => assertProofProvenance(proof)).toThrow(
      /projected-cost limitation/,
    );
  });
});

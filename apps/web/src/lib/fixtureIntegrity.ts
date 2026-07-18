import { fixtureClientError } from "@/lib/fixtureErrors";
import { assertSha256 } from "@/lib/fixtures/hashes";
import type {
  Dataset,
  ModelArtifactMeta,
  ProofReportView,
  StageBundle,
} from "@/lib/types";

function assertDatasetHashes(dataset: Dataset): void {
  assertSha256(dataset.content_sha256, "dataset content");
  assertSha256(dataset.split_sha256.train, "train split");
  assertSha256(dataset.split_sha256.validation, "validation split");
  if (dataset.split_sha256.test) assertSha256(dataset.split_sha256.test, "test split");
  if (dataset.split_sha256.iid_test) {
    assertSha256(dataset.split_sha256.iid_test, "IID split");
  }
  if (dataset.split_sha256.ood_test) {
    assertSha256(dataset.split_sha256.ood_test, "OOD split");
  }
  for (const [name, digest] of Object.entries(dataset.world_hashes)) {
    assertSha256(digest, `world hash ${name}`);
  }
}

function assertArtifactHashes(artifact: ModelArtifactMeta): void {
  for (const [name, digest] of Object.entries(artifact.checksums)) {
    assertSha256(digest, `artifact checksum ${name}`);
  }
}

export function assertProofProvenance(proof: ProofReportView): void {
  assertSha256(proof.protocol_sha256, "proof protocol");
  if (!proof.precomputed) {
    throw fixtureClientError(
      "FIXTURE_INTEGRITY_ERROR",
      "A fixture proof report must identify precomputed prior-run provenance.",
      "fixture",
      proof.report_id,
    );
  }

  if (
    proof.systems &&
    proof.systems.measurement_source !== "precomputed_prior_run"
  ) {
    throw fixtureClientError(
      "FIXTURE_INTEGRITY_ERROR",
      "Precomputed proof systems metrics must be labeled as prior-run measurements.",
      "fixture",
      proof.report_id,
    );
  }

  const noteClaimsProjection = /\b(projected|estimated|estimate|estimates)\b/i.test(
    proof.economics.note,
  );
  if (proof.economics.serving_cost_projected !== noteClaimsProjection) {
    throw fixtureClientError(
      "FIXTURE_INTEGRITY_ERROR",
      "Economics provenance conflicts with serving_cost_projected.",
      "fixture",
      proof.report_id,
      {
        serving_cost_projected: proof.economics.serving_cost_projected,
        note: proof.economics.note,
      },
    );
  }
  if (
    !proof.economics.serving_cost_projected &&
    proof.limitations.some((limitation) =>
      /\b(projected|estimated|estimate|estimates)\b/i.test(limitation),
    )
  ) {
    throw fixtureClientError(
      "FIXTURE_INTEGRITY_ERROR",
      "Measured economics cannot retain projected-cost limitation copy.",
      "fixture",
      proof.report_id,
    );
  }

  for (const artifact of proof.artifact_downloads) {
    assertSha256(artifact.sha256, `proof artifact ${artifact.name}`);
  }
}

export function assertFixtureBundleIntegrity(bundle: StageBundle): void {
  assertDatasetHashes(bundle.dataset);
  if (bundle.artifact) assertArtifactHashes(bundle.artifact);
  if (bundle.proof) assertProofProvenance(bundle.proof);
}

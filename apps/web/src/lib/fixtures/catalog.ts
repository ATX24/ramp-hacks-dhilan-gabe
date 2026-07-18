import type { ProofStatus, UiMode } from "@/lib/types";

export const DEFAULT_DATASET_ID = "ds_finance_world_v1_smoke";
export const DEFAULT_RUN_ID = "run_fixture_tinyfable_001";
export const RESTORED_RUN_ID = "run_fixture_tinyfable_restored_002";

export interface FixtureModeEntry {
  defaultRunId: string;
  compatibleRunIds: readonly string[];
  artifactId: string | null;
  reportId: string | null;
  proofStatus: ProofStatus | null;
}

export const FIXTURE_MODE_CATALOG: Record<UiMode, FixtureModeEntry> = {
  default: {
    defaultRunId: DEFAULT_RUN_ID,
    compatibleRunIds: [DEFAULT_RUN_ID, RESTORED_RUN_ID],
    artifactId: null,
    reportId: null,
    proofStatus: null,
  },
  precomputed: {
    defaultRunId: "run_fixture_precomputed_001",
    compatibleRunIds: ["run_fixture_precomputed_001"],
    artifactId: "art_tinyfable_precomputed_001",
    reportId: "pr_tinyfable_precomputed_001",
    proofStatus: "do_not_distill",
  },
  proved: {
    defaultRunId: "run_fixture_proved_001",
    compatibleRunIds: ["run_fixture_proved_001"],
    artifactId: "art_tinyfable_proved_001",
    reportId: "pr_tinyfable_proved_001",
    proofStatus: "proved",
  },
  do_not_distill: {
    defaultRunId: "run_fixture_do_not_distill_001",
    compatibleRunIds: ["run_fixture_do_not_distill_001"],
    artifactId: "art_tinyfable_do_not_distill_001",
    reportId: "pr_tinyfable_do_not_distill_001",
    proofStatus: "do_not_distill",
  },
  failed_quality: {
    defaultRunId: "run_fixture_failed_quality_001",
    compatibleRunIds: ["run_fixture_failed_quality_001"],
    artifactId: "art_tinyfable_failed_quality_001",
    reportId: "pr_tinyfable_failed_quality_001",
    proofStatus: "failed_quality",
  },
  failed_economics: {
    defaultRunId: "run_fixture_failed_economics_001",
    compatibleRunIds: ["run_fixture_failed_economics_001"],
    artifactId: "art_tinyfable_failed_economics_001",
    reportId: "pr_tinyfable_failed_economics_001",
    proofStatus: "failed_economics",
  },
  insufficient_evidence: {
    defaultRunId: "run_fixture_insufficient_evidence_001",
    compatibleRunIds: ["run_fixture_insufficient_evidence_001"],
    artifactId: "art_tinyfable_insufficient_evidence_001",
    reportId: "pr_tinyfable_insufficient_evidence_001",
    proofStatus: "insufficient_evidence",
  },
  error: {
    defaultRunId: "run_fixture_error_001",
    compatibleRunIds: ["run_fixture_error_001"],
    artifactId: null,
    reportId: null,
    proofStatus: null,
  },
  unavailable: {
    defaultRunId: "run_fixture_unavailable_001",
    compatibleRunIds: ["run_fixture_unavailable_001"],
    artifactId: null,
    reportId: null,
    proofStatus: null,
  },
  skipped_synthesis: {
    defaultRunId: "run_fixture_skipped_synthesis_001",
    compatibleRunIds: ["run_fixture_skipped_synthesis_001"],
    artifactId: null,
    reportId: null,
    proofStatus: null,
  },
  no_training_yet: {
    defaultRunId: "run_fixture_no_training_yet_001",
    compatibleRunIds: ["run_fixture_no_training_yet_001"],
    artifactId: null,
    reportId: null,
    proofStatus: null,
  },
  loading: {
    defaultRunId: "run_fixture_loading_001",
    compatibleRunIds: ["run_fixture_loading_001"],
    artifactId: null,
    reportId: null,
    proofStatus: null,
  },
  fetch_failure: {
    defaultRunId: "run_fixture_fetch_failure_001",
    compatibleRunIds: ["run_fixture_fetch_failure_001"],
    artifactId: null,
    reportId: null,
    proofStatus: null,
  },
};

export function getFixtureEntry(mode: UiMode): FixtureModeEntry {
  return FIXTURE_MODE_CATALOG[mode];
}

export function getDefaultRunId(mode: UiMode): string {
  return getFixtureEntry(mode).defaultRunId;
}

export function modesForRunId(runId: string): UiMode[] {
  return (Object.entries(FIXTURE_MODE_CATALOG) as Array<[UiMode, FixtureModeEntry]>)
    .filter(([, entry]) => entry.compatibleRunIds.includes(runId))
    .map(([mode]) => mode);
}

export function modeForArtifactId(artifactId: string): UiMode | null {
  const match = (Object.entries(FIXTURE_MODE_CATALOG) as Array<
    [UiMode, FixtureModeEntry]
  >).find(([, entry]) => entry.artifactId === artifactId);
  return match?.[0] ?? null;
}

export function modeForReportId(reportId: string): UiMode | null {
  const match = (Object.entries(FIXTURE_MODE_CATALOG) as Array<
    [UiMode, FixtureModeEntry]
  >).find(([, entry]) => entry.reportId === reportId);
  return match?.[0] ?? null;
}

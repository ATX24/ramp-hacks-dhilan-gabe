import { StatusBadge } from "@/components/StatusBadge";
import type { UiMode } from "@/lib/types";

const COPY: Record<UiMode, { title: string; body: string; tone: "info" | "warn" }> = {
  default: {
    title: "Fixture-driven rehearsal",
    body: "Static fixture responses only. No live API or active run.",
    tone: "info",
  },
  no_training_yet: {
    title: "No training yet",
    body: "Preflight and planned finite-job metadata only. Nothing has been submitted.",
    tone: "info",
  },
  skipped_synthesis: {
    title: "Synthesis skipped",
    body: "Valid responses already present (skip_reason=responses_already_present).",
    tone: "info",
  },
  precomputed: {
    title: "Precomputed artifacts",
    body: "Checksum-verified prior-run artifacts are shown with projected economics labels. Nothing is active.",
    tone: "warn",
  },
  proved: {
    title: "Proved",
    body:
      "A precomputed prior-run report satisfies the frozen proof gates. Nothing is active.",
    tone: "info",
  },
  do_not_distill: {
    title: "Do not distill",
    body:
      "A precomputed prior-run report records that a cheaper baseline won. Nothing is active.",
    tone: "warn",
  },
  failed_quality: {
    title: "Failed quality",
    body:
      "A precomputed prior-run report failed its quality gate. Nothing is active.",
    tone: "warn",
  },
  failed_economics: {
    title: "Failed economics",
    body:
      "A precomputed prior-run report failed its economics gate. Nothing is active.",
    tone: "warn",
  },
  error: {
    title: "Error fixture mode",
    body: "Rendering typed failure and failed leakage checks from static fixtures.",
    tone: "warn",
  },
  unavailable: {
    title: "Unavailable recipe mode",
    body: "Catalog-only recipe fails loud with RECIPE_NOT_IMPLEMENTED. No silent downgrade.",
    tone: "warn",
  },
  insufficient_evidence: {
    title: "Insufficient evidence",
    body:
      "A precomputed prior run completed, but missing replication leaves proof_status=insufficient_evidence. Nothing is active.",
    tone: "warn",
  },
  loading: {
    title: "Loading fixture state",
    body: "Showing the explicit loading interface. No network request is being made.",
    tone: "info",
  },
  fetch_failure: {
    title: "Fixture fetch failure",
    body: "Showing the explicit request-failure interface without calling a live API.",
    tone: "warn",
  },
};

export function ModeBanner({ mode }: { mode: UiMode }) {
  const copy = COPY[mode];
  return (
    <div
      className={`banner banner-${copy.tone === "info" ? "info" : "warn"}`}
      role="status"
      data-testid="mode-banner"
      data-mode={mode}
    >
      <div className="controls" style={{ justifyContent: "space-between" }}>
        <strong>{copy.title}</strong>
        {[
          "precomputed",
          "proved",
          "do_not_distill",
          "failed_quality",
          "failed_economics",
          "insufficient_evidence",
        ].includes(mode) ? (
          <StatusBadge tone="precomputed">Precomputed</StatusBadge>
        ) : null}
      </div>
      <p style={{ margin: 0 }}>{copy.body}</p>
    </div>
  );
}

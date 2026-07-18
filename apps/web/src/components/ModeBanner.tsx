import { StatusBadge } from "@/components/StatusBadge";
import type { UiMode } from "@/lib/types";

const COPY: Record<UiMode, { title: string; body: string; tone: "info" | "warn" }> = {
  default: {
    title: "This is a saved sample",
    body: "The page uses made-up data. It will not call a service or start a job.",
    tone: "info",
  },
  no_training_yet: {
    title: "Nothing has started",
    body: "You can inspect the plan, but no job has been sent.",
    tone: "info",
  },
  skipped_synthesis: {
    title: "The answers were already there",
    body: "This saved sample did not need to fill any gaps.",
    tone: "info",
  },
  precomputed: {
    title: "This is a saved run",
    body: "The files and results came from an earlier run. Nothing is running now.",
    tone: "warn",
  },
  proved: {
    title: "The saved run passed",
    body:
      "The earlier run passed its saved checks. Nothing is running now.",
    tone: "info",
  },
  do_not_distill: {
    title: "Keep the current model",
    body:
      "The saved result says a smaller model would not be the better choice.",
    tone: "warn",
  },
  failed_quality: {
    title: "The result was not accurate enough",
    body:
      "The saved run missed its accuracy target. Nothing is running now.",
    tone: "warn",
  },
  failed_economics: {
    title: "The result would not save money",
    body:
      "The saved run cost more than the plan allowed.",
    tone: "warn",
  },
  error: {
    title: "This sample has a data problem",
    body: "The page shows what happens when copied examples fail a safety check.",
    tone: "warn",
  },
  unavailable: {
    title: "This method is not available",
    body: "The page stops instead of choosing another method.",
    tone: "warn",
  },
  insufficient_evidence: {
    title: "The saved run needs another check",
    body:
      "The earlier run finished, but one repeat is missing. The result is not final.",
    tone: "warn",
  },
  loading: {
    title: "Loading the sample",
    body: "The page is showing a saved loading state. It is not making a request.",
    tone: "info",
  },
  fetch_failure: {
    title: "The sample did not load",
    body: "This is a saved error state. The page did not call a live service.",
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
          <StatusBadge tone="precomputed">Saved result</StatusBadge>
        ) : null}
      </div>
      <p style={{ margin: 0 }}>{copy.body}</p>
    </div>
  );
}

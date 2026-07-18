import type { UiMode } from "@/lib/types";

export const UI_MODES = [
  "default",
  "precomputed",
  "proved",
  "do_not_distill",
  "failed_quality",
  "failed_economics",
  "error",
  "unavailable",
  "insufficient_evidence",
  "skipped_synthesis",
  "no_training_yet",
  "loading",
  "fetch_failure",
] as const satisfies readonly UiMode[];

export function isUiMode(value: unknown): value is UiMode {
  return typeof value === "string" && (UI_MODES as readonly string[]).includes(value);
}

export function parseUiMode(value: unknown): UiMode {
  return isUiMode(value) ? value : "default";
}

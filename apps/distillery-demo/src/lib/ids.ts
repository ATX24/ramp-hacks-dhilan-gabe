import { fixtureClientError } from "@/lib/fixtureErrors";
import type { FixtureResourceKind } from "@/lib/types";

const PREFIX_BY_KIND: Record<FixtureResourceKind, string> = {
  dataset: "ds_",
  run: "run_",
  artifact: "art_",
  report: "pr_",
};

const RESOURCE_ID_PATTERN = /^[a-z][a-z0-9]*_[a-z0-9][a-z0-9_-]{2,126}$/;

export function isResourceId(kind: FixtureResourceKind, value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.startsWith(PREFIX_BY_KIND[kind]) &&
    RESOURCE_ID_PATTERN.test(value)
  );
}

export function assertResourceId(kind: FixtureResourceKind, value: string): void {
  if (!isResourceId(kind, value)) {
    throw fixtureClientError(
      "INVALID_RESOURCE_ID",
      `Invalid ${kind} fixture ID.`,
      kind,
      value,
      { expected_prefix: PREFIX_BY_KIND[kind] },
    );
  }
}

import type {
  FixtureClientErrorCode,
  FixtureClientErrorPayload,
  FixtureResourceKind,
} from "@/lib/types";

export class FixtureClientError extends Error {
  readonly payload: FixtureClientErrorPayload;

  constructor(payload: FixtureClientErrorPayload) {
    super(payload.message);
    this.name = "FixtureClientError";
    this.payload = payload;
  }
}

export function fixtureClientError(
  code: FixtureClientErrorCode,
  message: string,
  resourceKind: FixtureResourceKind | "fixture",
  resourceId: string | null,
  details: Record<string, unknown> = {},
): FixtureClientError {
  return new FixtureClientError({
    code,
    message,
    resource_kind: resourceKind,
    resource_id: resourceId,
    details,
    retryable: false,
  });
}

export function isFixtureClientError(error: unknown): error is FixtureClientError {
  return error instanceof FixtureClientError;
}

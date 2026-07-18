export type RunReferenceStatus =
  | "resolving"
  | "stored"
  | "session_only"
  | "invalid";

export function RunReferenceBar({
  runId,
  datasetId,
  status,
}: {
  runId: string;
  datasetId: string;
  status: RunReferenceStatus;
}) {
  return (
    <div
      className="meta-row"
      data-testid="run-reference"
      data-status={status}
      aria-live="polite"
    >
      <span>
        Run reference:{" "}
        <code>
          {status === "resolving"
            ? "resolving…"
            : status === "invalid"
              ? "rejected"
              : runId}
        </code>
      </span>
      <span>
        Dataset:{" "}
        <code>
          {status === "resolving"
            ? "resolving…"
            : status === "invalid"
              ? "unresolved"
              : datasetId}
        </code>
      </span>
      <span>
        {status === "resolving"
          ? "Restoring this mode’s stored run"
          : status === "stored"
            ? "Stored locally for this mode"
            : status === "invalid"
              ? "Explicit run reference is invalid"
              : "Local persistence unavailable; showing this session reference"}
      </span>
    </div>
  );
}

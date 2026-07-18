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
    <details
      className="rounded-[14px] border border-border bg-card px-3"
      data-testid="run-reference"
      data-status={status}
    >
      <summary className="min-h-11 cursor-pointer py-3 text-sm font-medium">
        Advanced run details
      </summary>
      <div className="meta-row pb-3" aria-live="polite">
        <span>
          Run ID:{" "}
          <code>
            {status === "resolving"
              ? "resolving..."
              : status === "invalid"
                ? "rejected"
                : runId}
          </code>
        </span>
        <span>
          Data ID:{" "}
          <code>
            {status === "resolving"
              ? "resolving..."
              : status === "invalid"
                ? "unresolved"
                : datasetId}
          </code>
        </span>
        <span>
          {status === "resolving"
            ? "Opening the saved run"
            : status === "stored"
              ? "Saved in this browser"
              : status === "invalid"
                ? "The run ID is not valid"
                : "This run will last for this visit only"}
        </span>
      </div>
    </details>
  );
}

import type { ErrorPayload } from "@/lib/types";

export function ErrorBanner({ error }: { error: ErrorPayload | null }) {
  if (!error) return null;
  return (
    <div className="banner banner-error" role="alert" aria-live="assertive">
      <strong>This step could not finish</strong>
      <p style={{ margin: 0 }}>{error.message}</p>
      <p style={{ margin: 0 }}>
        {error.retryable
          ? "You can try this step again."
          : "Change the inputs before you try again."}
      </p>
      <details className="text-sm">
        <summary className="min-h-11 cursor-pointer py-3 font-medium">
          Advanced error details
        </summary>
        <p style={{ margin: 0 }}>
          Error code: <code>{error.code}</code>
        </p>
        {error.run_id ? (
          <p style={{ margin: 0 }}>
            Run ID: <code>{error.run_id}</code>
          </p>
        ) : null}
      </details>
    </div>
  );
}

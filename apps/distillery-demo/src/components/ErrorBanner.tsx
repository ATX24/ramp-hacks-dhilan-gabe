import type { ErrorPayload } from "@/lib/types";

export function ErrorBanner({ error }: { error: ErrorPayload | null }) {
  if (!error) return null;
  return (
    <div className="banner banner-error" role="alert" aria-live="assertive">
      <strong>{error.code}</strong>
      <p style={{ margin: 0 }}>{error.message}</p>
      {error.run_id ? (
        <p style={{ margin: 0 }} className="mono">
          run_id: {error.run_id}
        </p>
      ) : null}
      <p style={{ margin: 0 }}>
        retryable: {error.retryable ? "yes" : "no"}
      </p>
    </div>
  );
}

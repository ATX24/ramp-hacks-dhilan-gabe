import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import type { ErrorPayload } from "@/lib/types";

export function ErrorBanner({ error }: { error: ErrorPayload | null }) {
  if (!error) return null;
  return (
    <Alert
      role="alert"
      aria-live="assertive"
      className="mb-4 border-[color-mix(in_oklab,var(--fail)_30%,transparent)] bg-[color-mix(in_oklab,var(--fail)_8%,transparent)]"
    >
      <AlertTitle className="font-serif">This step could not finish</AlertTitle>
      <AlertDescription>
        <p>{error.message}</p>
        <p className="mt-1 text-sm">
          {error.retryable
            ? "You can try this step again."
            : "Change the inputs before you try again."}
        </p>
        <details className="mt-2 text-sm">
          <summary className="min-h-11 cursor-pointer py-3 font-medium">
            Advanced error details
          </summary>
          <p>
            Error code: <code>{error.code}</code>
          </p>
          {error.run_id ? (
            <p className="mt-1">
              Run ID: <code>{error.run_id}</code>
            </p>
          ) : null}
        </details>
      </AlertDescription>
    </Alert>
  );
}

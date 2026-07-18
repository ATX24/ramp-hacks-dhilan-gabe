import type { ReactNode } from "react";
import type { StageLoadState } from "@/lib/types";

export function StageStateBoundary({
  state,
  stageName,
  children,
}: {
  state: StageLoadState;
  stageName: string;
  children: ReactNode;
}) {
  switch (state.status) {
    case "ready":
      return children;

    case "loading":
      return (
        <section aria-labelledby="stage-loading-heading" aria-busy="true">
          <div className="panel">
            <h1 id="stage-loading-heading">{stageName}</h1>
            <div className="banner banner-info" role="status" data-testid="stage-loading">
              <strong>Opening this page</strong>
              <p style={{ margin: 0 }}>{state.message}</p>
            </div>
          </div>
        </section>
      );

    case "failed":
      return (
        <section aria-labelledby="stage-failure-heading">
          <div className="panel">
            <h1 id="stage-failure-heading">{stageName}</h1>
            <div className="banner banner-error" role="alert" data-testid="stage-fetch-failure">
              <strong>{state.title}</strong>
              <p style={{ margin: 0 }}>{state.message}</p>
              <p style={{ margin: 0 }}>
                {state.retryable
                  ? "You can try again."
                  : "This saved state cannot be retried."}
              </p>
            </div>
          </div>
        </section>
      );

    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}

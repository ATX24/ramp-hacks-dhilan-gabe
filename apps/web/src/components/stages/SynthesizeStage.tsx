import { ErrorBanner } from "@/components/ErrorBanner";
import { StatusBadge } from "@/components/StatusBadge";
import type { ErrorPayload, SynthesisSummary } from "@/lib/types";

function sourceLabel(source: string): string {
  const labels: Record<string, string> = {
    imported: "Provided",
    oracle: "Known correct",
    teacher_generated: "Filled by source model",
    relabeled: "Fixed",
    rejected: "Rejected",
  };
  return labels[source] ?? source;
}

export function SynthesizeStage({
  synthesis,
  error,
  priorRun = false,
}: {
  synthesis: SynthesisSummary;
  error: ErrorPayload | null;
  priorRun?: boolean;
}) {
  return (
    <section aria-labelledby="synthesize-heading">
      <div className="panel">
        <p className="text-kicker text-[var(--orange)]">Synthesize</p>
        <h1 id="synthesize-heading">Fill the missing answers</h1>
        <p>
          Good answers stay as they are. A larger source model only fills gaps or
          replaces answers that fail the checks.
        </p>
        {priorRun ? (
          <div className="banner banner-info" role="status" data-testid="prior-synthesis">
            <strong>This came from a saved run</strong>
            <p style={{ margin: 0 }}>
              The counts and source records were saved earlier. Nothing is running now.
            </p>
          </div>
        ) : null}
        <ErrorBanner error={error} />
        {synthesis.skipped ? (
          <div className="banner banner-info" data-testid="synthesis-skipped" role="status">
            <strong>Nothing needed filling</strong>
            <p style={{ margin: 0 }}>
              The provided answers already passed the saved checks.
            </p>
          </div>
        ) : null}
      </div>

      <div className="panel">
        <h3>What happened to the answers</h3>
        <p>
          This shows which answers were kept, rejected, fixed, or added. The default
          keeps valid answers and only pays to fill gaps.
        </p>
        <div className="grid-3">
          <div className="stat">
            <span className="label">Kept as provided</span>
            <span className="value">{synthesis.counts.imported}</span>
          </div>
          <div className="stat">
            <span className="label">Rejected</span>
            <span className="value">{synthesis.counts.rejected}</span>
          </div>
          <div className="stat">
            <span className="label">Fixed</span>
            <span className="value">{synthesis.counts.relabeled}</span>
          </div>
          <div className="stat">
            <span className="label">Added</span>
            <span className="value">{synthesis.counts.generated}</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>
          {priorRun
            ? "Saved source model and cost"
            : "Source model and estimated cost"}
        </h3>
        <p>
          The source model supplies missing answers. The estimate matters because each
          call can add cost. Leaving auto on only calls it when an answer is missing.
        </p>
        {synthesis.teacher ? (
          <>
            <div className="meta-row">
              <span>
                {priorRun ? "Saved calls" : "Planned calls"}{" "}
                <strong>{synthesis.teacher.calls_planned}</strong>
              </span>
              <span>
                Cost estimate{" "}
                <strong>${synthesis.teacher.estimated_cost_usd.toFixed(2)}</strong>
              </span>
            </div>
            <p>
              This is an estimate. {priorRun ? "It came from the saved run. " : ""}
              This page does not call the source model.
            </p>
            <details className="rounded-[14px] border border-border px-3">
              <summary className="min-h-11 cursor-pointer py-3 font-medium">
                Advanced source model record
              </summary>
              <p>
                Model ID: <code>{synthesis.teacher.id}</code>
                <br />
                Version: <code>{synthesis.teacher.revision}</code>
              </p>
            </details>
          </>
        ) : (
          <p>
            The plan does not need the source model.
            {synthesis.skipped
              ? " The provided answers already passed the checks."
              : ""}
          </p>
        )}
      </div>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
          Advanced answer sources
        </summary>
        <p>
          Each row keeps its source so a reviewer can tell which answers were provided
          and which ones the source model created.
        </p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Record</th>
                <th scope="col">Job</th>
                <th scope="col">Source</th>
                <th scope="col">What happened</th>
              </tr>
            </thead>
            <tbody>
              {synthesis.provenance_examples.map((example) => (
                <tr key={example.example_id}>
                  <td>
                    <code>{example.example_id}</code>
                  </td>
                  <td>{example.task}</td>
                  <td>
                    <StatusBadge
                      tone={
                        example.label_source === "rejected"
                          ? "fail"
                          : example.label_source === "relabeled" ||
                              example.label_source === "teacher_generated"
                            ? "warn"
                            : "pass"
                      }
                    >
                      {sourceLabel(example.label_source)}
                    </StatusBadge>
                  </td>
                  <td>{example.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}

import { ErrorBanner } from "@/components/ErrorBanner";
import { StatusBadge } from "@/components/StatusBadge";
import { STAGE_PLAIN } from "@/lib/plainLanguage";
import type { ErrorPayload, SynthesisSummary } from "@/lib/types";

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
        <p className="text-kicker">{STAGE_PLAIN.synthesize.plain}</p>
        <h2 id="synthesize-heading">Synthesize</h2>
        <p>{STAGE_PLAIN.synthesize.description}</p>
        <p className="text-sm text-muted-foreground">
          Why this matters: {STAGE_PLAIN.synthesize.why}
        </p>
        {priorRun ? (
          <div className="banner banner-info" role="status" data-testid="prior-synthesis">
            <strong>Prior-run provenance</strong>
            <p style={{ margin: 0 }}>
              These counts and provenance records come from a precomputed prior run.
              Nothing is active.
            </p>
          </div>
        ) : null}
        <ErrorBanner error={error} />
        {synthesis.skipped ? (
          <div className="banner banner-info" data-testid="synthesis-skipped" role="status">
            <strong>Synthesis skipped</strong>
            <p style={{ margin: 0 }}>
              skip_reason=<code>{synthesis.skip_reason}</code>
            </p>
          </div>
        ) : null}
      </div>

      <div className="panel">
        <h3>Response provenance counts</h3>
        <div className="grid-3">
          <div className="stat">
            <span className="label">Imported</span>
            <span className="value">{synthesis.counts.imported}</span>
          </div>
          <div className="stat">
            <span className="label">Rejected</span>
            <span className="value">{synthesis.counts.rejected}</span>
          </div>
          <div className="stat">
            <span className="label">Relabeled</span>
            <span className="value">{synthesis.counts.relabeled}</span>
          </div>
          <div className="stat">
            <span className="label">Generated</span>
            <span className="value">{synthesis.counts.generated}</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>
          {priorRun
            ? "Recorded teacher provenance & cost estimate"
            : "Teacher identity & cost estimate"}
        </h3>
        {synthesis.teacher ? (
          <>
            <div className="meta-row">
              <span>
                Teacher <code>{synthesis.teacher.id}</code>
              </span>
              <span>
                Revision <code>{synthesis.teacher.revision}</code>
              </span>
              <span>
                {priorRun ? "Recorded calls" : "Planned calls"}{" "}
                <strong>{synthesis.teacher.calls_planned}</strong>
              </span>
              <span>
                Estimated cost{" "}
                <strong>${synthesis.teacher.estimated_cost_usd.toFixed(2)}</strong>
              </span>
            </div>
            <p>
              Estimate only. {priorRun ? "This is prior-run metadata. " : ""}
              No live teacher calls are issued from this UI.
            </p>
          </>
        ) : (
          <p>
            No teacher calls planned.
            {synthesis.skipped
              ? " Synthesis skipped because usable responses already exist."
              : ""}
          </p>
        )}
      </div>

      <div className="panel">
        <h3>Provenance examples</h3>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Example</th>
                <th scope="col">Task</th>
                <th scope="col">Source</th>
                <th scope="col">Note</th>
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
                      {example.label_source}
                    </StatusBadge>
                  </td>
                  <td>{example.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

import Link from "next/link";
import { ErrorBanner } from "@/components/ErrorBanner";
import { StatusBadge } from "@/components/StatusBadge";
import { buildStageHref } from "@/lib/navigation";
import type { Dataset, ErrorPayload, UiMode } from "@/lib/types";

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function CurateStage({
  dataset,
  error,
  mode,
  runId,
}: {
  dataset: Dataset;
  error: ErrorPayload | null;
  mode: UiMode;
  runId: string;
}) {
  return (
    <section aria-labelledby="curate-heading">
      <div className="panel">
        <h2 id="curate-heading">Curate</h2>
        <p>
          Select and freeze a synthetic finance dataset. Show mixture, schema issues,
          split/world hashes, and leakage checks before any synthesis or training.
        </p>
        <ErrorBanner error={error} />
        <div className="meta-row">
          <span>
            Dataset <code>{dataset.dataset_id}</code>
          </span>
          <span>
            Examples <strong>{dataset.example_count}</strong>
          </span>
          <StatusBadge tone={dataset.frozen ? "pass" : "fail"}>
            {dataset.frozen ? "Frozen" : "Not frozen"}
          </StatusBadge>
        </div>
        <p>{dataset.provenance_summary}</p>
        <div className="controls" style={{ marginBottom: "1rem" }}>
          {dataset.frozen ? (
            <Link
              href={buildStageHref("/synthesize", mode, runId)}
              className="btn btn-primary"
              data-testid="curate-continue"
            >
              Continue to Synthesize
            </Link>
          ) : (
            <button
              type="button"
              className="btn btn-primary"
              disabled
              data-testid="curate-blocked"
            >
              Resolve dataset checks first
            </button>
          )}
          <button type="button" className="btn" disabled>
            Upload (fixture mode)
          </button>
        </div>
      </div>

      <div className="panel">
        <h3>Task mixture</h3>
        <div className="grid-3">
          <div className="stat">
            <span className="label">transaction_review</span>
            <span className="value">{pct(dataset.task_mixture.transaction_review)}</span>
          </div>
          <div className="stat">
            <span className="label">variance_analysis</span>
            <span className="value">{pct(dataset.task_mixture.variance_analysis)}</span>
          </div>
          <div className="stat">
            <span className="label">cash_reconciliation</span>
            <span className="value">{pct(dataset.task_mixture.cash_reconciliation)}</span>
          </div>
        </div>
        <h3 style={{ marginTop: "1.25rem" }}>Difficulty mixture</h3>
        <div className="grid-3">
          <div className="stat">
            <span className="label">easy</span>
            <span className="value">{pct(dataset.difficulty_mixture.easy)}</span>
          </div>
          <div className="stat">
            <span className="label">medium</span>
            <span className="value">{pct(dataset.difficulty_mixture.medium)}</span>
          </div>
          <div className="stat">
            <span className="label">hard</span>
            <span className="value">{pct(dataset.difficulty_mixture.hard)}</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>Label sources</h3>
        <div className="grid-3">
          {Object.entries(dataset.label_sources).map(([source, count]) => (
            <div className="stat" key={source}>
              <span className="label">{source}</span>
              <span className="value">{count}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>Content & split hashes</h3>
          <ul className="list-plain">
            <li>
              content: <span className="hash">{dataset.content_sha256}</span>
            </li>
            <li>
              train: <span className="hash">{dataset.split_sha256.train}</span>
            </li>
            <li>
              validation: <span className="hash">{dataset.split_sha256.validation}</span>
            </li>
            <li>
              iid_test: <span className="hash">{dataset.split_sha256.iid_test ?? "—"}</span>
            </li>
            <li>
              ood_test: <span className="hash">{dataset.split_sha256.ood_test ?? "—"}</span>
            </li>
          </ul>
          <h3 style={{ marginTop: "1rem" }}>World hashes</h3>
          <ul className="list-plain">
            {Object.entries(dataset.world_hashes).map(([key, value]) => (
              <li key={key}>
                {key}: <span className="hash">{value}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="panel">
          <h3>Schema issues</h3>
          {dataset.schema_errors.length === 0 ? (
            <p>No schema issues recorded.</p>
          ) : (
            <ul className="list-plain">
              {dataset.schema_errors.map((issue) => (
                <li key={`${issue.example_id}-${issue.path}`}>
                  <StatusBadge tone={issue.severity === "error" ? "fail" : "warn"}>
                    {issue.severity}
                  </StatusBadge>{" "}
                  <code>{issue.example_id}</code> · {issue.path}: {issue.message}
                </li>
              ))}
            </ul>
          )}

          <h3 style={{ marginTop: "1rem" }}>Leakage checks</h3>
          <ul className="list-plain">
            {dataset.leakage_checks.map((check) => (
              <li key={check.check_id}>
                <StatusBadge tone={check.passed ? "pass" : "fail"}>
                  {check.passed ? "pass" : "fail"}
                </StatusBadge>{" "}
                <strong>{check.check_id}</strong> — {check.detail}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

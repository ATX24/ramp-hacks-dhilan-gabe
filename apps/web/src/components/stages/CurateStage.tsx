import Link from "next/link";
import { ErrorBanner } from "@/components/ErrorBanner";
import { StatusBadge } from "@/components/StatusBadge";
import { buildStageHref } from "@/lib/navigation";
import type { Dataset, ErrorPayload, UiMode } from "@/lib/types";

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function sourceLabel(source: string): string {
  const labels: Record<string, string> = {
    oracle: "Known correct",
    imported: "Provided",
    teacher_generated: "Filled by the source model",
    missing: "Missing",
  };
  return labels[source] ?? source;
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
        <p className="text-kicker text-[var(--orange)]">Curate</p>
        <h1 id="curate-heading">Check the data</h1>
        <p>
          Start with the data that teaches and tests the model. This page checks the
          file format, keeps test examples separate, and records the exact version.
        </p>
        <ErrorBanner error={error} />
        <div className="meta-row">
          <span>
            Records <strong>{dataset.example_count}</strong>
          </span>
          <StatusBadge tone={dataset.frozen ? "pass" : "fail"}>
            {dataset.frozen ? "Ready" : "Needs attention"}
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
              See how gaps are filled
            </Link>
          ) : (
            <button
              type="button"
              className="btn btn-primary"
              disabled
              data-testid="curate-blocked"
            >
              Fix the data checks first
            </button>
          )}
        </div>
      </div>

      <div className="panel">
        <h3>What is in the sample</h3>
        <p>
          This mix changes what the model practices. A balanced mix helps the
          generalist. Leave it as shown to use the saved sample.
        </p>
        <div className="grid-3">
          <div className="stat">
            <span className="label">Transaction review</span>
            <span className="value">{pct(dataset.task_mixture.transaction_review)}</span>
          </div>
          <div className="stat">
            <span className="label">Budget variance</span>
            <span className="value">{pct(dataset.task_mixture.variance_analysis)}</span>
          </div>
          <div className="stat">
            <span className="label">Cash matching</span>
            <span className="value">{pct(dataset.task_mixture.cash_reconciliation)}</span>
          </div>
        </div>
        <h3 style={{ marginTop: "1.25rem" }}>How hard the examples are</h3>
        <div className="grid-3">
          <div className="stat">
            <span className="label">Easy</span>
            <span className="value">{pct(dataset.difficulty_mixture.easy)}</span>
          </div>
          <div className="stat">
            <span className="label">Medium</span>
            <span className="value">{pct(dataset.difficulty_mixture.medium)}</span>
          </div>
          <div className="stat">
            <span className="label">Hard</span>
            <span className="value">{pct(dataset.difficulty_mixture.hard)}</span>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>Where the answers came from</h3>
        <p>
          The source matters because a wrong answer can teach the wrong behavior.
          Distillery keeps the source with every record.
        </p>
        <div className="grid-3">
          {Object.entries(dataset.label_sources).map(([source, count]) => (
            <div className="stat" key={source}>
              <span className="label">{sourceLabel(source)}</span>
              <span className="value">{count}</span>
            </div>
          ))}
        </div>
      </div>

      <details className="panel">
        <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
          Advanced data record
        </summary>
        <p>
          These details identify the exact files and checks behind this saved sample.
          You do not need them for the usual path.
        </p>
        <p>
          Data ID: <code>{dataset.dataset_id}</code>
        </p>
        <div className="grid-2">
          <div>
          <h3>Data fingerprints (hashes)</h3>
          <p>
            A fingerprint identifies the exact bytes in each split. It lets you repeat
            the run without guessing which file was used.
          </p>
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
              iid_test: <span className="hash">{dataset.split_sha256.iid_test ?? "Not available"}</span>
            </li>
            <li>
              ood_test: <span className="hash">{dataset.split_sha256.ood_test ?? "Not available"}</span>
            </li>
          </ul>
          <h3 style={{ marginTop: "1rem" }}>Generator fingerprints</h3>
          <ul className="list-plain">
            {Object.entries(dataset.world_hashes).map(([key, value]) => (
              <li key={key}>
                {key}: <span className="hash">{value}</span>
              </li>
            ))}
          </ul>
          </div>

          <div>
          <h3>Format problems (schema)</h3>
          {dataset.schema_errors.length === 0 ? (
            <p>The format checks found no problems.</p>
          ) : (
            <ul className="list-plain">
              {dataset.schema_errors.map((issue) => (
                <li key={`${issue.example_id}-${issue.path}`}>
                  <StatusBadge tone={issue.severity === "error" ? "fail" : "warn"}>
                    {issue.severity}
                  </StatusBadge>{" "}
                  <code>{issue.example_id}</code>. {issue.path}: {issue.message}
                </li>
              ))}
            </ul>
          )}

          <h3 style={{ marginTop: "1rem" }}>
            Checks for copied test examples (leakage)
          </h3>
          <p>
            These checks look for test examples that also appear in the teaching data.
            A copied example would make the score look better than it is.
          </p>
          <ul className="list-plain">
            {dataset.leakage_checks.map((check) => (
              <li key={check.check_id}>
                <StatusBadge tone={check.passed ? "pass" : "fail"}>
                  {check.passed ? "Passed" : "Failed"}
                </StatusBadge>{" "}
                <code>{check.check_id}</code>. {check.detail}
              </li>
            ))}
          </ul>
          </div>
        </div>
      </details>
    </section>
  );
}

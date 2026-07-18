import Link from "next/link";
import { ErrorBanner } from "@/components/ErrorBanner";
import { StatusBadge, proofTone } from "@/components/StatusBadge";
import type { ErrorPayload, ProofReportView } from "@/lib/types";

function formatIndex(value: number | null): string {
  return value === null ? "Not available" : value.toFixed(3);
}

function proofStatusLabel(status: ProofReportView["proof_status"]): string {
  const labels: Record<ProofReportView["proof_status"], string> = {
    proved: "Passed",
    do_not_distill: "Keep the current model",
    failed_quality: "Accuracy missed",
    failed_economics: "Cost target missed",
    insufficient_evidence: "More proof needed",
  };
  return labels[status];
}

function candidateLabel(armId: string): string {
  const labels: Record<string, string> = {
    rules: "Policy rules",
    teacher: "Source model",
    student_base: "Current base model",
    cheap_off_the_shelf: "Low-cost hosted model",
    oracle_sft: "Known-answer comparison",
    sequence_kd: "Answer distillation",
    logit_kd: "Score distillation",
    ce_ablation: "Plain training comparison",
  };
  return labels[armId] ?? "Additional comparison";
}

export function ProveStage({
  proof,
  error,
}: {
  proof: ProofReportView | null;
  error: ErrorPayload | null;
}) {
  if (!proof) {
    return (
      <section aria-labelledby="prove-heading">
        <div className="panel">
          <p className="text-kicker text-[var(--orange)]">Prove</p>
          <h1 id="prove-heading">Check the result</h1>
          <ErrorBanner error={error} />
          <p data-testid="prove-empty">
            There is no result to check yet. Start from the project setup to prepare
            the data and spending limit.
          </p>
          <Link href="/" className="btn btn-primary w-fit">
            Set up a run
          </Link>
        </div>
      </section>
    );
  }

  const servingEconomicsLabel = proof.economics.serving_cost_projected
    ? "Estimated running cost"
    : "Measured running cost";
  const servingValueLabel = proof.economics.serving_cost_projected
    ? "Estimated cost per request"
    : "Measured cost per request";
  const decision =
    proof.proof_status === "proved"
      ? "Use the smaller model for the next rollout check"
      : proof.proof_status === "do_not_distill"
        ? "Keep the current model"
        : "Do not promote this smaller model";
  const why =
    proof.proof_status === "proved"
      ? "It met the saved accuracy and cost checks."
      : proof.first_failed_gate
        ? "It missed at least one required check."
        : "The saved result is not strong enough to recommend a switch.";

  return (
    <section aria-labelledby="prove-heading">
      <div className="panel">
        <p className="text-kicker text-[var(--orange)]">Prove</p>
        <h1 id="prove-heading">Check the result</h1>
        <p>
          The smaller model must stay accurate on familiar and unfamiliar examples.
          It also needs to run fast enough and cost less. The saved check plan fixes
          those rules before training.
        </p>
        <ErrorBanner error={error} />
        <div
          className="mb-4 grid gap-3 rounded-[14px] border border-border bg-secondary/35 p-4"
          data-testid="proof-decision"
        >
          <div>
            <p className="text-kicker text-[var(--orange)]">Decision</p>
            <h3 className="mt-1">{decision}</h3>
            <p>{why}</p>
            <p className="text-sm text-muted-foreground">
              Confidence comes from a saved demo run. No live model is running now.
            </p>
          </div>
          <div className="grid gap-2 sm:grid-cols-3">
            <div className="stat">
              <span className="label">Quality</span>
              <span className="value">
                {proof.economics.quality_retention === null
                  ? "Not available"
                  : `${Math.round(proof.economics.quality_retention * 100)}% kept`}
              </span>
            </div>
            <div className="stat">
              <span className="label">Speed</span>
              <span className="value">
                {proof.systems
                  ? `95% within ${proof.systems.p95_latency_ms} ms`
                  : "Not measured"}
              </span>
            </div>
            <div className="stat">
              <span className="label">Cost</span>
              <span className="value">
                ${proof.economics.gross_experiment_cost_usd.toFixed(2)} run
              </span>
            </div>
          </div>
        </div>
        <div className="meta-row">
          <StatusBadge tone={proofTone(proof.proof_status)}>
            {proofStatusLabel(proof.proof_status)}
          </StatusBadge>
          {proof.precomputed ? (
            <StatusBadge tone="precomputed">Saved run</StatusBadge>
          ) : null}
          <StatusBadge
            tone={proof.economics.serving_cost_projected ? "projected" : "pass"}
          >
            {servingEconomicsLabel}
          </StatusBadge>
        </div>
        <details className="rounded-[14px] border border-border p-3">
          <summary className="min-h-11 cursor-pointer py-2 font-medium">
            Advanced record details
          </summary>
          <div className="mt-3 grid gap-2 text-sm">
            <p>
              Result ID: <code>{proof.report_id}</code>
            </p>
            <p>
              Check plan fingerprint: <code>{proof.protocol_sha256}</code>
            </p>
            {proof.first_failed_gate ? (
              <p>
                First failed check: <code>{proof.first_failed_gate}</code>
              </p>
            ) : (
              <p>The saved result does not record a failed check.</p>
            )}
            {proof.unevaluated_gates.length > 0 ? (
              <p>
                Checks that did not run:{" "}
                {proof.unevaluated_gates.map((gate) => (
                  <code key={gate} style={{ marginRight: "0.4rem" }}>
                    {gate}
                  </code>
                ))}
              </p>
            ) : null}
          </div>
        </details>
      </div>

      <div className="panel">
        <h3>Compare the candidates</h3>
        <p>
          Each candidate uses a different way to make the smaller model. The score
          includes a range that shows how much the result may vary.
        </p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Candidate</th>
                <th scope="col">Why it is here</th>
                <th scope="col">Main score</th>
                <th scope="col">Likely range (95%)</th>
                <th scope="col">Score kept on unfamiliar examples</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {proof.arms.map((arm) => (
                <tr key={arm.arm_id}>
                  <td>
                    {candidateLabel(arm.arm_id)}
                  </td>
                  <td>{arm.purpose}</td>
                  <td>{formatIndex(arm.primary_index)}</td>
                  <td>
                    {arm.ci_low === null || arm.ci_high === null
                      ? "Not available"
                      : `[${arm.ci_low.toFixed(3)}, ${arm.ci_high.toFixed(3)}]`}
                  </td>
                  <td>{formatIndex(arm.ood_retention)}</td>
                  <td>
                    {arm.excluded ? (
                      <StatusBadge tone="unavailable">Not compared</StatusBadge>
                    ) : (
                      <StatusBadge tone="pass">Compared</StatusBadge>
                    )}
                    {arm.exclusion_reason ? (
                      <div style={{ marginTop: "0.35rem" }}>{arm.exclusion_reason}</div>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <details className="panel">
          <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
            Advanced system measurements
          </summary>
          <p>
            These numbers show response time, capacity, memory use, and machine time.
            They matter when you estimate the cost of serving the model.
          </p>
          {proof.systems ? (
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th scope="col">Measurement</th>
                    <th scope="col">Value</th>
                    <th scope="col">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["p50 latency", `${proof.systems.p50_latency_ms} ms`],
                    ["p95 latency", `${proof.systems.p95_latency_ms} ms`],
                    [
                      "Throughput batch1",
                      `${proof.systems.throughput_rps_batch1} rps`,
                    ],
                    [
                      "Throughput batch8",
                      `${proof.systems.throughput_rps_batch8} rps`,
                    ],
                    ["Peak VRAM", `${proof.systems.peak_vram_gib} GiB`],
                    ["GPU hours", String(proof.systems.gpu_hours)],
                    ["Hardware", proof.systems.hardware],
                  ].map(([metric, value]) => (
                    <tr key={metric}>
                      <td>{metric}</td>
                      <td>{value}</td>
                      <td>
                        {proof.systems?.measurement_source ===
                        "precomputed_prior_run" ? (
                          <StatusBadge tone="precomputed">
                            Saved earlier run
                          </StatusBadge>
                        ) : (
                          <StatusBadge tone="pending">
                            Saved sample
                          </StatusBadge>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p>This result does not include system measurements.</p>
          )}
        </details>

        <div className="panel">
          <h3>
            What it cost{" "}
            <StatusBadge
              tone={proof.economics.serving_cost_projected ? "projected" : "pass"}
            >
              {proof.economics.serving_cost_projected ? "Estimate" : "Measured"}
            </StatusBadge>
          </h3>
          <ul className="list-plain">
            <li>
              Total experiment cost: ${proof.economics.gross_experiment_cost_usd.toFixed(2)}
            </li>
            <li>
              Accuracy kept from the source model:{" "}
              {proof.economics.quality_retention === null
                ? "Not available"
                : proof.economics.quality_retention.toFixed(3)}
            </li>
            <li>
              Gap recovered from the source model:{" "}
              {proof.economics.recovered_teacher_gap === null
                ? "Not available"
                : proof.economics.recovered_teacher_gap.toFixed(3)}
            </li>
            <li>
              Requests needed to earn back the run cost:{" "}
              {proof.economics.break_even_requests === null
                ? "Not available"
                : String(proof.economics.break_even_requests)}
            </li>
          </ul>
          <p>{proof.economics.note}</p>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Machine use</th>
                  <th scope="col">{servingValueLabel}</th>
                </tr>
              </thead>
              <tbody>
                {proof.economics.utilization_sensitivity.map((row) => (
                  <tr key={row.utilization}>
                    <td>{Math.round(row.utilization * 100)}%</td>
                    <td>${row.cost_per_request_usd.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>What this result does not prove</h3>
          <ul className="list-plain">
            {proof.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <details className="panel">
          <summary className="min-h-11 cursor-pointer py-3 font-serif text-xl">
            Advanced saved evidence files
          </summary>
          <p>
            These names and fingerprints identify the exact files used for this
            result.
          </p>
          <ul className="list-plain">
            {proof.artifact_downloads.map((item) => (
              <li key={item.name}>
                <strong>{item.name}</strong>
                <br />
                <code>{item.uri}</code>
                <br />
                <span className="hash">{item.sha256}</span>
              </li>
            ))}
          </ul>
        </details>
      </div>
    </section>
  );
}
